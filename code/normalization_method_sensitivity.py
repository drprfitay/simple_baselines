#!/usr/bin/env python3

r"""
Normalization sensitivity with two operation modes.

--mode transfer (default): existing MLP controlled-extrapolation workflow.
--mode reconstruction: existing bidirectional OHE/ESM Ridge reconstruction audit.
Each mode keeps its original arguments, defaults, cache handling and output files.
Transfer uses a 20-amino-acid OHE; reconstruction uses observed states or a
supplied --onehot_features_pt. Reconstruction requires a row-aligned raw cache.

Examples:
    python normalization_method_sensitivity.py --mode transfer --dataset_name pard3
    python normalization_method_sensitivity.py --mode reconstruction \
        --dataset_name pard3 --embedding_cache raw_embeddings.pt
Use --mode reconstruction --help for reconstruction-specific arguments.

Transfer workflow:
Normalization sanity check for PARD3 or HIS5 using ESM2-8M.

PARD3:
    Regression, model selection and evaluation by Spearman correlation.

HIS5:
    Classification, model selection and evaluation by ROC AUC.
    If activity is already binary, it is used directly.
    Otherwise the default classification rule is activity > mean(activity).

For each K:
    train/validation pool = num_muts < K
    80/20 internal train/validation split
    select a very small MLP architecture
    refit on all num_muts < K
    test on all num_muts >= K

Representations:
    1. legacy position-axis L2 -> mean -> pooled L2
    2. embedding-axis L2 -> mean -> pooled L2
    3. mean -> pooled L2
    4. raw mean
    5. one-hot encoding
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scipy.stats import spearmanr
from scipy.linalg import LinAlgWarning
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


DATASET_CONFIG = {
    "pard3": {"relative_path": "data/pard3/pard3.csv", "task": "regression"},
    "his5": {"relative_path": "data/his5/his5.csv", "task": "classification"},
}


def safe_spearman(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    rho = spearmanr(y_true, y_pred).statistic
    return float(rho) if rho is not None and np.isfinite(rho) else np.nan


def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(y_true) < 2 or len(np.unique(y_true)) != 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def infer_wt_and_positions(df, sequence_col, num_muts_col):
    nmuts = df[num_muts_col].astype(int).to_numpy()
    wt_indices = np.where(nmuts == 0)[0]

    if len(wt_indices) != 1:
        raise ValueError(f"Expected exactly one WT row with num_muts == 0, found {len(wt_indices)}")

    wt_idx = int(wt_indices[0])
    seqs = df[sequence_col].astype(str).tolist()
    wt = seqs[wt_idx]

    if any(len(seq) != len(wt) for seq in seqs):
        raise ValueError("All sequences must have the same length")

    seq_array = np.asarray([list(seq) for seq in seqs], dtype="U1")
    wt_array = np.asarray(list(wt), dtype="U1")
    varying_positions = np.where(np.any(seq_array != wt_array[None, :], axis=0))[0]

    if len(varying_positions) == 0:
        raise ValueError("No varying positions detected")

    calculated_nmuts = np.sum(seq_array[:, varying_positions] != wt_array[varying_positions][None, :], axis=1)

    if not np.array_equal(calculated_nmuts, nmuts):
        mismatch = np.where(calculated_nmuts != nmuts)[0]
        print(f"[warning] {len(mismatch)} rows have num_muts inconsistent with differences from WT")
        print("[warning] using the dataset num_muts column for splitting")

    return wt_idx, wt, varying_positions


def load_plm_base(repo_root):
    code_dir = Path(repo_root) / "code"

    if not code_dir.is_dir():
        raise FileNotFoundError(f"Could not find code directory: {code_dir}")

    sys.path.insert(0, str(code_dir))
    import plm_base

    plm_base.plm_init(str(Path(repo_root).resolve()))
    return plm_base


def generate_raw_designed_embeddings(df, sequence_col, varying_positions, repo_root, batch_size, device):
    plm_base = load_plm_base(repo_root)

    wrapper = plm_base.plmEmbeddingModel("esm2_t6_8M_UR50D", emb_only=True, device=device)
    wrapper.eval()
    wrapper.plm.eval()

    encoded = [torch.tensor(wrapper.encode(seq), dtype=torch.long) for seq in df[sequence_col].astype(str)]
    lengths = {int(x.numel()) for x in encoded}

    if len(lengths) != 1:
        raise ValueError(f"Tokenized sequences do not all have the same length: {sorted(lengths)}")

    tokens = torch.stack(encoded, dim=0)
    seq_len = len(df[sequence_col].iloc[0])

    if tokens.shape[1] != seq_len + 2:
        raise ValueError(f"Expected ESM token length sequence_length + 2, got {tokens.shape[1]} vs {seq_len + 2}")

    # ESM token zero is <cls>, so sequence position zero is token one.
    token_positions = torch.as_tensor(varying_positions + 1, dtype=torch.long, device=device)

    chunks = []

    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            end = min(start + batch_size, len(tokens))
            batch = tokens[start:end].to(device)
            hidden = wrapper(batch)
            selected = hidden.index_select(1, token_positions).float().cpu()
            chunks.append(selected)
            print(f"[embedding] {end}/{len(tokens)}", flush=True)

    return torch.cat(chunks, dim=0)


def make_embedding_features(raw):
    """
    raw shape = [N, P, D]
    returned features = [N, D]
    """

    raw = raw.float()

    # Legacy: normalize each latent dimension across positions.
    legacy = F.normalize(raw, p=2, dim=1).mean(dim=1)
    legacy = F.normalize(legacy, p=2, dim=-1)

    # More conventional: normalize each position vector across embedding dimension.
    embedding_axis = F.normalize(raw, p=2, dim=-1).mean(dim=1)
    embedding_axis = F.normalize(embedding_axis, p=2, dim=-1)

    # Mean first, then normalize pooled vector.
    mean_then_l2 = F.normalize(raw.mean(dim=1), p=2, dim=-1)

    # No L2 normalization.
    raw_mean = raw.mean(dim=1)

    return {
        "legacy_position_axis_then_pooled_l2": legacy.numpy(),
        "embedding_axis_then_pooled_l2": embedding_axis.numpy(),
        "mean_then_pooled_l2": mean_then_l2.numpy(),
        "raw_mean": raw_mean.numpy(),
    }


def make_ohe_features(df, sequence_col, varying_positions):
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}
    seqs = df[sequence_col].astype(str).tolist()

    X = np.zeros((len(seqs), len(varying_positions) * len(alphabet)), dtype=np.float32)

    for row, seq in enumerate(seqs):
        for p_idx, seq_pos in enumerate(varying_positions):
            aa = seq[int(seq_pos)]

            if aa not in aa_to_idx:
                raise ValueError(f"Unsupported amino acid {aa!r} at row {row}, position {seq_pos}")

            X[row, p_idx * len(alphabet) + aa_to_idx[aa]] = 1.0

    return X


def candidate_configs():
    return [
        {"hidden_layer_sizes": (50,), "alpha": 0.1, "learning_rate_init": 1e-3},
        {"hidden_layer_sizes": (100,), "alpha": 0.1, "learning_rate_init": 1e-3},
        {"hidden_layer_sizes": (100, 20), "alpha": 0.1, "learning_rate_init": 1e-3},
    ]


def fit_model(X_train, y_train, config, seed, task):
    common_kwargs = {
        "hidden_layer_sizes": config["hidden_layer_sizes"],
        "activation": "relu",
        "solver": "adam",
        "alpha": config["alpha"],
        "learning_rate_init": config["learning_rate_init"],
        "max_iter": 500,
        "random_state": seed,
        "early_stopping": False,
    }

    if task == "regression":
        estimator = MLPRegressor(**common_kwargs)
    elif task == "classification":
        estimator = MLPClassifier(**common_kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")

    model = Pipeline([("scale", StandardScaler()), ("mlp", estimator)])
    model.fit(X_train, y_train)

    return model


def predict_model(model, X, task):
    if task == "regression":
        return model.predict(X)

    if task == "classification":
        return model.predict_proba(X)[:, 1]

    raise ValueError(f"Unknown task: {task}")


def score_model(y_true, predictions, task):
    if task == "regression":
        return safe_spearman(y_true, predictions)

    if task == "classification":
        return safe_auc(y_true, predictions)

    raise ValueError(f"Unknown task: {task}")


def split_lower_pool(lower_idx, y, task, val_fraction, seed):
    """
    Train and validation both come exclusively from num_muts < K.
    """

    if task == "regression":
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(lower_idx)

        n_val = max(1, int(round(len(lower_idx) * val_fraction)))
        n_val = min(n_val, len(lower_idx) - 1)

        return shuffled[n_val:], shuffled[:n_val]

    y_lower = np.asarray(y)[lower_idx].astype(int)

    if len(np.unique(y_lower)) != 2:
        return None, None

    class_counts = np.bincount(y_lower, minlength=2)

    if class_counts.min() < 2:
        return None, None

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_local, val_local = next(splitter.split(np.zeros(len(lower_idx)), y_lower))

    return lower_idx[train_local], lower_idx[val_local]


def controlled_extrapolation(X, y, nmuts, representation, task, seed=42, val_fraction=0.2):
    rows = []
    max_order = int(np.max(nmuts))

    for K in range(2, max_order + 1):
        lower_idx = np.where(nmuts < K)[0]
        test_idx = np.where(nmuts >= K)[0]

        if len(lower_idx) < 10 or len(test_idx) < 2:
            continue

        train_idx, val_idx = split_lower_pool(lower_idx, y, task, val_fraction, seed + K)

        if train_idx is None:
            print(f"[skip] K={K}, {representation}: insufficient class diversity in num_muts < K")
            continue

        best = None

        for config_idx, config in enumerate(candidate_configs()):
            model = fit_model(X[train_idx], y[train_idx], config, seed + 1000 * K + config_idx, task)
            val_pred = predict_model(model, X[val_idx], task)
            val_metric = score_model(y[val_idx], val_pred, task)
            score = -np.inf if not np.isfinite(val_metric) else val_metric

            if best is None or score > best[0]:
                best = (score, config_idx, config, val_metric)

        _, best_config_idx, best_config, val_metric = best

        # Refit selected architecture on all lower-order variants.
        final_model = fit_model(X[lower_idx], y[lower_idx], best_config, seed + 100000 + K, task)

        test_pred = predict_model(final_model, X[test_idx], task)
        pooled_metric = score_model(y[test_idx], test_pred, task)

        common = {
            "representation": representation,
            "task": task,
            "K": int(K),
            "train_rule": f"num_muts < {K}",
            "test_rule": f"num_muts >= {K}",
            "n_lower": int(len(lower_idx)),
            "n_train_selection": int(len(train_idx)),
            "n_val_selection": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "selected_config_index": int(best_config_idx),
            "selected_hidden_layers": str(best_config["hidden_layer_sizes"]),
            "validation_metric": val_metric,
            "test_scope": "all_ge_K",
            "test_order": -1,
            "test_metric": pooled_metric,
        }

        if task == "classification":
            common["positive_rate_lower"] = float(np.mean(y[lower_idx]))
            common["positive_rate_train_selection"] = float(np.mean(y[train_idx]))
            common["positive_rate_val_selection"] = float(np.mean(y[val_idx]))
            common["positive_rate_test"] = float(np.mean(y[test_idx]))

        rows.append(common)

        for test_order in sorted(np.unique(nmuts[test_idx])):
            idx = test_idx[nmuts[test_idx] == test_order]
            exact_pred = predict_model(final_model, X[idx], task)
            exact_metric = score_model(y[idx], exact_pred, task)

            row = {
                **common,
                "n_test": int(len(idx)),
                "test_scope": "exact_order",
                "test_order": int(test_order),
                "test_metric": exact_metric,
            }

            if task == "classification":
                row["positive_rate_test"] = float(np.mean(y[idx]))

            rows.append(row)

    return pd.DataFrame(rows)


def make_target(df, dataset_name, activity_col, classification_mode, classification_threshold, classification_percentile):
    activity = pd.to_numeric(df[activity_col], errors="coerce").to_numpy(dtype=float)

    if dataset_name == "pard3":
        return activity, {"task": "regression", "activity_column": activity_col}

    unique = np.unique(activity[np.isfinite(activity)])

    if classification_mode == "auto":
        if len(unique) == 2 and set(unique).issubset({0.0, 1.0}):
            classification_mode = "binary"
        else:
            classification_mode = "mean"

    if classification_mode == "binary":
        if len(unique) != 2:
            raise ValueError(f"--classification_mode binary requires exactly two unique values, found {unique}")

        if set(unique).issubset({0.0, 1.0}):
            y = activity.astype(int)
            mapping = {"0": 0, "1": 1}
        else:
            ordered = np.sort(unique)
            y = (activity == ordered[1]).astype(int)
            mapping = {str(ordered[0]): 0, str(ordered[1]): 1}

        spec = {"task": "classification", "mode": "binary", "activity_column": activity_col, "mapping": mapping}

    elif classification_mode == "mean":
        threshold = float(np.mean(activity))
        y = (activity > threshold).astype(int)
        spec = {"task": "classification", "mode": "activity_above_mean", "activity_column": activity_col, "threshold": threshold}

    elif classification_mode == "median":
        threshold = float(np.median(activity))
        y = (activity > threshold).astype(int)
        spec = {"task": "classification", "mode": "activity_above_median", "activity_column": activity_col, "threshold": threshold}

    elif classification_mode == "percentile":
        threshold = float(np.percentile(activity, classification_percentile))
        y = (activity > threshold).astype(int)
        spec = {
            "task": "classification",
            "mode": "activity_above_percentile",
            "activity_column": activity_col,
            "percentile": float(classification_percentile),
            "threshold": threshold,
        }

    elif classification_mode == "fixed":
        if classification_threshold is None:
            raise ValueError("--classification_mode fixed requires --classification_threshold")

        threshold = float(classification_threshold)
        y = (activity > threshold).astype(int)
        spec = {
            "task": "classification",
            "mode": "activity_above_fixed_threshold",
            "activity_column": activity_col,
            "threshold": threshold,
        }

    else:
        raise ValueError(f"Unknown classification mode: {classification_mode}")

    if len(np.unique(y)) != 2:
        raise ValueError("HIS5 classification rule produced only one class")

    values, counts = np.unique(y, return_counts=True)
    spec["positive_rate"] = float(np.mean(y))
    spec["class_counts"] = {str(int(k)): int(v) for k, v in zip(values, counts)}

    return y.astype(int), spec


def run_transfer(argv=None):
    parser = argparse.ArgumentParser(description="Transfer normalization sensitivity")
    parser.add_argument("--mode", choices=["transfer", "reconstruction"], default="transfer",
                        help="Operation mode (default: transfer)")

    parser.add_argument("--dataset_name", choices=["pard3", "his5"], default="pard3")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--sequence_col", default="full_seq")
    parser.add_argument("--activity_col", default="activity")
    parser.add_argument("--num_muts_col", default="num_muts")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--raw_embedding_cache", default=None)
    parser.add_argument("--reuse_embeddings", action="store_true")

    parser.add_argument("--classification_mode", choices=["auto", "binary", "mean", "median", "percentile", "fixed"], default="auto")
    parser.add_argument("--classification_threshold", type=float, default=None)
    parser.add_argument("--classification_percentile", type=float, default=50.0)

    args = parser.parse_args(argv)

    dataset_cfg = DATASET_CONFIG[args.dataset_name]
    task = dataset_cfg["task"]
    repo_root = Path(args.repo_root).resolve()

    dataset_path = Path(args.dataset).resolve() if args.dataset else repo_root / dataset_cfg["relative_path"]
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"{args.dataset_name}_normalization_sanity")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.raw_embedding_cache) if args.raw_embedding_cache else output_dir / f"{args.dataset_name}_esm2_8m_raw_designed_embeddings.pt"

    print(f"[data] dataset={args.dataset_name}, task={task}")
    print(f"[data] path={dataset_path}")

    df = pd.read_csv(dataset_path)

    for col in [args.sequence_col, args.activity_col, args.num_muts_col]:
        if col not in df.columns:
            raise KeyError(f"Missing required column {col!r}. Available columns: {df.columns.tolist()}")

    activity_numeric = pd.to_numeric(df[args.activity_col], errors="coerce")
    keep = np.isfinite(activity_numeric.to_numpy(dtype=float))

    if not np.all(keep):
        print(f"[data] dropping {int((~keep).sum())} rows with non-finite {args.activity_col}")
        df = df.loc[keep].reset_index(drop=True)

    y, target_spec = make_target(
        df,
        dataset_name=args.dataset_name,
        activity_col=args.activity_col,
        classification_mode=args.classification_mode,
        classification_threshold=args.classification_threshold,
        classification_percentile=args.classification_percentile,
    )

    nmuts = df[args.num_muts_col].astype(int).to_numpy()
    wt_idx, wt_seq, varying_positions = infer_wt_and_positions(df, args.sequence_col, args.num_muts_col)

    print(f"[data] N={len(df)}, WT row={wt_idx}, sequence length={len(wt_seq)}, varying positions={len(varying_positions)}")
    print(f"[data] varying positions zero-based: {varying_positions.tolist()}")
    print(f"[data] mutation-order counts: {pd.Series(nmuts).value_counts().sort_index().to_dict()}")
    print(f"[target] {target_spec}")

    if args.reuse_embeddings:
        payload = torch.load(cache_path, map_location="cpu")
        raw = payload["raw_embeddings"] if isinstance(payload, dict) else payload

        if raw.shape[0] != len(df):
            raise ValueError(f"Cached embeddings contain {raw.shape[0]} rows but current dataset has {len(df)} rows")
        print(f"[embedding] reused {cache_path}: {tuple(raw.shape)}")

    else:
        raw = generate_raw_designed_embeddings(
            df=df,
            sequence_col=args.sequence_col,
            varying_positions=varying_positions,
            repo_root=repo_root,
            batch_size=args.batch_size,
            device=torch.device(args.device),
        )

        torch.save(
            {
                "raw_embeddings": raw,
                "varying_positions_zero_based": varying_positions,
                "wt_sequence": wt_seq,
                "model": "esm2_t6_8M_UR50D",
                "dataset_name": args.dataset_name,
                "n_variants": int(len(df)),
            },
            cache_path,
        )

        print(f"[embedding] saved {cache_path}: {tuple(raw.shape)}")

    features = make_embedding_features(raw)
    features["one_hot_reference"] = make_ohe_features(df, args.sequence_col, varying_positions)

    all_results = []

    for name, X in features.items():
        print(f"[fit] {name}: X={X.shape}", flush=True)

        result = controlled_extrapolation(
            X=X,
            y=y,
            nmuts=nmuts,
            representation=name,
            task=task,
            seed=args.seed,
            val_fraction=args.val_fraction,
        )

        if not result.empty:
            all_results.append(result)

    if not all_results:
        raise RuntimeError("No valid controlled-extrapolation splits were produced")

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(output_dir / "controlled_extrapolation_results.csv", index=False)

    pooled = results[results["test_scope"] == "all_ge_K"].copy()
    summary = pooled.pivot(index="K", columns="representation", values="test_metric").sort_index()

    metric_name = "spearman" if task == "regression" else "roc_auc"
    summary.to_csv(output_dir / f"pooled_test_{metric_name}_by_K.csv")

    legacy_col = "legacy_position_axis_then_pooled_l2"
    deltas = summary.subtract(summary[legacy_col], axis=0)
    deltas.to_csv(output_dir / "delta_vs_legacy_by_K.csv")

    exact = results[results["test_scope"] == "exact_order"].pivot_table(
        index=["K", "test_order"], columns="representation", values="test_metric"
    )
    exact.to_csv(output_dir / f"exact_order_test_{metric_name}.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for column in summary.columns:
        ax.plot(summary.index, summary[column], marker="o", label=column)

    if task == "regression":
        ax.axhline(0, linewidth=0.8, linestyle="--")
        ax.set_ylabel("Spearman on num_muts >= K")
    else:
        ax.axhline(0.5, linewidth=0.8, linestyle="--")
        ax.set_ylim(0, 1)
        ax.set_ylabel("ROC AUC on num_muts >= K")

    ax.set_xlabel("K, with training/validation on num_muts < K")
    ax.set_title(f"{args.dataset_name.upper()} ESM2-8M normalization sanity check")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / f"pooled_test_{metric_name}_by_K.png", dpi=200)
    plt.close(fig)

    metadata = {
        "dataset_name": args.dataset_name,
        "dataset": str(dataset_path),
        "task": task,
        "target_spec": target_spec,
        "n_variants": int(len(df)),
        "wt_index": int(wt_idx),
        "sequence_length": int(len(wt_seq)),
        "varying_positions_zero_based": [int(x) for x in varying_positions],
        "mutation_order_counts": {str(int(k)): int(v) for k, v in pd.Series(nmuts).value_counts().sort_index().items()},
        "normalizations": list(features.keys()),
        "split": "80/20 model selection within num_muts < K, refit on all num_muts < K, test on num_muts >= K",
        "architecture_grid": [
            {**config, "hidden_layer_sizes": list(config["hidden_layer_sizes"])}
            for config in candidate_configs()
        ],
        "seed": int(args.seed),
        "validation_fraction": float(args.val_fraction),
        "metric": metric_name,
    }

    with open(output_dir / "experiment_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    metric_label = "Spearman" if task == "regression" else "ROC AUC"

    print(f"\n=== Pooled held-out {metric_label}, test num_muts >= K ===")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n=== Delta versus exact legacy normalization ===")
    print(deltas.to_string(float_format=lambda x: f"{x:+.3f}"))

    print(f"\nWrote results to {output_dir.resolve()}")


ALPHAS = np.logspace(-2, 4, 13)


def infer_reconstruction_wt_and_positions(df, sequence_col="full_seq", num_muts_col="num_muts"):
    nmuts = df[num_muts_col].astype(int).to_numpy()
    wt_idx = np.where(nmuts == 0)[0]
    if len(wt_idx) != 1:
        raise ValueError(f"Expected exactly one WT row with num_muts == 0, found {len(wt_idx)}")
    wt_idx = int(wt_idx[0])
    seqs = df[sequence_col].astype(str).tolist()
    wt = seqs[wt_idx]
    if any(len(seq) != len(wt) for seq in seqs):
        raise ValueError("All sequences must have the same length")
    seq_array = np.asarray([list(seq) for seq in seqs], dtype="U1")
    wt_array = np.asarray(list(wt), dtype="U1")
    varying_positions = np.where(np.any(seq_array != wt_array[None, :], axis=0))[0]
    if len(varying_positions) == 0:
        raise ValueError("No varying positions detected")
    calculated_nmuts = np.sum(seq_array[:, varying_positions] != wt_array[varying_positions][None, :], axis=1)
    if not np.array_equal(calculated_nmuts, nmuts):
        mismatch = np.where(calculated_nmuts != nmuts)[0]
        print(f"[warning] {len(mismatch)} rows have num_muts inconsistent with differences from WT")
    return wt_idx, wt, varying_positions


def load_onehot_from_features_pt(path, n_rows):
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if "onehot" not in payload:
        raise KeyError(f"No 'onehot' key in {path}")
    ohe = payload["onehot"]
    if hasattr(ohe, "detach"):
        ohe = ohe.detach().cpu().numpy()
    ohe = np.asarray(ohe, dtype=np.float32).reshape(n_rows, -1)
    if ohe.shape[0] != n_rows or not np.isfinite(ohe).all():
        raise ValueError(f"Invalid onehot matrix loaded from {path}: shape={ohe.shape}")
    return ohe


def make_reconstruction_ohe_features(df, varying_positions, sequence_col="full_seq"):
    seqs = df[sequence_col].astype(str).tolist()
    pieces, labels = [], []
    for pos in varying_positions:
        values = np.asarray([seq[int(pos)] for seq in seqs])
        categories = sorted(np.unique(values).tolist())
        for aa in categories:
            pieces.append((values == aa).astype(np.float32)[:, None])
            labels.append(f"{int(pos)}_{aa}")
    return np.concatenate(pieces, axis=1), labels


def resolve_cache(dataset_name, repo_root, explicit_cache=None):
    if explicit_cache is not None:
        path = Path(explicit_cache).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    filename_candidates = [
        f"{dataset_name}_esm2_8m_raw_designed_embeddings.pt",
        f"{dataset_name}_esm2_8m_raw_designed_position_embeddings.pt",
        "esm2_8m_raw_designed_position_embeddings.pt",
    ]
    directory_candidates = [
        Path.cwd() / f"{dataset_name}_normalization_sanity",
        Path(repo_root) / f"{dataset_name}_normalization_sanity",
        Path(repo_root) / "notebooks" / f"{dataset_name}_normalization_sanity",
    ]
    candidates = [directory / filename for directory in directory_candidates for filename in filename_candidates]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    tried = "\n  ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"Could not find cached raw embeddings. Pass --embedding_cache explicitly. Tried:\n  {tried}")


def load_raw_cache(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        raw = payload.get("raw_embeddings")
        if raw is None:
            tensor_values = [v for v in payload.values() if isinstance(v, torch.Tensor) and v.ndim == 3]
            if len(tensor_values) != 1:
                raise ValueError(f"Could not identify a unique [N,P,D] raw embedding tensor in {path}")
            raw = tensor_values[0]
        metadata = {k: v for k, v in payload.items() if k != "raw_embeddings"}
    else:
        raw, metadata = payload, {}
    raw = torch.as_tensor(raw).float().cpu()
    if raw.ndim != 3:
        raise ValueError(f"Expected raw embedding cache shape [N,P,D], got {tuple(raw.shape)}")
    if not torch.isfinite(raw).all():
        raise ValueError("Raw embedding cache contains NaN/Inf")
    return raw, metadata


def global_r2(y, y_hat):
    numerator = np.square(y - y_hat).sum()
    denominator = np.square(y - y.mean(axis=0)).sum()
    return float(1.0 - numerator / denominator) if denominator > 0 else float("nan")


def make_ridge_model(use_fixed_alpha=False, fixed_alpha=100.0, ridge_cv_folds=5):
    if use_fixed_alpha:
        return make_pipeline(StandardScaler(), Ridge(alpha=fixed_alpha))
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=ridge_cv_folds))


def model_alpha(model, use_fixed_alpha=False, fixed_alpha=100.0):
    return float(fixed_alpha) if use_fixed_alpha else float(model.named_steps["ridgecv"].alpha_)


def reconstruction_r2(x, y, train_idx, test_idx, use_fixed_alpha=False, fixed_alpha=100.0, ridge_cv_folds=5):
    model = make_ridge_model(use_fixed_alpha, fixed_alpha, ridge_cv_folds)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=LinAlgWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model.fit(x[train_idx], y[train_idx])
    y_hat = model.predict(x[test_idx])
    y_test = y[test_idx]
    scores = {
        "global_r2": global_r2(y_test, y_hat),
        "weighted_r2": float(r2_score(y_test, y_hat, multioutput="variance_weighted")),
    }
    return scores, model_alpha(model, use_fixed_alpha, fixed_alpha)


def geometry_distance_spearman(ohe, feat):
    ohe_dist = pdist(ohe, metric="hamming")
    feat_dist = pdist(feat, metric="cosine")
    keep = np.isfinite(ohe_dist) & np.isfinite(feat_dist)
    if keep.sum() < 2 or np.std(ohe_dist[keep]) == 0 or np.std(feat_dist[keep]) == 0:
        return float("nan")
    return float(spearmanr(ohe_dist[keep], feat_dist[keep]).statistic)


def run_round(ohe, features, sample, train_idx, test_idx, iteration, dataset_name, include_geometry, use_fixed_alpha, fixed_alpha, ridge_cv_folds):
    rows = []
    for feature_name, feat in features.items():
        row = {"dataset": dataset_name, "iteration": iteration, "representation": feature_name, "sample_size": len(sample), "train_size": len(train_idx), "test_size": len(test_idx)}
        if include_geometry:
            row["geometry_distance_spearman"] = geometry_distance_spearman(ohe[sample], feat[sample])
        for prefix, x, y in (("ohe_to_feature", ohe, feat), ("feature_to_ohe", feat, ohe)):
            scores, alpha = reconstruction_r2(x, y, train_idx, test_idx, use_fixed_alpha, fixed_alpha, ridge_cv_folds)
            row[f"{prefix}_alpha"] = alpha
            row[f"{prefix}_global_r2"] = scores["global_r2"]
            row[f"{prefix}_weighted_r2"] = scores["weighted_r2"]
        rows.append(row)
    return rows


def summarize(results):
    metrics = [c for c in results.columns if c.endswith("_r2") or c == "geometry_distance_spearman"]
    summary_rows = []
    for representation, sdf in results.groupby("representation"):
        row = {"representation": representation}
        for metric in metrics:
            values = sdf[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric}_std"] = float(np.std(values)) if len(values) else np.nan
        for direction in ("ohe_to_feature", "feature_to_ohe"):
            alpha_values = sdf[f"{direction}_alpha"].to_numpy(dtype=float)
            row[f"{direction}_alpha_median"] = float(np.median(alpha_values))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def print_summary(summary, include_geometry):
    columns = ["representation"]
    if include_geometry:
        columns.append("geometry_distance_spearman_mean")
    columns += ["ohe_to_feature_weighted_r2_mean", "feature_to_ohe_weighted_r2_mean", "ohe_to_feature_global_r2_mean", "feature_to_ohe_global_r2_mean"]
    display = summary[columns].copy()
    rename = {
        "geometry_distance_spearman_mean": "geom_rho",
        "ohe_to_feature_weighted_r2_mean": "ohe->feat weighted R2",
        "feature_to_ohe_weighted_r2_mean": "feat->ohe weighted R2",
        "ohe_to_feature_global_r2_mean": "ohe->feat global R2",
        "feature_to_ohe_global_r2_mean": "feat->ohe global R2",
    }
    print(display.rename(columns=rename).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def run_reconstruction(argv=None):
    parser = argparse.ArgumentParser(description="Reconstruction normalization sensitivity")
    parser.add_argument("--mode", choices=["transfer", "reconstruction"], default="reconstruction",
                        help="Operation mode (default: transfer in the combined script)")
    parser.add_argument("--dataset_name", choices=["pard3", "his5"], required=True)
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--embedding_cache", default=None)
    parser.add_argument("--onehot_features_pt", default=None, help="Optional old features.pt; if supplied, use its exact onehot key instead of regenerating OHE")
    parser.add_argument("--sequence_col", default="full_seq")
    parser.add_argument("--num_muts_col", default="num_muts")
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--n_rounds", type=int, default=100)
    parser.add_argument("--test_frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge_cv_folds", type=int, default=5)
    parser.add_argument("--fixed_alpha", type=float, default=None, help="If supplied, use Ridge(alpha=value) instead of RidgeCV")
    parser.add_argument("--include_geometry", action="store_true", help="Also repeat the OHE-Hamming vs embedding-cosine geometry analysis")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve() if args.dataset else repo_root / DATASET_CONFIG[args.dataset_name]["relative_path"]
    cache_path = resolve_cache(args.dataset_name, repo_root, args.embedding_cache)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path(f"{args.dataset_name}_reconstruction_normalization_audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    for col in (args.sequence_col, args.num_muts_col):
        if col not in df.columns:
            raise KeyError(f"Missing required column {col!r} in {dataset_path}")

    wt_idx, wt_seq, varying_positions = infer_reconstruction_wt_and_positions(df, args.sequence_col, args.num_muts_col)
    raw, cache_metadata = load_raw_cache(cache_path)
    if raw.shape[0] != len(df):
        raise ValueError(f"Cached embeddings have {raw.shape[0]} rows but dataset has {len(df)} rows. This audit needs a row-aligned cache containing all variants.")
    if raw.shape[1] != len(varying_positions):
        raise ValueError(f"Cached embeddings have {raw.shape[1]} designed positions but dataset has {len(varying_positions)} varying positions")
    cached_positions = cache_metadata.get("varying_positions_zero_based")
    if cached_positions is not None and not np.array_equal(np.asarray(cached_positions, dtype=int), varying_positions):
        raise ValueError("Cached varying-position metadata does not match positions inferred from the current dataset")

    if args.onehot_features_pt:
        ohe = load_onehot_from_features_pt(args.onehot_features_pt, len(df))
        ohe_labels = None
        print(f"[OHE] using exact onehot matrix from {args.onehot_features_pt}")
    else:
        ohe, ohe_labels = make_reconstruction_ohe_features(df, varying_positions, args.sequence_col)
        print("[OHE] regenerated from observed amino-acid states at varying positions")
    features = make_embedding_features(raw)
    n = len(df)
    if args.sample_size > n:
        raise ValueError(f"sample_size={args.sample_size} exceeds dataset size {n}")
    train_size = int(round(args.sample_size * (1.0 - args.test_frac)))
    if train_size <= 0 or train_size >= args.sample_size:
        raise ValueError("test_frac leaves an empty train or test split")
    if args.fixed_alpha is None and train_size < args.ridge_cv_folds:
        raise ValueError("Training split is smaller than ridge_cv_folds")

    print(f"[data] dataset={args.dataset_name}, N={n}, WT={wt_idx}, sequence_length={len(wt_seq)}, varying_positions={len(varying_positions)}")
    print(f"[cache] {cache_path}")
    print(f"[features] raw={tuple(raw.shape)}, OHE={ohe.shape}")
    print(f"[audit] rounds={args.n_rounds}, sample_size={args.sample_size}, train={train_size}, test={args.sample_size - train_size}")
    print(f"[ridge] {'fixed alpha=' + str(args.fixed_alpha) if args.fixed_alpha is not None else 'RidgeCV alphas=' + str(ALPHAS.tolist())}")

    rng = np.random.default_rng(args.seed)
    rows = []
    for iteration in range(1, args.n_rounds + 1):
        sample = rng.choice(n, size=args.sample_size, replace=False)
        rng.shuffle(sample)
        train_idx, test_idx = sample[:train_size], sample[train_size:]
        rows.extend(run_round(ohe, features, sample, train_idx, test_idx, iteration, args.dataset_name, args.include_geometry, args.fixed_alpha is not None, 100.0 if args.fixed_alpha is None else args.fixed_alpha, args.ridge_cv_folds))
        if iteration == 1 or iteration % 10 == 0 or iteration == args.n_rounds:
            print(f"\nround {iteration}/{args.n_rounds}")
            print_summary(summarize(pd.DataFrame(rows)), args.include_geometry)

    results = pd.DataFrame(rows)
    summary = summarize(results)
    results.to_csv(output_dir / "reconstruction_iterations.csv", index=False)
    summary.to_csv(output_dir / "reconstruction_summary.csv", index=False)

    metadata = {
        "dataset_name": args.dataset_name,
        "dataset_path": str(dataset_path),
        "embedding_cache": str(cache_path),
        "n_variants": int(n),
        "raw_embedding_shape": list(raw.shape),
        "ohe_shape": list(ohe.shape),
        "ohe_columns": ohe_labels,
        "onehot_features_pt": args.onehot_features_pt,
        "varying_positions_zero_based": [int(x) for x in varying_positions],
        "representations": list(features.keys()),
        "sample_size": args.sample_size,
        "train_size": train_size,
        "test_size": args.sample_size - train_size,
        "n_rounds": args.n_rounds,
        "seed": args.seed,
        "ridge_cv_folds": args.ridge_cv_folds,
        "alphas": ALPHAS.tolist(),
        "fixed_alpha": args.fixed_alpha,
        "include_geometry": args.include_geometry,
    }
    with open(output_dir / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nfinal")
    print_summary(summary, args.include_geometry)
    print(f"\nSaved results to {output_dir.resolve()}")


def main(argv=None):
    # Choose a mode first, then let its original parser handle all other flags.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["transfer", "reconstruction"], default="transfer")
    args, remaining = parser.parse_known_args(argv)
    if args.mode == "reconstruction":
        run_reconstruction(remaining)
    else:
        run_transfer(remaining)


if __name__ == "__main__":
    main()
