#!/usr/bin/env python3
"""Run one mixed-representation job, or collect completed jobs.

Standalone snapshot of the subset scoring path in FuncScape's
src/experiments/run_scoring_experiment.py. Analysis functions below are copied
unchanged: matched splits/resamples, architecture grid, seeds, tie breaking,
training, metrics, standardization, model checkpoints, and cache formats.
Requires numpy, pandas, torch, scikit-learn, and scipy (optional).

Generate jobs with mixed_representation_analysis_gen_jobs.py, then run a .sh
file, or run_all.sh. Run collect_results.sh after individual jobs finish.
No FuncScape installation or job database is needed.
"""
import argparse
import itertools

import json

import math

import re

from datetime import datetime, timezone

from pathlib import Path

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

try:
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover - exercised only without scipy installed.
    spearmanr = None

DEFAULT_ARCHITECTURE_GRID = [
    {
        "name": "mlp_64",
        "hidden_layers": [64],
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.0,
        "batch_size": 64,
    },
    {
        "name": "mlp_128",
        "hidden_layers": [128],
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "batch_size": 64,
    },
    {
        "name": "mlp_100_100",
        "hidden_layers": [100, 100],
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "batch_size": 64,
    },
    {
        "name": "mlp_256_128",
        "hidden_layers": [256, 128],
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "batch_size": 64,
    },
    {
        "name": "mlp_256",
        "hidden_layers": [256],
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "batch_size": 64,
    },
    {
        "name": "mlp_512",
        "hidden_layers": [512],
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "batch_size": 64,
    },
    {
        "name": "mlp_1024",
        "hidden_layers": [1024],
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "dropout": 0.2,
        "batch_size": 64,
    },
    {
        "name": "mlp_64_64",
        "hidden_layers": [64, 64],
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "batch_size": 64,
    },
    {
        "name": "mlp_128_64",
        "hidden_layers": [128, 64],
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "batch_size": 64,
    },
    {
        "name": "mlp_128_128",
        "hidden_layers": [128, 128],
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "batch_size": 64,
    },
    {
        "name": "mlp_256_256",
        "hidden_layers": [256, 256],
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "batch_size": 64,
    },
    {
        "name": "mlp_512_256",
        "hidden_layers": [512, 256],
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "dropout": 0.15,
        "batch_size": 64,
    },
    {
        "name": "mlp_1024_256",
        "hidden_layers": [1024, 256],
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "dropout": 0.2,
        "batch_size": 64,
    },
    {
        "name": "mlp_64_128",
        "hidden_layers": [64, 128],
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "batch_size": 64,
    },
    {
        "name": "mlp_128_256",
        "hidden_layers": [128, 256],
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "batch_size": 64,
    },
    {
        "name": "mlp_256_512",
        "hidden_layers": [256, 512],
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "dropout": 0.15,
        "batch_size": 64,
    },
    {
        "name": "mlp_64_128_64",
        "hidden_layers": [64, 128, 64],
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "batch_size": 64,
    },
    {
        "name": "mlp_128_256_128",
        "hidden_layers": [128, 256, 128],
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "batch_size": 64,
    },
    {
        "name": "mlp_256_512_256",
        "hidden_layers": [256, 512, 256],
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "dropout": 0.15,
        "batch_size": 64,
    },
    {
        "name": "mlp_512_1024_256",
        "hidden_layers": [512, 1024, 256],
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "dropout": 0.2,
        "batch_size": 64,
    },
]

DEFAULT_CONFIG_GRID = DEFAULT_ARCHITECTURE_GRID

_N_CONFIG_BOOTSTRAPS = 5

_CONFIG_TRAIN_FRACTIONS = (0.15, 0.30, 0.50, 0.70, 0.90)

_RESULT_COLUMNS = [
    "results_file_name",
    "data_path",
    "target_scheme",
    "feature_name",
    "feature_combo",
    "split",
    "subset_fraction",
    "iteration",
    "n_train",
    "n_test",
    "task_type",
    "config_path",
    "architecture_name",
    "architecture_search_path",
    "architecture_model_path",
    "best_epoch",
    "architecture_best_score",
    "architecture_metric",
    "architecture_n_resamples",
    "architecture_n_seeds",
    "architecture_max_epochs",
    "architecture_eval_every",
    "hidden_layers",
    "lr",
    "weight_decay",
    "dropout",
    "epochs",
    "batch_size",
    "pearson",
    "spearman",
    "mse",
    "roc_auc",
    "precision",
    "accuracy",
    "precision_at_k",
    "error",
]

_METRIC_COLUMNS = [
    "pearson",
    "spearman",
    "mse",
    "roc_auc",
    "precision",
    "accuracy",
    "precision_at_k",
]

class _MLPScorer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def _standardize_feature_tensors(x_train, x_eval, enabled=False):
    if not enabled:
        return x_train, x_eval, None, None
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std > 1e-12, std, torch.ones_like(std))
    return (x_train - mean) / std, (x_eval - mean) / std, mean.detach().cpu(), std.detach().cpu()

def resolve_architecture(
    X,
    y,
    task_type,
    architecture_grid=None,
    max_epochs=200,
    eval_every=10,
    resample_fractions=(0.2, 0.5, 0.8),
    n_resamples_per_fraction=5,
    n_seeds=3,
    random_seed=0,
    device=None,
    standardize_features=False,
    prefer_simpler=True,
    indistinguishable_tolerance=0.01,
    save_path=None,
    verbose=True,
    resamples=None,
):
    """Resolve one MLP architecture/epoch by matched train/validation resampling."""

    X = _as_tensor(X, name="X").float()
    y = _as_tensor(y, name="y").reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows, but y has {y.shape[0]}.")
    architecture_grid = [_architecture_config(config, idx) for idx, config in enumerate(
        architecture_grid or DEFAULT_ARCHITECTURE_GRID
    )]
    device = _resolve_device(device)
    checkpoint_epochs = list(range(int(eval_every), int(max_epochs) + 1, int(eval_every)))
    if not checkpoint_epochs or checkpoint_epochs[-1] != int(max_epochs):
        checkpoint_epochs.append(int(max_epochs))
    metric_name = _architecture_metric_name(task_type)
    train_task_type = _training_task_type(task_type)

    if resamples is None:
        resamples = _make_architecture_resamples(
            n_rows=len(y),
            resample_fractions=resample_fractions,
            n_resamples_per_fraction=n_resamples_per_fraction,
            random_seed=random_seed,
        )
    else:
        resamples = _normalize_architecture_resamples(resamples)
    scores = np.full(
        (len(architecture_grid), len(checkpoint_epochs), len(resamples), int(n_seeds)),
        np.nan,
        dtype=np.float32,
    )
    rows = []

    for arch_idx, config in enumerate(architecture_grid):
        if verbose:
            print(f"Architecture search: {config['name']} ({arch_idx + 1}/{len(architecture_grid)})")
        for resample_idx, resample in enumerate(resamples):
            train_idx = np.asarray(resample["train_indices"], dtype=int)
            val_idx = np.asarray(resample["val_indices"], dtype=int)
            for seed_idx in range(int(n_seeds)):
                seed = int(random_seed) + arch_idx * 1_000_000 + resample_idx * 1_000 + seed_idx
                try:
                    result = train_scorer_with_checkpoints(
                        X[train_idx],
                        y[train_idx],
                        X[val_idx],
                        y[val_idx],
                        train_task_type,
                        config,
                        checkpoint_epochs,
                        device,
                        seed,
                        standardize_features=standardize_features,
                    )
                    checkpoint_scores = result["checkpoint_scores"]
                    checkpoint_metrics = result["checkpoint_metrics"]
                except Exception as exc:
                    checkpoint_scores = [float("nan")] * len(checkpoint_epochs)
                    checkpoint_metrics = [{"error": str(exc)} for _ in checkpoint_epochs]
                for ckpt_idx, epoch in enumerate(checkpoint_epochs):
                    score = checkpoint_scores[ckpt_idx]
                    scores[arch_idx, ckpt_idx, resample_idx, seed_idx] = (
                        np.nan if not np.isfinite(score) else float(score)
                    )
                    rows.append(
                        {
                            "architecture_index": arch_idx,
                            "architecture_name": config["name"],
                            "checkpoint_index": ckpt_idx,
                            "epoch": int(epoch),
                            "resample_index": resample_idx,
                            "resample_fraction": float(resample["train_fraction"]),
                            "resample_iteration": int(resample["iteration"]),
                            "seed_index": seed_idx,
                            "seed": seed,
                            "metric_name": metric_name,
                            "score": None if not np.isfinite(score) else float(score),
                            "metrics": json.dumps(_json_safe_metrics(checkpoint_metrics[ckpt_idx])),
                        }
                    )

    median_scores = np.nanmedian(scores, axis=(2, 3))
    if np.all(np.isnan(median_scores)):
        best_arch_idx, best_ckpt_idx = 0, len(checkpoint_epochs) - 1
        best_score = float("nan")
    else:
        best_score = float(np.nanmax(median_scores))
        eligible = np.argwhere(median_scores >= best_score - float(indistinguishable_tolerance))
        if prefer_simpler:
            best_arch_idx, best_ckpt_idx = min(
                ((int(a), int(c)) for a, c in eligible),
                key=lambda item: (
                    len(architecture_grid[item[0]].get("hidden_layers", [])),
                    sum(architecture_grid[item[0]].get("hidden_layers", [])),
                    checkpoint_epochs[item[1]],
                    -float(median_scores[item[0], item[1]]),
                ),
            )
        else:
            best_arch_idx, best_ckpt_idx = max(
                ((int(a), int(c)) for a, c in eligible),
                key=lambda item: float(median_scores[item[0], item[1]]),
            )

    best_config = _json_safe_config(architecture_grid[best_arch_idx])
    best_epoch = int(checkpoint_epochs[best_ckpt_idx])
    score_table = pd.DataFrame(rows)
    result = {
        "best_config": best_config,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "metric_name": metric_name,
        "scores": scores,
        "score_table": score_table,
        "architecture_grid": [_json_safe_config(config) for config in architecture_grid],
        "checkpoint_epochs": [int(epoch) for epoch in checkpoint_epochs],
        "n_seeds": int(n_seeds),
        "max_epochs": int(max_epochs),
        "eval_every": int(eval_every),
        "standardize_features": bool(standardize_features),
        "resample_metadata": [
            {
                "train_fraction": float(item["train_fraction"]),
                "iteration": int(item["iteration"]),
                "train_indices": [int(value) for value in item["train_indices"]],
                "val_indices": [int(value) for value in item["val_indices"]],
            }
            for item in resamples
        ],
    }
    if save_path is not None:
        _save_architecture_search_result(save_path, result)
    if verbose:
        print(
            "Selected architecture "
            f"{best_config.get('name', best_arch_idx)} "
            f"at epoch {best_epoch} "
            f"with {metric_name}={best_score:.4g}"
        )
    return result

def train_scorer_with_checkpoints(
    X_train,
    y_train,
    X_val,
    y_val,
    task_type,
    config,
    checkpoint_epochs,
    device,
    random_seed,
    standardize_features=False,
):
    _set_random_seed(random_seed)
    device = torch.device(_resolve_device(device))
    x_train = _as_tensor(X_train, name="X_train").float().to(device)
    x_val = _as_tensor(X_val, name="X_val").float().to(device)
    x_train, x_val, _, _ = _standardize_feature_tensors(
        x_train,
        x_val,
        enabled=standardize_features,
    )
    y_train_tensor = _as_tensor(y_train, name="y_train")
    y_val_tensor = _as_tensor(y_val, name="y_val")

    if task_type == "classification":
        y_train_device = y_train_tensor.long().to(device)
        output_dim = int(torch.max(torch.cat([y_train_tensor.long(), y_val_tensor.long()])).item()) + 1
        loss_fn = nn.CrossEntropyLoss()
    else:
        y_train_device = y_train_tensor.float().reshape(-1, 1).to(device)
        output_dim = 1
        loss_fn = nn.MSELoss()

    model = _MLPScorer(
        input_dim=x_train.shape[1],
        output_dim=output_dim,
        hidden_layers=config.get("hidden_layers", [64]),
        dropout=config.get("dropout", 0.0),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 0.0),
    )
    batch_size = int(config.get("batch_size", 64))
    checkpoint_epochs = [int(epoch) for epoch in checkpoint_epochs]
    checkpoint_set = set(checkpoint_epochs)
    checkpoint_scores = []
    checkpoint_metrics = []

    for epoch in range(1, max(checkpoint_epochs) + 1):
        model.train()
        order = torch.randperm(x_train.shape[0], device=device)
        for start in range(0, x_train.shape[0], batch_size):
            batch_idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            output = model(x_train[batch_idx])
            loss = loss_fn(output, y_train_device[batch_idx])
            loss.backward()
            optimizer.step()
        if epoch in checkpoint_set:
            model.eval()
            with torch.no_grad():
                output = model(x_val)
                if task_type == "classification":
                    predictions = torch.softmax(output, dim=1).detach().cpu().numpy()
                else:
                    predictions = output.reshape(-1).detach().cpu().numpy()
            metrics = evaluate_predictions(y_val_tensor.detach().cpu().numpy(), predictions, task_type)
            checkpoint_metrics.append(metrics)
            checkpoint_scores.append(_primary_score(metrics, task_type))

    return {
        "checkpoint_scores": checkpoint_scores,
        "checkpoint_metrics": checkpoint_metrics,
    }

def resolve_or_create_scoring_splits(
    data_path,
    target_scheme,
    n_rows,
    train_indices=None,
    test_indices=None,
    all_indices=None,
    train_breaks=6,
    n_iterations=5,
    subset_sizes=None,
    architecture_resample_fractions=(0.15, 0.25, 0.5, 0.75, 0.8),
    architecture_n_resamples_per_fraction=5,
    num_muts=None,
    max_muts_in_train=None,
    random_seed=0,
    save_splits=True,
    refresh_splits=False,
    scoring_experiment_name="scoring_experiment",
):
    split_dir = _scoring_experiment_dir(data_path, scoring_experiment_name) / "splits"
    split_path = split_dir / _split_cache_filename(
        target_scheme, train_indices, test_indices, all_indices, train_breaks,
        n_iterations, subset_sizes, architecture_resample_fractions,
        architecture_n_resamples_per_fraction, num_muts, max_muts_in_train
    )
    if save_splits and split_path.exists() and not refresh_splits:
        payload = _torch_load(split_path)
        payload["path"] = str(split_path)
        return payload

    _validate_split_args(train_indices, test_indices, all_indices, subset_sizes, num_muts, max_muts_in_train)
    splits = _create_scoring_splits(
        n_rows=n_rows,
        train_indices=train_indices,
        test_indices=test_indices,
        all_indices=all_indices,
        train_breaks=train_breaks,
        n_iterations=n_iterations,
        subset_sizes=subset_sizes,
        architecture_resample_fractions=architecture_resample_fractions,
        architecture_n_resamples_per_fraction=architecture_n_resamples_per_fraction,
        num_muts=num_muts,
        max_muts_in_train=max_muts_in_train,
        random_seed=random_seed,
    )
    split_mode = "num_muts" if num_muts is not None else ("random_size" if subset_sizes is not None else "random")
    payload = {
        "target_scheme": target_scheme,
        "split_mode": split_mode,
        "splits": splits,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if save_splits:
        split_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, split_path)
        payload["path"] = str(split_path)
    return payload

def run_architecture_resolution_job(job_config):
    _print_scoring_experiment_device(
        job_config.get("device"),
        verbose=job_config.get("verbose", True),
    )
    context = _architecture_job_context(job_config)
    split = next(
        item for item in context["split_payload"]["splits"]
        if item["split_id"] == job_config["split_id"]
    )
    split_job = {
        "split_id": job_config["split_id"],
        "cache_path": job_config["cache_path"],
        "architecture_resamples": job_config.get("architecture_resamples"),
        "random_seed": job_config["random_seed"],
    }
    _run_architecture_resolution_for_split(context, split, split_job)

def _architecture_job_context(job_config):
    data_path = Path(job_config["data_path"])
    features = _load_features(data_path, job_config["y_key"])
    target_kwargs = job_config.get("target_kwargs", {})
    y, task_type, target_scheme = _prepare_target(
        features[job_config["y_key"]],
        extremes=target_kwargs.get("extremes"),
        ordinal=target_kwargs.get("ordinal"),
        discrete=target_kwargs.get("discrete", False),
    )
    split_payload = _torch_load(job_config["split_file_path"])
    combo_info = job_config["combo_info"]
    x_all = _make_feature_matrix(features, combo_info["combo"])
    return {
        "job_config": job_config,
        "data_path": data_path,
        "features": features,
        "y": y,
        "task_type": task_type,
        "target_scheme": target_scheme,
        "split_payload": split_payload,
        "combo_info": combo_info,
        "x_all": x_all,
    }

def _run_architecture_resolution_for_split(context, split, split_job):
    job_config = context["job_config"]
    data_path = context["data_path"]
    y = context["y"]
    task_type = context["task_type"]
    target_scheme = context["target_scheme"]
    combo_info = context["combo_info"]
    x_all = context["x_all"]
    train_idx = np.asarray(split["train_indices"], dtype=int)
    architecture_resamples = split_job.get("architecture_resamples")
    if architecture_resamples is None:
        architecture_resamples = _require_architecture_resamples(split)
    _print_split_resolution_progress(
        split,
        architecture_resamples,
        n_seeds=job_config["n_seeds"],
        feature_name=combo_info["name"],
        verbose=job_config.get("verbose", True),
    )

    architecture = resolve_architecture(
        x_all[train_idx],
        y[train_idx],
        task_type=_architecture_task_type(task_type, target_scheme),
        architecture_grid=job_config["architecture_grid"],
        max_epochs=job_config["max_epochs"],
        eval_every=job_config["eval_every"],
        resample_fractions=job_config["resample_fractions"],
        n_resamples_per_fraction=job_config["n_resamples_per_fraction"],
        n_seeds=job_config["n_seeds"],
        random_seed=split_job["random_seed"],
        device=job_config.get("device"),
        save_path=split_job["cache_path"],
        verbose=job_config.get("verbose", True),
        resamples=architecture_resamples,
        standardize_features=job_config.get("standardize_features", False),
    )
    row = _run_single_resolved_split_score(
        x_all=x_all,
        y=y,
        task_type=task_type,
        target_scheme=target_scheme,
        combo_info=combo_info,
        split=split,
        architecture=architecture,
        result_path=job_config.get("result_path", ""),
        data_path=data_path,
        precision_at_k=job_config.get("precision_at_k"),
        device=job_config.get("device"),
        random_seed=split_job["random_seed"],
        standardize_features=job_config.get("standardize_features", False),
    )
    _save_split_result_row(split_job["cache_path"], row)

def _load_features(data_path, y_key):
    feature_path = _features_file_path(data_path)
    if not feature_path.exists():
        raise FileNotFoundError(f"Expected feature file does not exist: {feature_path}")
    try:
        features = torch.load(feature_path, map_location="cpu", weights_only=False)
    except TypeError:
        features = torch.load(feature_path, map_location="cpu")
    _validate_features(features, y_key)
    return features

def _features_file_path(data_path):
    path = Path(data_path).expanduser()
    if path.is_file() or path.suffix == ".pt":
        return path
    return path / "features.pt"

def _validate_features(features, y_key):
    if not isinstance(features, dict):
        raise ValueError("features.pt must contain a dictionary.")
    if y_key not in features:
        raise ValueError(f"Target key {y_key!r} was not found in features.pt.")

    feature_names = [name for name in features if name != y_key]
    if len(feature_names) < 1:
        raise ValueError("features.pt must contain at least one feature key.")

    y = _as_tensor(features[y_key], name=y_key)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    if y.ndim != 1:
        raise ValueError(f"Target {y_key!r} must have shape [N] or [N, 1].")
    if not torch.isfinite(y.float()).all():
        raise ValueError(f"Target {y_key!r} contains NaN or infinite values.")

    n_rows = int(y.shape[0])
    for name in feature_names:
        value = _as_tensor(features[name], name=name)
        if value.ndim < 1:
            raise ValueError(f"Feature {name!r} must have a first dimension.")
        if int(value.shape[0]) != n_rows:
            raise ValueError(
                f"Feature {name!r} has length {value.shape[0]}, but target has length {n_rows}."
            )
        if not torch.isfinite(value.float()).all():
            raise ValueError(f"Feature {name!r} contains NaN or infinite values.")

def _prepare_target(y, extremes, ordinal, discrete):
    modes = [extremes is not None, ordinal is not None, bool(discrete)]
    if sum(modes) > 1:
        raise ValueError("Only one target mode may be set: extremes, ordinal, or discrete.")
    if extremes not in (None, "positives", "negatives"):
        raise ValueError("extremes must be one of None, 'positives', or 'negatives'.")

    y_np = _as_tensor(y, name="target").detach().cpu().reshape(-1).numpy()
    if not np.isfinite(y_np.astype(float)).all():
        raise ValueError("Target contains NaN or infinite values.")

    if discrete:
        labels = y_np.astype(int)
        if not np.allclose(y_np, labels):
            raise ValueError("discrete=True requires integer class labels.")
        _, encoded = np.unique(labels, return_inverse=True)
        return torch.as_tensor(encoded, dtype=torch.long), "classification", "y_discrete"

    if extremes == "positives":
        threshold = np.percentile(y_np.astype(float), 90)
        labels = (y_np >= threshold).astype(np.int64)
        return torch.as_tensor(labels, dtype=torch.long), "classification", "y_positives"

    if extremes == "negatives":
        threshold = np.percentile(y_np.astype(float), 10)
        labels = np.where(y_np <= threshold, 0, 1).astype(np.int64)
        return torch.as_tensor(labels, dtype=torch.long), "classification", "y_negatives"

    if ordinal is not None:
        labels = _make_ordinal_labels(y_np.astype(float), ordinal)
        return torch.as_tensor(labels, dtype=torch.float32), "ordinal", "y_ordinal"

    return torch.as_tensor(y_np, dtype=torch.float32), "regression", "y"

def _make_ordinal_labels(y, ordinal):
    if isinstance(ordinal, dict):
        if not ordinal:
            raise ValueError("ordinal dict must contain at least one threshold.")
        items = sorted((float(k), float(v)) for k, v in ordinal.items())
        thresholds = [item[0] for item in items]
        labels = [item[1] for item in items]
        output = np.empty(len(y), dtype=np.float32)
        output[y < thresholds[0]] = labels[0]
        for idx in range(1, len(thresholds)):
            mask = (y >= thresholds[idx - 1]) & (y < thresholds[idx])
            output[mask] = labels[idx]
        output[y >= thresholds[-1]] = max(labels) + 1
        return output

    thresholds = [float(value) for value in ordinal]
    if not thresholds:
        raise ValueError("ordinal thresholds must contain at least one value.")
    if thresholds != sorted(thresholds):
        raise ValueError("ordinal list thresholds must be sorted increasingly.")
    return np.digitize(y, thresholds, right=False).astype(np.float32)

def _feature_ndims(features, feature_names):
    return {
        name: _as_tensor(features[name], name=name).ndim
        for name in feature_names
    }

def _build_feature_combinations(
    feature_names,
    all_singles,
    all_doubles,
    custom_combinations,
    combine_all,
    feature_ndims=None,
):
    feature_names = list(feature_names)
    feature_set = set(feature_names)
    feature_ndims = dict(feature_ndims or {})
    combinations = []

    if all_singles:
        for name in feature_names:
            if feature_ndims.get(name) == 1:
                continue
            combinations.append({"name": name, "combo": [name]})

    if all_doubles:
        for combo in itertools.combinations(feature_names, 2):
            combinations.append({"name": "+".join(combo), "combo": list(combo)})

    if custom_combinations:
        for combo in custom_combinations:
            combo = list(combo)
            missing = [name for name in combo if name not in feature_set]
            if missing:
                raise ValueError(f"Custom feature combination contains missing keys: {missing}")
            combinations.append({"name": "+".join(combo), "combo": combo})

    if combine_all:
        combinations.append({"name": "all_features", "combo": feature_names})

    deduped = []
    seen = set()
    for item in combinations:
        key = (item["name"], tuple(item["combo"]))
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    if not deduped:
        raise ValueError("No valid feature combinations were requested.")
    return deduped

def _make_feature_matrix(features, feature_combo):
    matrices = []
    for name in feature_combo:
        if name not in features:
            raise ValueError(f"Feature key {name!r} was not found.")
        tensor = _as_tensor(features[name], name=name).float()
        matrices.append(tensor.reshape(tensor.shape[0], -1))
    return torch.cat(matrices, dim=1)

def _resolve_results_path(
    results_file_name,
    data_path,
    scoring_experiment_name="scoring_experiment",
):
    result_path = Path(results_file_name)
    if result_path.is_absolute():
        return result_path
    return _scoring_experiment_dir(data_path, scoring_experiment_name) / "results" / result_path

def _scoring_experiment_dir(data_path, scoring_experiment_name="scoring_experiment"):
    path = Path(data_path).expanduser()
    if path.is_file() or path.suffix == ".pt":
        path = path.parent
    return path / _sanitize_name(scoring_experiment_name)

def _validate_split_args(train_indices, test_indices, all_indices, subset_sizes, num_muts, max_muts_in_train):
    if (train_indices is None) != (test_indices is None):
        raise ValueError("train_indices and test_indices must be provided together.")
    if train_indices is not None and all_indices is not None:
        raise ValueError("all_indices cannot be combined with train_indices/test_indices.")
    if subset_sizes is not None and (num_muts is not None or max_muts_in_train is not None):
        raise ValueError("subset_sizes cannot be combined with num_muts mode.")
    if num_muts is not None or max_muts_in_train is not None:
        if train_indices is not None or test_indices is not None:
            raise ValueError("train_indices/test_indices are not supported with num_muts mode.")
        if num_muts is None or max_muts_in_train is None:
            raise ValueError("num_muts mode requires both num_muts and max_muts_in_train.")

def _create_scoring_splits(
    n_rows,
    train_indices=None,
    test_indices=None,
    all_indices=None,
    train_breaks=6,
    n_iterations=5,
    subset_sizes=None,
    architecture_resample_fractions=(0.15, 0.25, 0.5, 0.75, 0.8),
    architecture_n_resamples_per_fraction=5,
    num_muts=None,
    max_muts_in_train=None,
    random_seed=0,
):
    rng = np.random.default_rng(random_seed)
    splits = []
    if num_muts is not None:
        num_muts = np.asarray(num_muts).reshape(-1)
        if len(num_muts) != n_rows:
            raise ValueError(f"num_muts has length {len(num_muts)}, but expected {n_rows}.")
        base_indices = _selected_indices(n_rows, all_indices)
        if base_indices is None:
            base_indices = np.arange(n_rows)
        for max_k in range(1, int(max_muts_in_train) + 1):
            train_idx = base_indices[num_muts[base_indices] <= max_k]
            test_idx = base_indices[num_muts[base_indices] > max_k]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            splits.append(
                _split_payload(
                    split_id=f"num_muts_le_{max_k}",
                    train_idx=train_idx,
                    test_idx=test_idx,
                    subset_fraction=None,
                    iteration=0,
                    metadata={"split_mode": "num_muts", "max_muts_in_train": int(max_k)},
                    architecture_resample_fractions=architecture_resample_fractions,
                    architecture_n_resamples_per_fraction=architecture_n_resamples_per_fraction,
                    architecture_random_seed=random_seed + len(splits) * 100_000,
                )
            )
        return splits

    train_indices = None if train_indices is None else _normalize_indices(train_indices, n_rows)
    test_indices = None if test_indices is None else _normalize_indices(test_indices, n_rows)
    base_indices = _selected_indices(n_rows, all_indices)
    if base_indices is None:
        base_indices = np.arange(n_rows)
    normalized_subset_sizes = _normalize_subset_sizes(subset_sizes)
    if normalized_subset_sizes is not None:
        for size_idx, subset_size in enumerate(normalized_subset_sizes):
            for iteration in range(int(n_iterations)):
                if train_indices is not None:
                    train_idx, test_idx = _make_subset_train_size_split(
                        train_indices, test_indices, subset_size, rng
                    )
                    split_source = "train_test_indices"
                    denominator = len(train_indices)
                else:
                    train_idx, test_idx = _make_random_size_split(base_indices, subset_size, rng)
                    split_source = "all_indices" if all_indices is not None else "all_rows"
                    denominator = len(base_indices)
                if len(train_idx) == 0 or len(test_idx) == 0:
                    continue
                splits.append(
                    _split_payload(
                        split_id=f"split_{len(splits) + 1:03d}",
                        train_idx=train_idx,
                        test_idx=test_idx,
                        subset_fraction=float(len(train_idx) / denominator) if denominator else None,
                        iteration=int(iteration),
                        metadata={
                            "split_mode": "random_size",
                            "split_source": split_source,
                            "size_index": int(size_idx),
                            "requested_subset_size": int(subset_size),
                        },
                        architecture_resample_fractions=architecture_resample_fractions,
                        architecture_n_resamples_per_fraction=architecture_n_resamples_per_fraction,
                        architecture_random_seed=random_seed + len(splits) * 100_000,
                    )
                )
        return splits

    subset_fractions = np.linspace(0.15, 0.90, int(train_breaks))
    for fraction_idx, subset_fraction in enumerate(subset_fractions):
        for iteration in range(int(n_iterations)):
            if train_indices is not None:
                train_idx, test_idx = _make_subset_train_split(
                    train_indices, test_indices, subset_fraction, rng
                )
                split_source = "train_test_indices"
            else:
                train_idx, test_idx = _make_random_split(base_indices, subset_fraction, rng)
                split_source = "all_indices" if all_indices is not None else "all_rows"
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            splits.append(
                _split_payload(
                    split_id=f"split_{len(splits) + 1:03d}",
                    train_idx=train_idx,
                    test_idx=test_idx,
                    subset_fraction=float(subset_fraction),
                    iteration=int(iteration),
                    metadata={
                        "split_mode": "random",
                        "split_source": split_source,
                        "fraction_index": int(fraction_idx),
                    },
                    architecture_resample_fractions=architecture_resample_fractions,
                    architecture_n_resamples_per_fraction=architecture_n_resamples_per_fraction,
                    architecture_random_seed=random_seed + len(splits) * 100_000,
                )
            )
    return splits

def _split_payload(
    split_id,
    train_idx,
    test_idx,
    subset_fraction,
    iteration,
    metadata,
    architecture_resample_fractions=None,
    architecture_n_resamples_per_fraction=None,
    architecture_random_seed=0,
):
    train_idx = np.asarray(train_idx, dtype=int)
    payload = {
        "split_id": str(split_id),
        "subset_fraction": subset_fraction,
        "iteration": int(iteration),
        "train_indices": [int(value) for value in train_idx],
        "test_indices": [int(value) for value in np.asarray(test_idx, dtype=int)],
        "metadata": dict(metadata),
    }
    if architecture_resample_fractions is not None and architecture_n_resamples_per_fraction is not None:
        payload["architecture_resamples"] = _make_architecture_resamples(
            n_rows=len(train_idx),
            resample_fractions=architecture_resample_fractions,
            n_resamples_per_fraction=architecture_n_resamples_per_fraction,
            random_seed=architecture_random_seed,
        )
    return payload

def _split_cache_filename(
    target_scheme,
    train_indices,
    test_indices,
    all_indices,
    train_breaks,
    n_iterations,
    subset_sizes,
    architecture_resample_fractions,
    architecture_n_resamples_per_fraction,
    num_muts,
    max_muts_in_train,
):
    target = _sanitize_name(target_scheme)
    arch_label = _architecture_resample_cache_label(
        architecture_resample_fractions,
        architecture_n_resamples_per_fraction,
    )
    if num_muts is not None:
        selected = "all_indices" if all_indices is not None else "all"
        return f"splits__{target}__num_muts__{selected}__max_{int(max_muts_in_train)}__arch_{arch_label}.pt"
    if train_indices is not None and test_indices is not None:
        source = "train_test_indices"
    elif all_indices is not None:
        source = "all_indices"
    else:
        source = "all"
    normalized_subset_sizes = _normalize_subset_sizes(subset_sizes)
    if normalized_subset_sizes is not None:
        size_label = "_".join(str(value) for value in normalized_subset_sizes)
        return (
            f"splits__{target}__random_size__{source}"
            f"__sizes_{size_label}__iters_{int(n_iterations)}"
            f"__arch_{arch_label}.pt"
        )
    return (
        f"splits__{target}__random__{source}"
        f"__trainbreaks_{int(train_breaks)}__iters_{int(n_iterations)}"
        f"__arch_{arch_label}.pt"
    )

def _save_architecture_winners(rows, data_path, scoring_experiment_name="scoring_experiment"):
    winner_columns = [
        "target_scheme",
        "feature_name",
        "feature_combo",
        "split",
        "subset_fraction",
        "iteration",
        "n_train",
        "n_test",
        "architecture_name",
        "best_epoch",
        "architecture_model_path",
        "architecture_best_score",
        "architecture_metric",
        "architecture_search_path",
        "hidden_layers",
        "lr",
        "weight_decay",
        "dropout",
        "batch_size",
        "architecture_n_resamples",
        "architecture_n_seeds",
        "architecture_max_epochs",
        "architecture_eval_every",
    ]
    winner_rows = []
    for row in rows:
        if not row.get("architecture_name") and not row.get("architecture_search_path"):
            continue
        winner_rows.append({column: row.get(column, np.nan) for column in winner_columns})
    path = _architecture_winners_path(data_path, scoring_experiment_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(winner_rows, columns=winner_columns).to_csv(path, index=False)
    return path

def _architecture_winners_path(data_path, scoring_experiment_name="scoring_experiment"):
    return _scoring_experiment_dir(data_path, scoring_experiment_name) / "architecture_winners.csv"

def _run_single_resolved_split_score(
    x_all,
    y,
    task_type,
    target_scheme,
    combo_info,
    split,
    architecture,
    result_path,
    data_path,
    precision_at_k,
    device,
    random_seed,
    standardize_features=False,
):
    train_idx = np.asarray(split["train_indices"], dtype=int)
    test_idx = np.asarray(split["test_indices"], dtype=int)
    config = dict(architecture["best_config"])
    best_epoch = int(architecture["best_epoch"])
    row = _base_result_row(
        result_path,
        data_path,
        target_scheme,
        combo_info,
        split["split_id"],
        split.get("subset_fraction"),
        split.get("iteration", 0),
        len(train_idx),
        len(test_idx),
        task_type,
        architecture.get("architecture_search_path", ""),
        {**config, "epochs": best_epoch},
    )
    row.update(_architecture_result_row(architecture))
    try:
        result = train_scorer(
            x_all[train_idx],
            y[train_idx],
            x_all[test_idx],
            y[test_idx],
            task_type,
            config,
            device,
            random_seed,
            epochs=best_epoch,
            standardize_features=standardize_features,
        )
        metrics = evaluate_predictions(
            y[test_idx].detach().cpu().numpy(),
            result["predictions"],
            task_type,
            precision_at_k=precision_at_k,
        )
        row.update(metrics)
        model_path = _save_split_winning_model(
            architecture=architecture,
            result=result,
            row=row,
            combo_info=combo_info,
            split=split,
            task_type=task_type,
            target_scheme=target_scheme,
            x_train=x_all[train_idx],
        )
        row["architecture_model_path"] = str(model_path)
    except Exception as exc:
        row["error"] = str(exc)
    return row

def _save_split_winning_model(
    architecture,
    result,
    row,
    combo_info,
    split,
    task_type,
    target_scheme,
    x_train,
):
    model_dir = Path(architecture["architecture_search_path"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "winning_model.pt"
    model = result["model"].to("cpu")
    payload = {
        "model_state_dict": model.state_dict(),
        "input_dim": int(_as_tensor(x_train, name="x_train").shape[1]),
        "output_dim": _model_output_dim(model),
        "feature_name": combo_info["name"],
        "feature_combo": combo_info["combo"],
        "target_scheme": target_scheme,
        "task_type": task_type,
        "split_id": split["split_id"],
        "subset_fraction": split.get("subset_fraction"),
        "iteration": int(split.get("iteration", 0)),
        "n_train": int(row.get("n_train", 0)),
        "n_test": int(row.get("n_test", 0)),
        "train_indices": [int(value) for value in split["train_indices"]],
        "test_indices": [int(value) for value in split["test_indices"]],
        "best_config": _json_safe_config(architecture["best_config"]),
        "best_epoch": int(architecture["best_epoch"]),
        "architecture_best_score": row.get("architecture_best_score"),
        "architecture_metric": row.get("architecture_metric"),
        "metrics": _json_safe_metrics({key: row.get(key) for key in _METRIC_COLUMNS}),
        "validation_loss": float(result["validation_loss"]),
        "architecture_search_path": str(model_dir),
        "standardize_features": bool(result.get("standardize_features", False)),
    }
    if result.get("x_mean") is not None and result.get("x_std") is not None:
        payload["x_mean"] = result["x_mean"]
        payload["x_std"] = result["x_std"]
    torch.save(payload, model_path)
    architecture["architecture_model_path"] = str(model_path)
    return model_path

def _architecture_result_row(result):
    config = result["best_config"]
    return {
        "architecture_name": config.get("name", ""),
        "architecture_search_path": result.get("architecture_search_path", ""),
        "architecture_model_path": result.get("architecture_model_path", ""),
        "best_epoch": int(result.get("best_epoch", 0)),
        "architecture_best_score": result.get("best_score", np.nan),
        "architecture_metric": result.get("metric_name", ""),
        "architecture_n_resamples": len(result.get("resample_metadata", [])),
        "architecture_n_seeds": int(result.get("n_seeds", 0)),
        "architecture_max_epochs": int(result.get("max_epochs", 0)),
        "architecture_eval_every": int(result.get("eval_every", 0)),
    }

def _architecture_config(config, index):
    safe = dict(config)
    safe.setdefault("name", f"architecture_{int(index):03d}")
    safe.setdefault("hidden_layers", [64])
    safe.setdefault("lr", 1e-3)
    safe.setdefault("weight_decay", 0.0)
    safe.setdefault("dropout", 0.0)
    safe.setdefault("batch_size", 64)
    return safe

def _make_architecture_resamples(
    n_rows,
    resample_fractions,
    n_resamples_per_fraction,
    random_seed,
):
    rng = np.random.default_rng(random_seed)
    indices = np.arange(int(n_rows))
    resamples = []
    for fraction in resample_fractions:
        for iteration in range(int(n_resamples_per_fraction)):
            train_idx, val_idx = _make_random_split(indices, float(fraction), rng)
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            resamples.append(
                {
                    "train_fraction": float(fraction),
                    "iteration": int(iteration),
                    "train_indices": [int(value) for value in train_idx],
                    "val_indices": [int(value) for value in val_idx],
                }
            )
    return resamples

def _normalize_architecture_resamples(resamples):
    normalized = []
    for idx, item in enumerate(resamples):
        normalized.append(
            {
                "train_fraction": float(item["train_fraction"]),
                "iteration": int(item.get("iteration", idx)),
                "train_indices": [int(value) for value in item["train_indices"]],
                "val_indices": [int(value) for value in item["val_indices"]],
            }
        )
    return normalized

def _print_split_resolution_progress(
    split,
    architecture_resamples,
    *,
    n_seeds,
    feature_name=None,
    verbose=True,
):
    if not verbose:
        return
    split_label = str(split.get("split_id", ""))
    if split_label.startswith("split_"):
        split_label = split_label[len("split_") :]
    outer_train = len(split.get("train_indices") or [])
    outer_test = len(split.get("test_indices") or [])
    resamples = _normalize_architecture_resamples(architecture_resamples)
    validation_iterations = len(resamples) * int(n_seeds)
    train_sizes = [len(item["train_indices"]) for item in resamples]
    val_sizes = [len(item["val_indices"]) for item in resamples]
    train_text = _count_or_range(train_sizes)
    val_text = _count_or_range(val_sizes)
    feature_text = f" for {feature_name}" if feature_name else ""
    print(
        f"Resolving split {split_label}{feature_text} "
        f"(Train: {outer_train}, test {outer_test}) "
        f"Overall {validation_iterations} validation iterations of "
        f"({train_text} train {val_text} validation)",
        flush=True,
    )

def _count_or_range(values):
    if not values:
        return "0"
    low = min(values)
    high = max(values)
    if low == high:
        return str(low)
    return f"{low}-{high}"

def _require_architecture_resamples(split):
    resamples = split.get("architecture_resamples")
    if resamples is None:
        split_id = split.get("split_id", "<unknown>")
        raise RuntimeError(
            f"Scoring split {split_id!r} is missing architecture_resamples. "
            "Recreate the scoring splits with refresh=True/architecture_refresh_cache=True "
            "so all feature architectures share the same internal train/validation splits."
        )
    return resamples

def _architecture_task_type(task_type, target_scheme):
    if target_scheme == "y":
        return "continuous"
    if target_scheme == "y_ordinal":
        return "ordinal"
    if target_scheme == "y_discrete":
        return "discrete"
    if target_scheme == "y_positives":
        return "positives"
    if target_scheme == "y_negatives":
        return "negatives"
    return task_type

def _training_task_type(task_type):
    if task_type in ("continuous", "regression"):
        return "regression"
    if task_type == "ordinal":
        return "ordinal"
    if task_type in ("discrete", "positives", "negatives", "classification"):
        return "classification"
    raise ValueError(
        "task_type must be one of continuous, ordinal, discrete, positives, negatives."
    )

def _architecture_metric_name(task_type):
    train_type = _training_task_type(task_type)
    return "spearman" if train_type in ("regression", "ordinal") else "roc_auc"

def _architecture_search_cache_path(
    data_path,
    target_scheme,
    combo_info,
    split_id,
    scoring_experiment_name="scoring_experiment",
    standardize_features=False,
):
    path = (
        _scoring_experiment_dir(data_path, scoring_experiment_name)
        / "architecture_search"
    )
    if standardize_features:
        path = path / "standardized_features"
    return path / _sanitize_name(target_scheme) / _sanitize_name(combo_info["name"]) / _sanitize_name(split_id)

def _save_architecture_search_result(save_path, result):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_config": _json_safe_config(result["best_config"]),
        "best_epoch": int(result["best_epoch"]),
        "best_score": None if not np.isfinite(result["best_score"]) else float(result["best_score"]),
        "metric_name": result["metric_name"],
        "architecture_grid": result["architecture_grid"],
        "checkpoint_epochs": result["checkpoint_epochs"],
        "resample_metadata": result["resample_metadata"],
        "architecture_search_path": str(save_path),
        "standardize_features": bool(result.get("standardize_features", False)),
        "n_seeds": _infer_n_seeds(result["scores"]),
        "max_epochs": max(result["checkpoint_epochs"]) if result["checkpoint_epochs"] else 0,
        "eval_every": _infer_eval_every(result["checkpoint_epochs"]),
    }
    with (save_path / "best_architecture.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    result["score_table"].to_csv(save_path / "architecture_scores.csv", index=False)
    torch.save(
        {
            "scores": result["scores"],
            "checkpoint_epochs": result["checkpoint_epochs"],
            "resample_metadata": result["resample_metadata"],
            "architecture_grid": result["architecture_grid"],
        },
        save_path / "architecture_scores.pt",
    )
    result.update(payload)

def _architecture_search_cache_complete(cache_path, require_split_result=False):
    cache_path = Path(cache_path)
    if not (cache_path / "best_architecture.json").is_file():
        return False
    if require_split_result and not _split_result_cache_path(cache_path).is_file():
        return False
    return True

def _split_result_cache_path(cache_path):
    return Path(cache_path) / "split_result.json"

def _save_split_result_row(cache_path, row):
    path = _split_result_cache_path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(_json_safe_row(row), handle, indent=2)

def _load_split_result_row(path):
    with Path(path).open() as handle:
        return json.load(handle)

def _normalize_result_row(row):
    normalized = {column: np.nan for column in _RESULT_COLUMNS}
    normalized.update(dict(row))
    return normalized

def _infer_n_seeds(scores):
    return int(scores.shape[3]) if hasattr(scores, "shape") and len(scores.shape) == 4 else 0

def _infer_eval_every(checkpoint_epochs):
    if len(checkpoint_epochs) < 2:
        return int(checkpoint_epochs[0]) if checkpoint_epochs else 0
    return int(checkpoint_epochs[1]) - int(checkpoint_epochs[0])

def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def train_scorer(
    X_train,
    y_train,
    X_test,
    y_test,
    task_type,
    config,
    device,
    random_seed,
    select_best_by_validation_loss=False,
    epochs=None,
    standardize_features=False,
):
    _set_random_seed(random_seed)
    device = torch.device(_resolve_device(device))
    x_train = _as_tensor(X_train, name="X_train").float().to(device)
    x_test = _as_tensor(X_test, name="X_test").float().to(device)
    x_train, x_test, x_mean, x_std = _standardize_feature_tensors(
        x_train,
        x_test,
        enabled=standardize_features,
    )
    y_train_tensor = _as_tensor(y_train, name="y_train")
    y_test_tensor = _as_tensor(y_test, name="y_test")

    if task_type == "classification":
        y_train_device = y_train_tensor.long().to(device)
        output_dim = int(torch.max(torch.cat([y_train_tensor.long(), y_test_tensor.long()])).item()) + 1
        loss_fn = nn.CrossEntropyLoss()
    else:
        y_train_device = y_train_tensor.float().reshape(-1, 1).to(device)
        output_dim = 1
        loss_fn = nn.MSELoss()

    model = _MLPScorer(
        input_dim=x_train.shape[1],
        output_dim=output_dim,
        hidden_layers=config.get("hidden_layers", [64]),
        dropout=config.get("dropout", 0.0),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 0.0),
    )
    batch_size = int(config.get("batch_size", 64))
    epochs = int(epochs if epochs is not None else config.get("epochs", 100))
    best_state = None
    best_validation_loss = float("inf")

    model.train()
    for _ in range(epochs):
        order = torch.randperm(x_train.shape[0], device=device)
        for start in range(0, x_train.shape[0], batch_size):
            batch_idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            output = model(x_train[batch_idx])
            loss = loss_fn(output, y_train_device[batch_idx])
            loss.backward()
            optimizer.step()
        if select_best_by_validation_loss:
            model.eval()
            with torch.no_grad():
                validation_output = model(x_test)
                current_loss = _compute_loss_value(
                    loss_fn, validation_output, y_test_tensor, task_type, device
                )
            if current_loss < best_validation_loss:
                best_validation_loss = current_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        output = model(x_test)
        validation_loss = _compute_loss_value(
            loss_fn, output, y_test_tensor, task_type, device
        )
        if task_type == "classification":
            predictions = torch.softmax(output, dim=1).detach().cpu().numpy()
        else:
            predictions = output.reshape(-1).detach().cpu().numpy()

    metrics = evaluate_predictions(
        y_test_tensor.detach().cpu().numpy(), predictions, task_type
    )
    return {
        "model": model,
        "metrics": metrics,
        "predictions": predictions,
        "validation_loss": validation_loss,
        "standardize_features": bool(standardize_features),
        "x_mean": x_mean,
        "x_std": x_std,
    }

def _compute_loss_value(loss_fn, output, y_true, task_type, device):
    if task_type == "classification":
        target = y_true.long().to(device)
    else:
        target = y_true.float().reshape(-1, 1).to(device)
    return float(loss_fn(output, target).item())

def evaluate_predictions(y_true, y_pred, task_type, precision_at_k=None):
    y_true = np.asarray(y_true).reshape(-1)
    metrics = {
        "pearson": np.nan,
        "spearman": np.nan,
        "mse": np.nan,
        "roc_auc": np.nan,
        "precision": np.nan,
        "accuracy": np.nan,
        "precision_at_k": np.nan,
    }

    if len(y_true) == 0:
        return metrics

    if task_type in ("regression", "ordinal"):
        y_pred = np.asarray(y_pred).reshape(-1)
        metrics["mse"] = float(np.mean((y_true.astype(float) - y_pred.astype(float)) ** 2))
        metrics["pearson"] = _pearson(y_true, y_pred)
        metrics["spearman"] = _spearman(y_true, y_pred)
        return metrics

    probabilities = np.asarray(y_pred)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 1)
    pred_labels = np.argmax(probabilities, axis=1)
    labels_present = np.unique(y_true)
    average = "binary" if len(labels_present) <= 2 and probabilities.shape[1] <= 2 else "macro"

    metrics["accuracy"] = float(accuracy_score(y_true, pred_labels))
    metrics["precision"] = float(
        precision_score(y_true, pred_labels, average=average, zero_division=0)
    )
    metrics["roc_auc"] = _roc_auc(y_true, probabilities)

    if precision_at_k is not None and probabilities.shape[1] >= 2:
        k = min(int(precision_at_k), len(y_true))
        if k > 0:
            ranked = np.argsort(-probabilities[:, 1])[:k]
            metrics["precision_at_k"] = float(np.mean(y_true[ranked] == 1))

    return metrics

def _base_result_row(
    result_path,
    data_path,
    target_scheme,
    combo_info,
    split_name,
    subset_fraction,
    iteration,
    n_train,
    n_test,
    task_type,
    config_path,
    config,
):
    row = {column: np.nan for column in _RESULT_COLUMNS}
    row.update(
        {
            "results_file_name": str(result_path),
            "data_path": str(data_path),
            "target_scheme": target_scheme,
            "feature_name": combo_info["name"],
            "feature_combo": json.dumps(combo_info["combo"]),
            "split": split_name,
            "subset_fraction": subset_fraction,
            "iteration": iteration,
            "n_train": n_train,
            "n_test": n_test,
            "task_type": task_type,
            "config_path": str(config_path),
            "hidden_layers": json.dumps(config.get("hidden_layers", [])),
            "lr": config.get("lr", np.nan),
            "weight_decay": config.get("weight_decay", np.nan),
            "dropout": config.get("dropout", np.nan),
            "epochs": config.get("epochs", np.nan),
            "batch_size": config.get("batch_size", np.nan),
            "error": "",
        }
    )
    return row

def _make_random_split(indices, train_fraction, rng):
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        return indices, indices
    shuffled = rng.permutation(indices)
    n_train = int(round(float(train_fraction) * len(shuffled)))
    if len(shuffled) > 1:
        n_train = min(max(n_train, 1), len(shuffled) - 1)
    else:
        n_train = len(shuffled)
    return shuffled[:n_train], shuffled[n_train:]

def _make_random_size_split(indices, train_size, rng):
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        return indices, indices
    shuffled = rng.permutation(indices)
    n_train = int(train_size)
    if len(shuffled) > 1:
        n_train = min(max(n_train, 1), len(shuffled) - 1)
    else:
        n_train = len(shuffled)
    return shuffled[:n_train], shuffled[n_train:]

def _make_subset_train_split(train_indices, test_indices, train_fraction, rng):
    train_indices = np.asarray(train_indices, dtype=int)
    n_train = int(round(float(train_fraction) * len(train_indices)))
    n_train = min(max(n_train, 1), len(train_indices)) if len(train_indices) else 0
    sampled_train = rng.permutation(train_indices)[:n_train]
    return sampled_train, np.asarray(test_indices, dtype=int)

def _make_subset_train_size_split(train_indices, test_indices, train_size, rng):
    train_indices = np.asarray(train_indices, dtype=int)
    n_train = int(train_size)
    n_train = min(max(n_train, 1), len(train_indices)) if len(train_indices) else 0
    sampled_train = rng.permutation(train_indices)[:n_train]
    return sampled_train, np.asarray(test_indices, dtype=int)

def _normalize_subset_sizes(subset_sizes):
    if subset_sizes is None:
        return None
    if isinstance(subset_sizes, (str, bytes)):
        raw = [item.strip() for item in str(subset_sizes).split(",") if item.strip()]
    elif np.isscalar(subset_sizes):
        raw = [subset_sizes]
    else:
        raw = list(subset_sizes)
    sizes = []
    for value in raw:
        size = int(value)
        if size < 1:
            raise ValueError("subset_sizes must contain positive integers.")
        if size not in sizes:
            sizes.append(size)
    if not sizes:
        raise ValueError("subset_sizes must contain at least one positive integer.")
    return sizes

def _architecture_resample_cache_label(resample_fractions, n_resamples_per_fraction):
    fractions = "_".join(_float_cache_label(value) for value in resample_fractions)
    return f"{fractions}x{int(n_resamples_per_fraction)}"

def _float_cache_label(value):
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")

def _selected_indices(n_rows, all_indices):
    if all_indices is None:
        return None
    return _normalize_indices(all_indices, n_rows)

def _normalize_indices(indices, n_rows):
    indices = np.asarray(indices, dtype=int).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= n_rows):
        raise ValueError(f"Indices must be between 0 and {n_rows - 1}.")
    return indices

def _as_tensor(value, name):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    try:
        return torch.as_tensor(value)
    except Exception as exc:
        raise ValueError(f"{name!r} could not be converted to a tensor.") from exc

def _resolve_device(device):
    if device is not None:
        return str(torch.device(device))
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _print_scoring_experiment_device(device, verbose=True):
    if verbose:
        print(f"Scoring experiment using device: {_resolve_device(device)}", flush=True)

def _set_random_seed(random_seed):
    torch.manual_seed(int(random_seed))
    np.random.seed(int(random_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(random_seed))

def _pearson(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def _spearman(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    if spearmanr is not None:
        value = spearmanr(y_true, y_pred).correlation
        return float(value) if value is not None else float("nan")
    return float(pd.Series(y_true).rank().corr(pd.Series(y_pred).rank()))

def _roc_auc(y_true, probabilities):
    y_true = np.asarray(y_true).reshape(-1)
    probabilities = np.asarray(probabilities)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        if probabilities.shape[1] == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1]))
        labels = np.arange(probabilities.shape[1])
        return float(
            roc_auc_score(y_true, probabilities, labels=labels, multi_class="ovr")
        )
    except Exception:
        return float("nan")

def _primary_score(metrics, task_type):
    if task_type in ("regression", "ordinal"):
        return _first_finite(metrics.get("spearman"), metrics.get("pearson"))
    return _first_finite(metrics.get("roc_auc"), metrics.get("precision"))

def _first_finite(*values):
    for value in values:
        if value is not None and np.isfinite(value):
            return float(value)
    return float("nan")

def _json_safe_config(config):
    safe = dict(config)
    safe["hidden_layers"] = [int(value) for value in safe.get("hidden_layers", [])]
    for key in ("lr", "weight_decay", "dropout"):
        if key in safe:
            safe[key] = float(safe[key])
    for key in ("epochs", "batch_size"):
        if key in safe:
            safe[key] = int(safe[key])
    return safe

def _json_safe_metrics(metrics):
    safe = {}
    for key, value in metrics.items():
        if isinstance(value, str):
            safe[key] = value
        elif value is None:
            safe[key] = None
        else:
            safe[key] = None if not np.isfinite(value) else float(value)
    return safe

def _json_safe_row(row):
    safe = {}
    for key, value in row.items():
        if isinstance(value, (np.integer,)):
            safe[key] = int(value)
        elif isinstance(value, (np.floating,)):
            safe[key] = None if not np.isfinite(value) else float(value)
        elif isinstance(value, float):
            safe[key] = None if not np.isfinite(value) else value
        else:
            safe[key] = value
    return safe

def _model_output_dim(model):
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            return int(module.out_features)
    return None

def _sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")



def run_job(job):
    """Execute the same architecture-selection and split-scoring worker."""
    cache_path = Path(job["cache_path"])
    if job.get("skip_cache", False) and _architecture_search_cache_complete(
        cache_path, require_split_result=True
    ):
        print(f"Using completed result: {cache_path}")
        return
    run_architecture_resolution_job(job)


def collect_results(manifest):
    """Merge separate worker rows only after all requested jobs are complete."""
    missing = [job["cache_path"] for job in manifest["jobs"]
               if not _architecture_search_cache_complete(
                   job["cache_path"], require_split_result=True)]
    if missing:
        raise RuntimeError(
            f"{len(missing)} jobs are incomplete; run their .sh files first. "
            f"First missing result: {missing[0]}"
        )
    rows = [_normalize_result_row(_load_split_result_row(
        _split_result_cache_path(job["cache_path"]))) for job in manifest["jobs"]]
    path = Path(manifest["result_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=_RESULT_COLUMNS)
    frame.to_csv(path, index=False)
    _save_architecture_winners(rows, manifest["data_path"],
                               manifest["scoring_experiment_name"])
    print(f"Collected {len(rows)} rows: {path}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--job-index", type=int)
    action.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    with Path(args.manifest).expanduser().open() as handle:
        manifest = json.load(handle)
    if args.collect:
        collect_results(manifest)
    else:
        if not 0 <= args.job_index < len(manifest["jobs"]):
            parser.error("--job-index is outside this manifest's job range")
        run_job(manifest["jobs"][args.job_index])


if __name__ == "__main__":
    main()
