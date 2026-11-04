#!/usr/bin/env python3
r"""Generate plain Bash jobs for the standalone mixed-representation analysis.

Python usage (the score_queued call from FuncScape can be used here):
    from mixed_representation_analysis_gen_jobs import score_queued
    jobs = score_queued(
        "notebooks/data/gfp/external_scoring/new_features.pt",
        task_name="gfp_ext_scoring_pssm", score_parts="subsets",
        y_key="inactive", discrete=True,
        all_singles=False, all_doubles=False, combine_all=False,
        custom_combinations=[["onehot"], ["onehot", "pssm_score"]],
        architecture_resample_fractions=(0.2, 0.5, 0.8),
        architecture_n_resamples_per_fraction=3, architecture_n_seeds=1,
        architecture_max_epochs=200, architecture_eval_every=10,
        subset_sizes=[500, 1000], n_iterations=10, skip_if_exists=False,
        job_type="gpu", job_length="short", conda_env="esm_env",
    )

CLI usage (repeat --custom-combination for each combination):
    python code/mixed_representation_analysis_gen_jobs.py \
        notebooks/data/gfp/external_scoring/new_features.pt \
        --task-name gfp_ext_scoring_pssm --y-key inactive --discrete \
        --no-all-singles --no-all-doubles --subset-sizes 500 1000 \
        --custom-combination onehot --custom-combination onehot pssm_score \
        --architecture-resample-fractions 0.2 0.5 0.8 \
        --architecture-n-resamples-per-fraction 3 --architecture-max-epochs 200 \
        --n-iterations 10 --no-skip-if-exists --conda-env esm_env

Only the requested score_parts="subsets" path is supported. The generator
prepares shared outer splits and training-only architecture resamples once.
Each .sh runs selection, retraining and held-out scoring for one combination
and split. run_all.sh runs jobs sequentially and collects results; alternatively
run jobs individually, then collect_results.sh. Nothing is auto-submitted.

Jobs default to code/mixed_representation_jobs/<task_name>/ beside these scripts.
Results retain FuncScape's layout beside features.pt, under scoring_experiment/.
job_type/job_length are compatibility metadata, not scheduler directives.
conda_env="current" uses the generator's Python interpreter; other names use conda run.
To reuse completed caches, set architecture_refresh_cache=False and
skip_if_exists=True (FuncScape defaults architecture_refresh_cache to True).
Uses the bundled run_feature_comp_analysis.py implementation by default. A
compatible external run_analysis.py on PYTHONPATH is still accepted for
backward compatibility.
"""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path


def _write_script(path, command):
    path.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
        + command + '\n'
    )
    path.chmod(0o755)


def score_queued(
    data_path, *, task_name=None, score_parts="subsets", y_key="activity",
    results_file_name="scoring_results.csv", scoring_experiment_name="scoring_experiment",
    all_singles=True, all_doubles=True, combine_all=False, custom_combinations=None,
    train_breaks=5, n_iterations=3, subset_sizes=None,
    extremes=None, ordinal=None, discrete=False, precision_at_k=None,
    architecture_resample_fractions=(0.15, 0.25, 0.5, 0.75, 0.8),
    architecture_n_resamples_per_fraction=10, architecture_n_seeds=1,
    architecture_max_epochs=500, architecture_eval_every=10,
    architecture_refresh_cache=True, skip_if_exists=True, refresh=False,
    architecture_grid=None, config_grid=None, standardize_features=False,
    random_seed=0, device=None, train_indices=None, test_indices=None,
    all_indices=None, num_muts=None, max_muts_in_train=None, refresh_splits=False,
    job_type="regular", job_length="short", conda_env="current",
    output_dir=None, python_executable=None,
):
    """Write jobs without running training; return paths to scripts and results.

    Preserves score_queued's subset defaults and combo/split seed ordering.
    The only execution change is plain .sh files instead of FuncScape's queue.
    output_dir controls shell files; scoring_experiment_name controls analysis
    outputs. Relative input paths resolve against the generator's working dir.
    """
    if score_parts != "subsets":
        raise ValueError('This standalone script supports score_parts="subsets" only.')
    if not isinstance(discrete, bool):
        raise ValueError("discrete must be a Python bool; use discrete=True for GFP.")
    if int(architecture_max_epochs) < 1 or int(architecture_eval_every) < 1:
        raise ValueError("architecture_max_epochs and architecture_eval_every must be positive.")
    if int(architecture_n_seeds) < 1 or int(architecture_n_resamples_per_fraction) < 1:
        raise ValueError("Architecture seeds and resamples must be positive.")
    architecture_resample_fractions = tuple(architecture_resample_fractions)
    if not architecture_resample_fractions or any(
        not 0 < float(value) < 1 for value in architecture_resample_fractions
    ):
        raise ValueError("architecture_resample_fractions must contain values between 0 and 1.")

    try:
        import run_analysis as analysis
    except ModuleNotFoundError:
        import run_feature_comp_analysis as analysis

    data_path = Path(data_path).expanduser().resolve()
    features = analysis._load_features(data_path, y_key)
    target_kwargs = {"extremes": extremes, "ordinal": ordinal, "discrete": discrete}
    y, task_type, target_scheme = analysis._prepare_target(features[y_key], **target_kwargs)
    feature_names = [name for name in features if name != y_key]
    combos = analysis._build_feature_combinations(
        feature_names, all_singles, all_doubles, custom_combinations, combine_all,
        feature_ndims=analysis._feature_ndims(features, feature_names),
    )
    analysis._validate_split_args(
        train_indices, test_indices, all_indices, subset_sizes, num_muts, max_muts_in_train)
    splits_payload = analysis.resolve_or_create_scoring_splits(
        data_path=data_path, target_scheme=target_scheme, n_rows=len(y),
        train_indices=train_indices, test_indices=test_indices, all_indices=all_indices,
        train_breaks=train_breaks, n_iterations=n_iterations, subset_sizes=subset_sizes,
        architecture_resample_fractions=architecture_resample_fractions,
        architecture_n_resamples_per_fraction=architecture_n_resamples_per_fraction,
        num_muts=num_muts, max_muts_in_train=max_muts_in_train, random_seed=random_seed,
        refresh_splits=refresh_splits or refresh,
        scoring_experiment_name=scoring_experiment_name,
    )
    splits = splits_payload["splits"]
    if not splits:
        raise ValueError("No nonempty train/test splits were created; check subset sizes and row count.")
    grid = architecture_grid or config_grid or analysis.DEFAULT_ARCHITECTURE_GRID
    grid = [analysis._json_safe_config(item) for item in grid]
    result_path = analysis._resolve_results_path(
        str(Path(results_file_name).expanduser()), data_path, scoring_experiment_name)
    task_name = task_name or f"{data_path.parent.name}_mixed_representations"
    output_dir = (Path(output_dir).expanduser().resolve() if output_dir else
                  Path(__file__).resolve().parent / "mixed_representation_jobs" /
                  analysis._sanitize_name(task_name))
    refresh_cache = architecture_refresh_cache or not skip_if_exists or refresh
    jobs = []
    for combo_idx, combo in enumerate(combos):
        for split_idx, split in enumerate(splits):
            cache_path = analysis._architecture_search_cache_path(
                data_path, target_scheme, combo, split["split_id"],
                scoring_experiment_name, standardize_features=standardize_features)
            jobs.append({
                "data_path": str(data_path), "y_key": y_key, "target_kwargs": target_kwargs,
                "combo_info": combo, "split_file_path": splits_payload["path"],
                "split_id": split["split_id"], "cache_path": str(cache_path),
                "result_path": str(result_path), "architecture_grid": grid,
                "max_epochs": int(architecture_max_epochs), "eval_every": int(architecture_eval_every),
                "resample_fractions": [float(value) for value in architecture_resample_fractions],
                "n_resamples_per_fraction": int(architecture_n_resamples_per_fraction),
                "n_seeds": int(architecture_n_seeds), "standardize_features": bool(standardize_features),
                "random_seed": int(random_seed + combo_idx * 1000 + split_idx),
                "precision_at_k": precision_at_k, "device": device, "verbose": True,
                "skip_cache": not refresh_cache,
            })
    manifest = {
        "data_path": str(data_path), "result_path": str(result_path),
        "scoring_experiment_name": scoring_experiment_name, "task_name": task_name,
        "job_type": job_type, "job_length": job_length, "jobs": jobs,
    }
    # Serialize before writing shell files, so invalid parameters fail early.
    manifest_text = json.dumps(manifest, indent=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(manifest_text + "\n")
    runner_relative = os.path.relpath(Path(analysis.__file__).resolve(), output_dir)
    executable = python_executable or (sys.executable if conda_env in (None, "current") else "python")
    prefix = [executable, "-u"]
    if conda_env not in (None, "current"):
        prefix = ["conda", "run", "--no-capture-output", "-n", conda_env, *prefix]
    command = (shlex.join(prefix) + ' "$SCRIPT_DIR"/' + shlex.quote(runner_relative)
               + ' --manifest "$SCRIPT_DIR/manifest.json"')
    scripts = []
    for index, job in enumerate(jobs):
        if not refresh_cache and analysis._architecture_search_cache_complete(
            job["cache_path"], require_split_result=True
        ):
            continue
        filename = (f"job_{index + 1:04d}__"
                    f"{analysis._sanitize_name(job['combo_info']['name'])}__{job['split_id']}.sh")
        path = output_dir / filename
        _write_script(path, command + f" --job-index {index}")
        scripts.append(path)
    collect_script = output_dir / "collect_results.sh"
    _write_script(collect_script, command + " --collect")
    run_all_script = output_dir / "run_all.sh"
    _write_script(run_all_script, "\n".join(
        'bash "$SCRIPT_DIR"/' + shlex.quote(path.name)
        for path in [*scripts, collect_script]))
    print(f"Wrote {len(scripts)} jobs ({len(jobs)} total results expected): {output_dir}")
    print(f"Run sequentially: bash {shlex.quote(str(run_all_script))}")
    print(f"Results after collection: {result_path}")
    return {"scripts": [str(path) for path in scripts], "manifest": str(manifest_path),
            "run_all": str(run_all_script), "collect_results": str(collect_script),
            "result_path": str(result_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_path")
    for name, default in (("task_name", None), ("y_key", "activity"),
                          ("output_dir", None), ("conda_env", "current"),
                          ("device", None), ("scoring_experiment_name", "scoring_experiment"),
                          ("results_file_name", "scoring_results.csv")):
        parser.add_argument("--" + name.replace("_", "-"), default=default)
    for name, default in (("n_iterations", 3), ("train_breaks", 5),
                          ("architecture_max_epochs", 500), ("architecture_eval_every", 10),
                          ("architecture_n_resamples_per_fraction", 10),
                          ("architecture_n_seeds", 1), ("random_seed", 0)):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    for name, default in (("all_singles", True), ("all_doubles", True), ("combine_all", False),
                          ("discrete", False), ("skip_if_exists", True),
                          ("architecture_refresh_cache", True), ("standardize_features", False)):
        parser.add_argument("--" + name.replace("_", "-"),
                            action=argparse.BooleanOptionalAction, default=default)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--subset-sizes", type=int, nargs="+")
    parser.add_argument("--architecture-resample-fractions", type=float, nargs="+",
                        default=(0.15, 0.25, 0.5, 0.75, 0.8))
    parser.add_argument("--custom-combination", dest="custom_combinations", nargs="+", action="append")
    score_queued(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
