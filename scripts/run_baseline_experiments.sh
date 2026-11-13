#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

baseline_conda_env="${BASELINE_CONDA_ENV:-esm_env}"

run_python() {
  conda run -n "$baseline_conda_env" python "$@"
}

fitness_dataset_args=()
if [[ -n "${FITNESS_DATASETS:-}" ]]; then
  read -r -a fitness_datasets <<<"$FITNESS_DATASETS"
  fitness_dataset_args=(--datasets "${fitness_datasets[@]}")
fi

fitness_model_args=()
if [[ -n "${FITNESS_MODELS:-}" ]]; then
  read -r -a fitness_models <<<"$FITNESS_MODELS"
  fitness_model_args=(--models "${fitness_models[@]}")
fi

run_fitness_classic() {
  run_python code/calculate_fitness_for_plm_original_additivity.py \
    "${fitness_dataset_args[@]}" \
    "${fitness_model_args[@]}"
}

run_fitness_l2r() {
  run_python code/calculate_fitness_for_plm_multiple_decodings.py \
    "${fitness_dataset_args[@]}" \
    "${fitness_model_args[@]}" \
    --scoring_modes left_to_right \
    --resume_missing_models
}

run_fitness_single_mutations() {
  run_python code/calculate_fitness_for_plm_multiple_decodings.py \
    "${fitness_dataset_args[@]}" \
    "${fitness_model_args[@]}" \
    --scoring_modes one_mut \
    --resume_missing_models
}

run_fitness_pll() {
  run_python code/calculate_fitness_for_plm_multiple_decodings.py \
    "${fitness_dataset_args[@]}" \
    "${fitness_model_args[@]}" \
    --scoring_modes mutation_mask \
    --resume_missing_models
}

run_his2_controlled_regression_embeddings() {
  run_python code/train_classifiers_over_embeddings.py \
    --dataset_name his2 \
    --model_name prot_bert progen2-small \
    --mean_embeddings \
    --regression \
    --external_labels_column activity \
    --n_start 1 \
    --n_end 6 \
    --output_dir data/his2/controlled_extrapolation/regression/embeddings
}

run_his2_controlled_classification_embeddings() {
  run_python code/train_classifiers_over_embeddings.py \
    --dataset_name his2 \
    --model_name prot_bert progen2-small \
    --mean_embeddings \
    --external_labels_column activity \
    --n_start 1 \
    --n_end 6 \
    --output_dir data/his2/controlled_extrapolation/classification/embeddings
}

run_his2_controlled_regression_one_hot() {
  run_python code/train_classifiers_over_ohe.py \
    --dataset_name his2 \
    --regression \
    --external_labels_column activity \
    --n_start 1 \
    --n_end 6 \
    --output_dir data/his2/controlled_extrapolation/regression/one_hot
}

run_his2_controlled_classification_one_hot() {
  run_python code/train_classifiers_over_ohe.py \
    --dataset_name his2 \
    --external_labels_column activity \
    --n_start 1 \
    --n_end 6 \
    --output_dir data/his2/controlled_extrapolation/classification/one_hot
}

run_his2_subsamples_regression() {
  run_python code/train_regressors_over_embeddings_subsamples.py \
    --dataset_name his2 \
    --dataset_path data/his2/his2.csv \
    --model_names prot_bert progen2-small \
    --mean_embeddings \
    --regression \
    --external_labels_column activity \
    --n_samples 25 50 100 200 500 1000 \
    --iters 30 \
    --output_root data/his2/subsamples/collected_results
}

run_his2_subsamples_classification() {
  run_python code/train_regressors_over_embeddings_subsamples.py \
    --dataset_name his2 \
    --dataset_path data/his2/his2.csv \
    --model_names prot_bert progen2-small \
    --mean_embeddings \
    --classifier \
    --external_labels_column activity \
    --n_samples 25 50 100 200 500 1000 \
    --iters 30 \
    --output_root data/his2/subsamples/collected_results
}

run_pard3_controlled_regression_embeddings() {
  run_python code/train_classifiers_over_embeddings.py \
    --dataset_name pard3 \
    --model_name esm_35m esm_150m esm_650m \
    --mean_embeddings \
    --regression \
    --external_labels_column activity \
    --n_start 1 \
    --n_end 6 \
    --output_dir data/pard3/controlled_extrapolation/regression/embeddings
}

run_pard3_controlled_classification_embeddings() {
  run_python code/train_classifiers_over_embeddings.py \
    --dataset_name pard3 \
    --model_name esm_35m esm_150m esm_650m \
    --mean_embeddings \
    --external_labels_column activity \
    --n_start 1 \
    --n_end 6 \
    --output_dir data/pard3/controlled_extrapolation/classification/embeddings
}

run_pard3_controlled_regression_one_hot() {
  run_python code/train_classifiers_over_ohe.py \
    --dataset_name pard3 \
    --regression \
    --external_labels_column activity \
    --n_start 1 \
    --n_end 6 \
    --output_dir data/pard3/controlled_extrapolation/regression/one_hot
}

run_pard3_controlled_classification_one_hot() {
  run_python code/train_classifiers_over_ohe.py \
    --dataset_name pard3 \
    --external_labels_column activity \
    --n_start 1 \
    --n_end 6 \
    --output_dir data/pard3/controlled_extrapolation/classification/one_hot
}

run_pard3_subsamples_regression() {
  run_python code/train_regressors_over_embeddings_subsamples.py \
    --dataset_name pard3 \
    --dataset_path data/pard3/pard3.csv \
    --model_names esm_35m esm_150m esm_650m \
    --mean_embeddings \
    --regression \
    --external_labels_column activity \
    --n_samples 25 50 100 200 500 1000 \
    --iters 30 \
    --output_root data/pard3/subsamples/collected_results
}

run_pard3_subsamples_classification() {
  run_python code/train_regressors_over_embeddings_subsamples.py \
    --dataset_name pard3 \
    --dataset_path data/pard3/pard3.csv \
    --model_names esm_35m esm_150m esm_650m \
    --mean_embeddings \
    --classifier \
    --external_labels_column activity \
    --n_samples 25 50 100 200 500 1000 \
    --iters 30 \
    --output_root data/pard3/subsamples/collected_results
}

run_casp_controlled_regression_embeddings() {
  run_python code/train_classifiers_over_embeddings.py \
    --dataset_name casp \
    --model_name saprot saprot_aav_sites \
    --mean_embeddings \
    --regression \
    --external_labels_column activity \
    --n_start 1 \
    --n_end 6 \
    --output_dir data/casp/controlled_extrapolation/regression/embeddings
}

run_casp_controlled_classification_embeddings() {
  run_python code/train_classifiers_over_embeddings.py \
    --dataset_name casp \
    --model_name saprot saprot_aav_sites \
    --mean_embeddings \
    --external_labels_column activity \
    --n_start 1 \
    --n_end 6 \
    --output_dir data/casp/controlled_extrapolation/classification/embeddings
}

run_casp_controlled_regression_one_hot() {
  run_python code/train_classifiers_over_ohe.py \
    --dataset_name casp \
    --regression \
    --external_labels_column activity \
    --n_start 1 \
    --n_end 6 \
    --output_dir data/casp/controlled_extrapolation/regression/one_hot
}

run_casp_controlled_classification_one_hot() {
  run_python code/train_classifiers_over_ohe.py \
    --dataset_name casp \
    --external_labels_column activity \
    --n_start 1 \
    --n_end 6 \
    --output_dir data/casp/controlled_extrapolation/classification/one_hot
}

run_casp_subsamples_regression() {
  run_python code/train_regressors_over_embeddings_subsamples.py \
    --dataset_name casp \
    --dataset_path data/casp/casp.csv \
    --model_names saprot saprot_aav_sites \
    --model_output_names saprot saprot_aav_sites \
    --mean_embeddings \
    --regression \
    --external_labels_column activity \
    --n_samples 25 50 100 200 500 1000 \
    --iters 30 \
    --output_root data/casp/subsamples/collected_results
}

run_casp_subsamples_classification() {
  run_python code/train_regressors_over_embeddings_subsamples.py \
    --dataset_name casp \
    --dataset_path data/casp/casp.csv \
    --model_names saprot saprot_aav_sites \
    --model_output_names saprot saprot_aav_sites \
    --mean_embeddings \
    --classifier \
    --external_labels_column activity \
    --n_samples 25 50 100 200 500 1000 \
    --iters 30 \
    --output_root data/casp/subsamples/collected_results
}

experiments=(
  fitness-classic
  fitness-l2r
  fitness-single-mutations
  fitness-pll
  his2-controlled-regression-embeddings
  his2-controlled-classification-embeddings
  his2-controlled-regression-one-hot
  his2-controlled-classification-one-hot
  his2-subsamples-regression
  his2-subsamples-classification
  pard3-controlled-regression-embeddings
  pard3-controlled-classification-embeddings
  pard3-controlled-regression-one-hot
  pard3-controlled-classification-one-hot
  pard3-subsamples-regression
  pard3-subsamples-classification
  casp-controlled-regression-embeddings
  casp-controlled-classification-embeddings
  casp-controlled-regression-one-hot
  casp-controlled-classification-one-hot
  casp-subsamples-regression
  casp-subsamples-classification
)

list_experiments() {
  printf '%s\n' "${experiments[@]}"
}

usage() {
  cat <<'EOF'
Usage:
  scripts/run_baseline_experiments.sh list
  scripts/run_baseline_experiments.sh <experiment-name>
  scripts/run_baseline_experiments.sh all

Set BASELINE_CONDA_ENV to override the default conda environment (esm_env).
Set FITNESS_DATASETS and/or FITNESS_MODELS to restrict a fitness run. Values
are space-separated (for example, FITNESS_DATASETS="pard3" FITNESS_MODELS="esm_8m").
See docs/BASELINE_EXPERIMENTS.md for details and output paths.
EOF
}

run_experiment() {
  local experiment="$1"
  local function_name="run_${experiment//-/_}"
  if ! declare -F "$function_name" >/dev/null; then
    echo "Unknown experiment: $experiment" >&2
    usage >&2
    echo >&2
    echo "Available experiments:" >&2
    list_experiments >&2
    return 2
  fi
  echo "Running $experiment"
  "$function_name"
}

selection="${1:-}"
case "$selection" in
  list)
    list_experiments
    ;;
  all)
    for experiment in "${experiments[@]}"; do
      run_experiment "$experiment"
    done
    ;;
  "")
    usage >&2
    exit 2
    ;;
  *)
    run_experiment "$selection"
    ;;
esac
