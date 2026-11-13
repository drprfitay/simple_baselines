# Example runs for every `code/` module

This guide gives one concrete invocation or import example for every Python module in `code/`. Run commands from the `simple_baselines` repository root after activating the environment:

```bash
source .venv/bin/activate
```

Paths below are repository-relative. Commands that train models, generate embeddings, or score complete datasets can require a CUDA GPU and may take hours. Start with smaller sample counts, fewer epochs, or the bundled ESM2 8M model when testing a new installation.

Use `python code/<script>.py --help` for the complete option list of CLI-driven scripts. A few historical scripts have configuration blocks rather than a CLI; their sections identify the values to edit before execution. Support modules are imported by other scripts and are not standalone programs.

## Embedding and model modules

### `embedding_engine.py`

Generate chunked ESM2 8M embedding jobs for PARD3:

```bash
python code/embedding_engine.py \
  --action embed_parallel \
  --dataset_file data/pard3/pard3.csv \
  --sequence_colname full_seq \
  --model esm2_t6_8M_UR50D \
  --embedding_identifier esm_8m \
  --evaluation_path results/pard3/esm_8m \
  --chunk_size 500 \
  --internal_batch_size 20
```

The command prints a timestamped `jobs_path`. Run one generated chunk with:

```bash
python code/embedding_engine.py \
  --action specific_job \
  --jobs_path results/pard3/esm_8m/jobs_<timestamp>_parallel \
  --specific_job_i 0
```

Repeat `--specific_job_i` for the remaining rows in `key_dataset.csv`, use the LSF worker path on a configured cluster, or launch local worker subprocesses with:

```bash
python code/embedding_engine.py \
  --action execute_workers \
  --jobs_path results/pard3/esm_8m/jobs_YYYYMMDD_HHMMSS_parallel \
  --job_executing_type local \
  --n_workers 1
```

Local execution waits for all workers and records their logs, PIDs, commands, and return codes under the timestamped `job_execution_*` directory. Each worker loads a separate model copy, so increase `--n_workers` only when memory permits. `embed_sequences`, `embed_dataset`, and `embed_view_job_file` remain placeholders.

### `dataset.py`

This is an import-only module containing `PREDataset` and `PREActivityDataset`. A minimal cached-token dataset example is:

```python
from dataset import PREDataset
from plm_base import load_model, plm_init

plm_init(".")
wrapper = load_model("esm2_t6_8M_UR50D")
dataset = PREDataset(
    dataset_path="data/pard3/pard3.csv",
    indices=None,
    encoding_function=wrapper.get_encode(),
    encoding_identifier="esm_8m",
    sequence_column_name="full_seq",
    label_column_name="activity",
)
print(len(dataset), dataset[0][0].shape)
```

Run the snippet from an interactive Python session started with `PYTHONPATH=code`, or place it in a project script. `embedding_engine.py` is the normal end-to-end entry point.

### `embedder.py`

This is an import-only collection of evaluation/finalization functions used by `embedding_engine.py`. Verify the standard protein embedding hooks with:

```bash
PYTHONPATH=code python -c "from embedder import embeddings_evaluate_function, embeddings_finalize_function; print(embeddings_evaluate_function.__name__)"
```

The normal scientific run is the `embedding_engine.py --action embed_parallel` workflow above.

### `plm_base.py`

Load the bundled ESM2 8M checkpoint and run a CPU forward pass:

```bash
PYTHONPATH=code python -c "import torch; from plm_base import plm_init, load_model; plm_init('.'); w=load_model('esm2_t6_8M_UR50D'); x=torch.tensor([w.get_encode()('ACDE')]); logits,hidden=w.get_forward()(x); print(logits.shape, hidden.shape)"
```

This module must be initialized with the repository root before `load_model` is called.

### `trainer.py`

Fine-tune ESM2 8M with LoRA on an aligned FASTA or one-sequence-per-line MSA:

```bash
python code/trainer.py \
  --plm_name esm2_t6_8M_UR50D \
  --mode lora \
  --msa_path /path/to/aligned_msa.fasta \
  --output_path results/pard3/esm8m_lora_mlm \
  --epochs 3 \
  --batch_size 8 \
  --lr 1e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --device cuda
```

Replace `/path/to/aligned_msa.fasta`; an MSA is not bundled in the prepared payload.

### `example_run_jobs.py`

This import-only helper runs Python subprocesses and submits LSF jobs. A local smoke example is:

```bash
PYTHONPATH=code python -c "from example_run_jobs import run_python_script; run_python_script(['code/create_hamming_mds.py', '--help'])"
```

`run_lsf_job` requires `bsub` and a valid Conda environment on the cluster.

## Embedding collection and feature preparation

### `collect_gfp_embeddings.py`

Collect GFP ESM2 8M chunks from `results/gfp/esm_8m/evaluations` into mutation-order tensors under `data/gfp/new_embeddings/esm_8m`:

```bash
python code/collect_gfp_embeddings.py \
  --input_csv data/gfp/gfp_dataset_10mut.csv \
  --model_name esm_8m \
  --hidden_dim_size 320 \
  --mutations_to_select 0 1 2 3 4 5 6 7 8 9 10
```

If embeddings contain the full sequence, add `--positions_to_select <positions...>`. Model-position indexing is one-based and includes the BOS offset.

### `collect_protgym_embeddings.py`

Collect PARD3 ESM2 8M chunks from `results/pard3/esm_8m/evaluations` into `data/pard3/embeddings/esm_8m`:

```bash
python code/collect_protgym_embeddings.py \
  --input_csv data/pard3/pard3.csv \
  --dataset_name pard3 \
  --model_name esm_8m \
  --hidden_dim_size 320
```

The input CSV must contain `num_muts`, and each completed chunk must contain `train/embeddings.pt`, `train/y_value.pt`, and `train/indices.pt`.

### `merge_embedding_chunks.py`

Merge and validate all PARD3 ESM2 8M mutation-order tensors:

```bash
python code/merge_embedding_chunks.py \
  --dataset_csv data/pard3/pard3.csv \
  --y_col activity \
  --embedding_dirs data/pard3/embeddings/esm_8m \
  --max_nmut 10
```

This writes `embeddings.pt`, `y_values.pt`, and `indices.pt` into each supplied embedding directory. Explicit `--dataset_csv` and `--embedding_dirs` are recommended.

### `subset_embedding_chunks.py`

Preview splitting a parent HIS embedding/cache payload into HIS2 and HIS5 subsets:

```bash
python code/subset_embedding_chunks.py \
  --parent_dir /path/to/data/his \
  --subset_dirs data/his2 data/his5 \
  --parent_csv /path/to/data/his/his.csv \
  --embedding_names esm_8m esm_35m \
  --copy_tokenized \
  --first_col_his L7 \
  --last_col_his D211 \
  --first_col_his2 N137 \
  --last_col_his2 F163 \
  --first_col_his5 R181 \
  --last_col_his5 D211 \
  --dry_run
```

Remove `--dry_run` after reviewing the reported mappings. The combined `data/his` parent payload is not present in this checkout, so replace the two `/path/to` values with its location.

### `mine_foldseek_tokens.py`

Generate aligned Foldseek 3Di tokens for PARD3:

```bash
python code/mine_foldseek_tokens.py \
  pard3 \
  data/pard3/pard3.csv \
  data/pard3/pard3.pdb
```

This positional interface accepts dataset name, CSV path, and structure path. It requires the `foldseek` executable and writes the aligned-token pickle and text files beside the dataset CSV.

### `collect_scoring_features_for_representation_analyses.py`

Build a PARD3 feature dictionary containing one-hot features, activities, and mean ESM2 8M embeddings:

```bash
python code/collect_scoring_features_for_representation_analyses.py \
  --embedding_dir data/pard3/embeddings/esm_8m \
  --embedding_key esm_8m \
  --sequence_df_path data/pard3/pard3.csv \
  --first_col L48 \
  --last_col R82 \
  --activity_col activity \
  --num_muts_col num_muts \
  --mean \
  --output data/pard3/external_scoring/example_features.pt
```

Use repeated values after `--embedding_dir` and `--embedding_key` to collect several models in one file. Add `--pssm_path data/pard3/pssm_scores.csv` when a matching PSSM table is available.

## Baseline training and model selection

### `train_classifiers_over_embeddings.py`

Controlled-extrapolation regression over PARD3 pLM embeddings:

```bash
python code/train_classifiers_over_embeddings.py \
  --model_name esm_8m \
  --dataset_name pard3 \
  --mean_embeddings \
  --regression \
  --external_labels_column activity \
  --n_start 1 \
  --n_end 6
```

The current implementation uses `range(n_start, n_end)`, so `--n_end 6` trains through mutation order 5. Use `--n_end 7` if order 6 must also be a training threshold.

Run the default MLP regressor on mean-pooled CASP SaProt AAV-site embeddings,
training on single mutants and evaluating mutation orders 2 through 18:

```bash
conda run -n esm_env python code/train_classifiers_over_embeddings.py \
  --model_name saprot_aav_sites \
  --dataset_name casp \
  --mean_embeddings \
  --regression \
  --external_labels_column activity \
  --n_start 1 \
  --n_end 2 \
  --output_dir data/casp/controlled_extrapolation/saprot
```

Embedding model directories and available mutation orders are discovered from
`data/<dataset>/embeddings/<model>/`. The default MLP uses one 100-unit hidden
layer and at most 50 iterations; increase `--niters` if it does not converge.
Without `--output_dir`, regression results default to
`data/<dataset>/controlled_extrapolation/<model>/`.

### `train_classifiers_over_ohe.py`

Matching controlled-extrapolation regression over one-hot encoding:

```bash
python code/train_classifiers_over_ohe.py \
  --dataset_name pard3 \
  --external_labels_column activity \
  --regression \
  --n_start 1 \
  --n_end 6
```

The accepted option is `--dataset_name`, not `--dataset`. As above, `n_end` is exclusive in the current implementation.

For CASP, use the default MLP regressor and save files named
`one_hot_evaluation_train_on_<n>_mlp.csv`
under `data/casp/controlled_extrapolation/one_hot/`:

```bash
conda run -n esm_env python code/train_classifiers_over_ohe.py \
  --dataset_name casp \
  --regression \
  --external_labels_column activity \
  --n_start 1 \
  --n_end 6 \
  --niters 200
```

Available mutation orders are read from the dataset, so CASP is evaluated
through order 18. Use `--hl 100,50`, `--alpha`, `--op`, `--lr`, and `--niters`
to override the MLP architecture and optimizer. The architecture is included
in each output CSV's `grid_*` columns rather than its filename.

### `train_regressors_over_embeddings_subsamples.py`

Compare PARD3 pLM embeddings and OHE on matched random training subsets:

```bash
python code/train_regressors_over_embeddings_subsamples.py \
  --dataset_name pard3 \
  --external_labels_column activity \
  --model_names esm_8m \
  --n_samples 100 200 500 1000 5000 \
  --iters 10 \
  --output_csv data/pard3/example_regression_result_by_training_samples.csv
```

Regression is the default. Use `--classifier` for classification and `--mean_embeddings` to mean-pool position embeddings instead of flattening them.

For the collected CASP SaProt AAV-site embeddings:

```bash
conda run -n esm_env python code/train_regressors_over_embeddings_subsamples.py \
  --dataset_name casp \
  --dataset_path data/casp/casp.csv \
  --external_labels_column activity \
  --model_names saprot_aav_sites \
  --model_output_names saprot \
  --mean_embeddings \
  --n_samples 100 200 500 1000 5000 \
  --iters 10 \
  --output_root data/casp/subsamples
```

This single matched-split run writes one-hot-only columns to
`data/casp/subsamples/one_hot/` and SaProt-only columns to
`data/casp/subsamples/saprot/`. Architecture-specific filenames prevent
different MLP settings from overwriting each other. Override the two models
independently with the `--ohe_*` and `--llm_*` architecture options.

### `train_classifiers_over_subsamples_new.py`

Run the parameterized GFP random-split classification comparison:

```bash
python code/train_classifiers_over_subsamples_new.py \
  --dataset_name gfp \
  --embedding_model esm_8m \
  --n_samples 100 500 1000 \
  --iters 10 \
  --ohe_hl 200 \
  --llm_hl 64 \
  --output_csv data/gfp/subsamples/example \
  --shuffle_labels
```

Despite exposing `--dataset_name`, this script currently uses GFP-specific paths, columns, and labels. `--output_csv` is treated as an output directory.

### `train_classifiers_over_subsamples.py`

This is the legacy, configuration-driven GFP subsampling reproduction:

```bash
python code/train_classifiers_over_subsamples.py
```

It has no CLI and immediately runs the hard-coded GFP/ESM35M experiment over many sample sizes and 20 iterations, writing under `data/gfp/subsamples/`. Prefer `train_classifiers_over_subsamples_new.py` for new experiments.

### `exetnsive_model_selection.py`

Run one PARD3 controlled-extrapolation model-selection/scoring job using OHE and ESM2 8M features:

```bash
python code/exetnsive_model_selection.py \
  --dataset_path data/pard3/pard3.csv \
  --cache_path data/pard3/scoring_cache \
  --onehot \
  --embedding esm_8m \
  --regressor \
  --activity_column_name activity \
  --num_muts_colname num_muts \
  --first_mutation_col L48 \
  --last_mutation_col R82 \
  --train_type mutation \
  --min_muts 1 \
  --max_muts 10 \
  --min_train_muts 1 \
  --max_train_muts 5 \
  --max_test_muts 10 \
  --architecture_resolution full \
  --score_k 1 \
  --mean_embeddings \
  --load_all \
  --create_missing_splits \
  --device cuda
```

The historical filename typo, `exetnsive`, is intentional. For a complete architecture-resolution and scoring grid, use the job generator below.

### `generate_jobs_for_extensive_model_selection.py`

Generate the complete PARD3 architecture-resolution and scoring job set:

```bash
python code/generate_jobs_for_extensive_model_selection.py \
  --output_dir data/pard3/example_scoring_jobs \
  --sequence_path data/pard3/pard3.csv \
  --activity_column_name activity \
  --num_muts_colname num_muts \
  --regressor \
  --min_nmuts 1 \
  --max_nmuts 10 \
  --min_train_muts 1 \
  --max_train_muts 5 \
  --max_test_muts 10 \
  --first_col L48 \
  --last_col R82 \
  --embeddings esm_8m \
  --train_sizes 25 50 100 200 \
  --niters 10 \
  --load_all \
  --device cuda
```

This writes preflight, architecture-resolution, and post-resolution scoring shell scripts plus `scoring_manifest.csv`. Inspect the generated commands before launching them.

The generator does not execute the scripts. Because this example uses relative
data paths, launch all three phases from the repository root and in dependency
order:

```bash
cd /path/to/simple_baselines
JOB_DIR=data/pard3/example_scoring_jobs

bash "$JOB_DIR/preflight/00_make_splits.sh"

for job in "$JOB_DIR"/resolve_architectures/*.sh; do
  bash "$job"
done

for job in "$JOB_DIR"/score_after_resolving/*.sh; do
  bash "$job"
done
```

Do not `cd` into one of the generated subdirectories before running a script:
relative inputs such as `data/pard3/pard3.csv` would then resolve underneath
that subdirectory. Architecture and scoring scripts can be submitted to a
scheduler instead of being run sequentially, but scoring must wait for its
corresponding architecture-resolution jobs. If `--train_sizes` is omitted, the
generator's notice that random jobs were skipped is informational rather than
an error.

Preflight creates split JSON files only. Every scoring command includes
`--require_resolved_architecture`, so its matching architecture job must first
produce `best_architecture.json`. A missing file under
`scoring_cache/<label>/architectures/<feature>/<resolution>/` means the
architecture phase was skipped or did not complete successfully; removing the
requirement merely attempts the same resolution inside the scoring process.

For the current CASP/SaProt files, note that `casp.csv` reaches mutation order
18 but `embeddings/saprot` has chunks only for orders 0–10. Full-resolution
splits currently include the complete CSV and consequently require SaProt
chunks 11–18. Until those chunks are generated or full split creation is made
to honor the maximum mutation bound, use the generated `less_than_K` and
`equal_to_K` architecture/scoring pairs. For K=1, for example:

```bash
bash data/casp/scoring_jobs/saprot/resolve_architectures/0002_mean_saprot_less_than_K_K_1.sh
bash data/casp/scoring_jobs/saprot/score_after_resolving/0006_mean_saprot_mutation_less_than_K_train_K_1.sh
```

### `collect_model_selection_scoring_results.py`

Collect the standard five groups of PARD3 scorer output:

```bash
python code/collect_model_selection_scoring_results.py \
  --cache_path data/pard3/scoring_cache/regressor \
  --output_dir data/pard3/scoring_cache/regressor/results/collected
```

The outputs are `by_mutation_val_less.csv`, `by_mutation_val_eq.csv`, `by_mutation_val_full.csv`, `random_full.csv`, and `random_internal.csv`.

### `transfer_learning_across_layers.py`

Run the pLM layer/pooling transfer-learning grid for PARD3, training on mutation order 1 or lower:

```bash
python code/transfer_learning_across_layers.py \
  --dataset pard3 \
  --train_muts 1 \
  --delta
```

The script reads `data/merged_datasets/oracle_transfer_learning/pard3_for_tl.pkl`. Remove `--delta` for regular rather than WT-delta embeddings.

### `transfer_learning_ohe_grid.py`

Run the matching OHE transfer-learning grid:

```bash
python code/transfer_learning_ohe_grid.py \
  --dataset pard3 \
  --train_muts 1
```

It uses the same `pard3_for_tl.pkl` prepared input and writes its CSV beside the other oracle transfer-learning results.

### `lora_controlled_extrap.py`

Fine-tune ESM2 8M with LoRA on PARD3 variants of mutation order 1 or lower, then evaluate higher orders:

```bash
python code/lora_controlled_extrap.py \
  --dataset pard3 \
  --model_name esm_8m \
  --train_mutations 1 \
  --output_path data/merged_datasets/lora_finetune/pard3_esm8m_example.csv \
  --batch_size 8 \
  --eval_batch_size 16 \
  --max_epochs 10 \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing \
  --eval_bf16 \
  --device cuda
```

For a classification version, add `--classifier`. The script also writes a matching `_predictions.csv` file.

## Zero-shot scoring and epistasis

### `calculate_fitness_for_plm_multiple_decodings.py`

This is the current multi-mode zero-shot scoring entry point:

```bash
python code/calculate_fitness_for_plm_multiple_decodings.py \
  --datasets pard3 \
  --models esm_8m \
  --scoring_modes mutation_mask \
  --forward_batch_size 60 \
  --variant_score_batch_size 2000
```

The main controls are:

- `SCORING_MODES`: `mutation_mask` (PLL), `one_mut`, or `left_to_right`;
- `full_model_names`: models to score;
- `USE_SAPROT`: whether to include SaProt;
- batch sizes and mixed-precision settings for the available GPU.

Use `--datasets` and `--models` to expand the run. Each dataset needs cached token tensors under `data/<dataset>/<dataset>_cache/misc/`. Results are written under `results/<dataset>/fitness_results/`. This replaces the generic historical command `python calculate_fitness_for_plm.py`, because that filename is not present in this repository.

`one_mut` writes `new_om_normed_fitness_all.csv` and `mutation_mask` writes
`pll_normed_fitness_all.csv`; both use the unsuffixed canonical model names.
`left_to_right` writes `new_l2r_normed_fitness_all.csv` and scores only
`progen2-medium_left_to_right` and `progen2-small_left_to_right`.

### `calculate_fitness_for_plm_original_additivity.py`

Run the original additive masked-position baseline:

```bash
python code/calculate_fitness_for_plm_original_additivity.py \
  --datasets pard3 \
  --models esm_8m
```

It uses the same token caches and writes `fitness_all.csv` and `normed_fitness_all.csv` under each dataset's `results/<dataset>/fitness_results/` directory. Pass additional values to `--datasets` or `--models` to expand the run.

### `detect_epistasis.py`

Run binary and classical epistasis detection:

```bash
python code/detect_epistasis.py \
  --datasets pard3 \
  --max_mutations 10 \
  --percentile 0.5 \
  --output_dir data/merged_datasets/linear_baslines/pard3_example
```

Pass several names after `--datasets` for a combined run. Results are written beneath the selected output directory; the historical `linear_baslines` spelling is preserved.

## Representation and feature-combination analyses

### `generate_scoring_for_feature_comp.py`

Generate FuncScape jobs for single and combined PARD3 representations:

```bash
python code/generate_scoring_for_feature_comp.py \
  --funcscape_root /path/to/FuncScape \
  --features_pt data/pard3/external_scoring/new_features.pt \
  --output_dir data/pard3/example_feature_scoring_jobs \
  --combo onehot \
  --combo onehot partial_esm_35m_mean \
  --y_key activity \
  --subset_sizes 100 200 \
  --n_iterations 3 \
  --device cuda
```

Replace `/path/to/FuncScape`. Run the generated job scripts, then run `collect_results.bash` in the output directory.

### `mixed_representation_analysis_gen_jobs.py`

Generate standalone-style mixed-representation jobs:

```bash
python code/mixed_representation_analysis_gen_jobs.py \
  data/pard3/external_scoring/new_features.pt \
  --task-name pard3_mixed_example \
  --y-key activity \
  --output-dir data/pard3/example_mixed_jobs \
  --no-all-singles \
  --no-all-doubles \
  --custom-combination onehot \
  --custom-combination onehot partial_esm_35m_mean \
  --subset-sizes 100 200 \
  --n-iterations 3 \
  --architecture-max-epochs 200 \
  --device cuda
```

The generator uses the bundled `run_feature_comp_analysis.py` implementation. A compatible external module named `run_analysis` on `PYTHONPATH` is still accepted for backward compatibility.

### `run_feature_comp_analysis.py`

Run one job from a JSON manifest generated by `mixed_representation_analysis_gen_jobs.py`:

```bash
python code/run_feature_comp_analysis.py \
  --manifest data/pard3/example_mixed_jobs/manifest.json \
  --job-index 0
```

After every job is complete, collect the rows with:

```bash
python code/run_feature_comp_analysis.py \
  --manifest data/pard3/example_mixed_jobs/manifest.json \
  --collect
```

Job indices are zero-based.

### `representation_geometry.py`

Run a compact PARD3 reconstruction analysis using one representation:

```bash
python code/representation_geometry.py \
  --dataset_name pard3 \
  --features_pt data/pard3/external_scoring/new_features.pt \
  --keys partial_esm_35m_mean \
  --sample_size 1000 \
  --rounds 10 \
  --fixed_alpha 100 \
  --no-save_reconstruction_examples \
  --save_csv data/pard3/external_scoring/representation_geometry_example.csv
```

Use `--controlled` for mutation-order controlled extrapolation. The checked-in defaults retain the historical HIS5 configuration, so explicit paths and feature keys are recommended.

### `normalization_method_sensitivity.py`

Run the PARD3 controlled-extrapolation normalization comparison:

```bash
python code/normalization_method_sensitivity.py \
  --mode transfer \
  --dataset_name pard3 \
  --repo_root . \
  --batch_size 128 \
  --device cuda \
  --output_dir data/pard3/normalization_sanity
```

Reuse the generated raw embedding cache in a later run with `--reuse_embeddings`. The reconstruction audit is a second mode:

```bash
python code/normalization_method_sensitivity.py \
  --mode reconstruction \
  --dataset_name pard3 \
  --repo_root . \
  --embedding_cache data/pard3/normalization_sanity/pard3_esm2_8m_raw_designed_embeddings.pt \
  --sample_size 1000 \
  --n_rounds 20 \
  --include_geometry \
  --output_dir data/pard3/reconstruction_normalization_audit
```

## Geometry and visualization

### `create_hamming_mds.py`

Create a Hamming-distance matrix, two-dimensional MDS coordinates, and a plot for 100 HIS2 variants:

```bash
python code/create_hamming_mds.py \
  --dataset_name his2 \
  --first_col N137 \
  --last_col F163 \
  --N 100 \
  --output_type both \
  --output_path data/his2/mds/hamming_100.npy \
  --embedding_path data/his2/mds/hamming_100_mds.npy \
  --figure_path data/his2/mds/hamming_100.png
```

The wild type is added automatically when the dataset contains a `num_muts == 0` row.

### `tsne_candidate_libraries.py`

Plot example candidate libraries on the PARD3 ESM2 8M embedding landscape:

```bash
python code/tsne_candidate_libraries.py \
  --dataset pard3 \
  --model esm_8m \
  --embeddings_path data/pard3/embeddings/esm_8m/embeddings.pt \
  --library_index_mode variant_rows \
  --best_indices 0 1 2 \
  --worst_indices 3 4 5 \
  --pssm_indices 6 7 8 \
  --plm_indices 9 10 11 \
  --K 3 \
  --sample_size 1000 \
  --perplexity 30 \
  --output_path data/tsne_embeddings/pard3_example.svg
```

The four index lists are experiment inputs, not universal defaults; replace them with the nominated variant rows. Use `feature_columns` or `mutation_columns` mode when the lists instead refer to OHE columns.

## Shared analysis utilities

### `utils.py`

This is an import-only registry and analysis-helper module. Inspect the portable dataset registry and load PARD3 labels with:

```bash
PYTHONPATH=code python -c "from utils import DATASET_PATHS, get_labels; print(DATASET_PATHS['pard3']); y=get_labels('pard3'); print(len(y))"
```

Most training, scoring, and notebook code imports this module automatically.

### `utils_for_analysis.py`

This is a backward-compatible re-export of `utils.py`. Verify the shared registry identity with:

```bash
PYTHONPATH=code python -c "import utils, utils_for_analysis as ufa; print(ufa.DATASET_PATHS is utils.DATASET_PATHS)"
```

New code can import `utils` directly; the compatibility name is retained for historical scripts and notebooks.

## Coverage checklist

The sections above cover all 35 modules currently under `code/`:

```text
calculate_fitness_for_plm_multiple_decodings.py
calculate_fitness_for_plm_original_additivity.py
collect_gfp_embeddings.py
collect_model_selection_scoring_results.py
collect_protgym_embeddings.py
collect_scoring_features_for_representation_analyses.py
create_hamming_mds.py
dataset.py
detect_epistasis.py
embedder.py
embedding_engine.py
example_run_jobs.py
exetnsive_model_selection.py
generate_jobs_for_extensive_model_selection.py
generate_scoring_for_feature_comp.py
lora_controlled_extrap.py
merge_embedding_chunks.py
mine_foldseek_tokens.py
mixed_representation_analysis_gen_jobs.py
normalization_method_sensitivity.py
plm_base.py
representation_geometry.py
run_feature_comp_analysis.py
subset_embedding_chunks.py
train_classifiers_over_embeddings.py
train_classifiers_over_ohe.py
train_classifiers_over_subsamples.py
train_classifiers_over_subsamples_new.py
train_regressors_over_embeddings_subsamples.py
trainer.py
transfer_learning_across_layers.py
transfer_learning_ohe_grid.py
tsne_candidate_libraries.py
utils.py
utils_for_analysis.py
```
