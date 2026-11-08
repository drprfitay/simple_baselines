```text
 __  _  _   _  ___ _    ___       ___  _   __  ___  _    _  _  _  ___  __ 
/ _|| || \_/ || o \ |  | __|     | o )/ \ / _|| __|| |  | || \| || __|/ _|
\_ \| || \_/ ||  _/ |_ | _|      | o \ o |\_ \| _| | |_ | || \\ || _| \_ \
|__/|_||_| |_||_| |___||___|     |___/_n_||__/|___||___||_||_|\_||___||__/
                                                                      
```

# Protein Fitness · Simple Baselines

This repository is the portable, minimal-change extraction of the protein-fitness workflows from `fitness_learning`. It contains the original baseline and embedding scripts, the seven `lean_*` notebooks, local PLM implementations/checkpoints, and the prepared analysis data available in the source workspace.

The scientific logic is intentionally unchanged. Repository-specific edits are limited to resolving paths from this checkout, making notebook imports independent of launch directory, and adding installation, documentation, and verification files.

## Install

Python 3.10 or 3.11 is required. From a fresh clone:

```bash
git clone <repository-url> simple_baselines
cd simple_baselines
./install.sh
source .venv/bin/activate
```

The installer creates `.venv`, installs [requirements.txt](requirements.txt), installs the flat modules in `code/` as editable modules, and runs a smoke test. To reuse a prepared Conda environment:

```bash
./install.sh --no-venv --skip-deps --python "$CONDA_PREFIX/bin/python" --model-check
```

Run a strict local check, including the data payload and bundled ESM2 8M model:

```bash
python scripts/verify_install.py --strict-data --model-check
```

For CUDA-specific PyTorch builds, install the appropriate PyTorch wheel in the environment first, then run the installer with `--skip-deps` after installing the remaining requirements.

## Repository layout

```text
simple_baselines/
├── code/          embedding, scoring, training, collection, and analysis scripts
├── data/          datasets, cached features, embeddings, and prepared notebook inputs
├── models/        local ESM2/ProGen implementations and model checkpoints
├── notebooks/     seven lean analysis notebooks
├── docs/          per-component usage reference
├── scripts/       installation verification
├── install.sh
├── pyproject.toml
└── requirements.txt
```

`data/` and checkpoint files are intentionally ignored by Git because the full payload is extremely large. They are present in the prepared workspace, but a code-only clone must obtain the data/model artifact separately and place it at `data/` and `models/weights/`. The installer reports missing data without failing; `--strict-data` turns that warning into an error.

## Working model

Run commands and start Jupyter from the repository root. The pipeline is:

```text
data/<dataset>/<dataset>.csv
        |
        v
embedding_engine.py  -->  results/<dataset>/<model>/evaluations (chunked output)
        |
        v
collect_* / merge_embedding_chunks.py  -->  data/<dataset>/embeddings
        |
        v
training, model-selection, scoring, and feature-analysis scripts
        |
        v
data/merged_datasets and other prepared tables  -->  notebooks/lean_*.ipynb
```

Some feature-combination and scoring workflows deliberately produce intermediate manifests or chunked outputs. They require a collection/processing step before becoming notebook-ready data; this mirrors the source workflow.

## Quick starts

Inspect any command before launching a long CPU/GPU job:

```bash
python code/embedding_engine.py --help
python code/exetnsive_model_selection.py --help
python code/train_classifiers_over_ohe.py --help
```

Create a small Hamming-distance matrix:

```bash
python code/create_hamming_mds.py \
  --dataset_name his2 \
  --first_col N137 \
  --last_col F163 \
  --N 100 \
  --output_path data/his2/mds/hamming_100.npy
```

### CASP SaProt embeddings with LSF or local workers

The following example embeds the 25 AAV positions in the CASP dataset with
SaProt. Run it from the repository root. It creates 500-sequence chunks and
immediately submits five GPU workers through LSF:

```bash
conda run -n esm_env python code/embedding_engine.py \
  --action embed_parallel \
  --dataset_file data/casp/casp.csv \
  --sequence_colname full_seq \
  --label_column_name activity \
  --model saprot \
  --saprot_pdb_sequence_file data/casp/casp_pdb_sequence.txt \
  --saprot_token_sequence_file data/casp/casp_foldseek_3di.txt \
  --positions_to_embed 561 562 563 565 566 567 568 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 \
  --evaluation_path results/casp/saprot_aav_sites \
  --chunk_size 500 \
  --internal_batch_size 2 \
  --execute_after_parallel_generation \
  --job_executing_type lsf \
  --n_workers 5
```

The positions are one-based protein positions; the embedding tensor therefore
has 25 positions per sequence. Save the timestamped path printed after
`[INFO] JOBS PATH:` for the monitoring commands below. For example:

```bash
JOBS_PATH=results/casp/saprot_aav_sites/jobs_YYYYMMDD_HHMMSS_parallel
```

To execute locally instead of submitting to LSF, use
`--job_executing_type local`. With `--execute_after_parallel_generation`, the
command starts `--n_workers` local subprocesses and waits for all of them to
finish. Each worker loads its own model, so use one worker unless the machine
has enough CPU/GPU memory for several copies. An already-generated queue can
also be started locally with:

```bash
conda run -n esm_env python code/embedding_engine.py \
  --action execute_workers \
  --jobs_path "$JOBS_PATH" \
  --job_executing_type local \
  --n_workers 1
```

Local stdout, stderr, process IDs, commands, and final return codes are stored
under `$JOBS_PATH/job_execution_*/` in the same metadata layout used for LSF.

Check LSF itself to see whether workers are pending, running, or finished:

```bash
bjobs
bjobs -a
bjobs -l JOB_ID
```

Check chunk-level progress independently of LSF:

```bash
conda run -n esm_env python code/embedding_engine.py \
  --action check_on_jobs \
  --jobs_path "$JOBS_PATH"
```

A chunk's `.job` marker is deleted only after its output is saved, so it is
safe to collect when this reports `Jobs left: 0` and does not warn about
missing `train/indices.pt` files. Worker stdout/stderr paths are recorded in
`$JOBS_PATH/job_execution_*/workers_metadata.csv`; inspect the latest output
with:

```bash
tail -n 50 "$JOBS_PATH"/job_execution_*/job_outputs/out_file_worker_*
tail -n 50 "$JOBS_PATH"/job_execution_*/job_outputs/err_file_worker_*
```

Once all chunks are complete, collect them into tensors split by mutation
count. CASP contains variants with 0 through 18 mutations:

```bash
conda run -n esm_env python code/collect_protgym_embeddings.py \
  --input_csv data/casp/casp.csv \
  --dataset_name casp \
  --model_name saprot_aav_sites \
  --hidden_dim_size 1280 \
  --mutations_to_select 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
  --evaluation_dir results/casp/saprot_aav_sites/evaluations \
  --output_dir data/casp/embeddings/saprot_aav_sites
```

This writes `embeddings_of_nmut_*.pt`, `y_values_of_nmut_*.pt`, and
`indices_of_nmut_*.pt`. To also create one index-ordered tensor for the whole
dataset, merge and validate those files against the source CSV:

```bash
conda run -n esm_env python code/merge_embedding_chunks.py \
  --dataset_csv data/casp/casp.csv \
  --y_col activity \
  --embedding_dirs data/casp/embeddings/saprot_aav_sites \
  --max_nmut 18
```

`--embedding_dirs` must point directly to a leaf directory containing the
`embeddings_of_nmut_*.pt` files; the command does not search subdirectories.
For example, merge the separate `saprot/` chunks for mutation counts 0 through
10 with:

```bash
python code/merge_embedding_chunks.py \
  --dataset_csv data/casp/casp.csv \
  --y_col activity \
  --embedding_dirs data/casp/embeddings/saprot \
  --max_nmut 10
```

Do not pass the parent directory `data/casp/embeddings/`. Also avoid running
the `saprot_aav_sites/` merge with a lower `--max_nmut` after completing the
18-mutation merge, because the partial tensors would replace the complete
merged outputs in that directory.

The merged files are `embeddings.pt`, `y_values.pt`, and `indices.pt` in
`data/casp/embeddings/saprot_aav_sites/`. For the current CASP data, the
final embedding shape is expected to be `(18465, 25, 1280)`.

Run the matched SaProt-versus-one-hot regression benchmark with mean-pooled
embeddings:

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

`--mean_embeddings` reduces each 25-by-1280 tensor to 1280 features. Without
it, the script flattens each tensor to 32,000 features. The matched run writes
the one-hot metrics under `data/casp/subsamples/one_hot/` and the SaProt metrics
under `data/casp/subsamples/saprot/`. Architecture settings are encoded in the
CSV filenames. Override them with `--ohe_hl`, `--llm_hl`, `--ohe_solver`,
`--llm_solver`, `--ohe_alpha`, `--llm_alpha`, `--ohe_max_iter`, and
`--llm_max_iter`.

Run controlled extrapolation with the default MLP regressor, training a
separate cumulative model through mutation orders 1, 2, 3, 4, and 5 and
testing each model on every available higher mutation order:

```bash
conda run -n esm_env python code/train_classifiers_over_embeddings.py \
  --model_name saprot_aav_sites \
  --dataset_name casp \
  --mean_embeddings \
  --regression \
  --external_labels_column activity \
  --n_start 1 \
  --n_end 6 \
  --niters 200 \
  --output_dir data/casp/controlled_extrapolation/saprot
```

`--n_end` is exclusive. CASP mutation-order files are discovered automatically,
so evaluation continues through order 18. Results are written under
`data/casp/controlled_extrapolation/saprot/`. The script's general default is
`data/<dataset>/controlled_extrapolation/<model>/`; the explicit `--output_dir`
above gives this SaProt run its shorter output name.

Run the matching controlled-extrapolation experiment with one-hot features:

```bash
conda run -n esm_env python code/train_classifiers_over_ohe.py \
  --dataset_name casp \
  --regression \
  --external_labels_column activity \
  --n_start 1 \
  --n_end 6 \
  --niters 200
```

This writes architecture-specific CSVs under
`data/casp/controlled_extrapolation/one_hot/` and evaluates every available
mutation order above the training threshold, through CASP order 18. Override
the MLP architecture with `--hl 100,50`, and tune it with `--alpha`, `--op`,
`--lr`, and `--niters`. Use `--output_dir` only when a different destination is
needed.

Generate NMT model-selection jobs using repository-relative paths:

```bash
python code/generate_jobs_for_extensive_model_selection.py \
  --output_dir data/nmt/scoring_jobs \
  --sequence_path data/nmt/nmt.csv \
  --activity_column_name activity \
  --num_muts_colname num_muts \
  --regressor \
  --min_nmuts 5 --max_nmuts 11 \
  --min_train_muts 5 --max_train_muts 9 --max_test_muts 11 \
  --first_col Y20 --last_col F253 \
  --embeddings esm_35m esm_8m \
  --train_sizes 25 50 75 100 200 \
  --niters 10 --load_all
```

The generator only writes job scripts; it does not run them. When the command
uses repository-relative paths, run every generated script from the repository
root. Do not change into `preflight/`, `resolve_architectures/`, or
`score_after_resolving/`, because paths such as `data/nmt/nmt.csv` are resolved
from the current working directory. Run the three phases in order:

```bash
cd /path/to/simple_baselines
JOB_DIR=data/nmt/scoring_jobs

# 1. Create the shared train/validation/test splits.
bash "$JOB_DIR/preflight/00_make_splits.sh"

# 2. Resolve every required architecture. These may instead be submitted to a scheduler.
for job in "$JOB_DIR"/resolve_architectures/*.sh; do
  bash "$job"
done

# 3. Score only after all required architecture jobs have completed.
for job in "$JOB_DIR"/score_after_resolving/*.sh; do
  bash "$job"
done
```

For example, a CASP/SaProt preflight generated under
`data/casp/scoring_jobs/saprot` should be launched from the repository root as
`bash data/casp/scoring_jobs/saprot/preflight/00_make_splits.sh`. The message
`--train_sizes not provided; random train_type scripts were skipped` is
informational: mutation-based jobs were generated, but random-subsampling jobs
were not. Add `--train_sizes ...` to the generator command only when those
random jobs are wanted. Absolute `--sequence_path` and `--cache_path` values
can be used when generated scripts need to run from arbitrary directories.

Each scoring script requires the matching architecture-resolution script to
finish successfully first. Preflight creates only the data splits; it does not
create `best_architecture.json`. For example, CASP/SaProt full-resolution
scoring for training order 1 has this dependency:

```bash
# Creates scoring_cache/regressor/architectures/mean_saprot/full/best_architecture.json
bash data/casp/scoring_jobs/saprot/resolve_architectures/0001_mean_saprot_full.sh

# Run only after the architecture command succeeds.
bash data/casp/scoring_jobs/saprot/score_after_resolving/0001_mean_saprot_mutation_full_train_K_1.sh
```

There is an additional limitation for the currently bundled CASP/SaProt data:
`data/casp/casp.csv` contains mutation orders through 18, while
`data/casp/embeddings/saprot` contains embedding chunks only for orders 0–10.
The scorer's `full` architecture split uses every CSV row, so the full
architecture command above will try to load missing chunks beginning with
`embeddings_of_nmut_11.pt`. Full mode therefore requires either generating the
SaProt chunks for orders 11–18 or changing full split creation to respect the
requested maximum mutation order. The bounded `less_than_K` and `equal_to_K`
workflows can use the existing chunks. For example:

```bash
bash data/casp/scoring_jobs/saprot/resolve_architectures/0002_mean_saprot_less_than_K_K_1.sh
bash data/casp/scoring_jobs/saprot/score_after_resolving/0006_mean_saprot_mutation_less_than_K_train_K_1.sh
```

Do not remove `--require_resolved_architecture` from a scoring script as a
workaround: that only moves architecture resolution into the scoring process
and does not fix missing embedding chunks.

Launch the notebooks:

```bash
jupyter lab
```

The notebook bootstrap cells accept either the repository root or `notebooks/` as the kernel working directory, switch to the repository root, and add `code/` to the import path.

## Runtime dependencies outside pip

- `mine_foldseek_tokens.py` requires the Foldseek executable.
- LSF job submission modes require `bsub`; generated shell jobs can instead be inspected/run manually.
- `generate_scoring_for_feature_comp.py` and `mixed_representation_analysis_gen_jobs.py` integrate with a separate FuncScape checkout for scoring execution.
- Hugging Face models such as SaProt, ProtBERT, Ankh, and IgBert require network/cache access. Bundled ESM2 and ProGen checkpoints work locally.
- Large embedding/training jobs commonly require a CUDA GPU and substantial RAM/VRAM.

The historical command archive is retained at `data/tmp_scripts.txt`. It contains absolute paths from the source workspace; use the repository-relative examples above and in [the code reference](docs/CODE_REFERENCE.md) for new runs.

## Documentation

See [docs/CODE_REFERENCE.md](docs/CODE_REFERENCE.md) for every module, its role, execution style, important inputs/outputs, and notebook descriptions. See [docs/EXAMPLE_RUNS.md](docs/EXAMPLE_RUNS.md) for concrete commands or import examples covering every module under `code/`.
