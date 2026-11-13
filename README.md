# Protein Fitness: Simple Baselines

This repository contains the baseline, embedding, and analysis workflows extracted from `fitness_learning`: seven lean analysis notebooks, model and training code, local PLM implementations, and scripts for downloading the prepared inputs.

The fastest way to reproduce the current analyses is to use the prepared data and run the notebooks. Recomputing embeddings and model outputs is a separate, substantially longer workflow described below.

## Reproduce the current analyses

### 1. Install

Python 3.10 or 3.11 is required.

```bash
git clone <repository-url> simple_baselines
cd simple_baselines
./install.sh
source .venv/bin/activate
```

The installer creates `.venv`, installs [requirements.txt](requirements.txt), installs the modules under `code/`, and runs a smoke test. To use an existing Conda environment instead:

```bash
./install.sh --no-venv --skip-deps --python "$CONDA_PREFIX/bin/python"
```

### 2. Download the prepared analysis data

From the repository root, run:

```bash
bash scripts/download_data.sh
```

This downloads two public archives: one containing the datasets and one containing the merged analysis tables plus prepared embeddings. Datasets are installed under `data/`, merged tables under the root-level `merged_datasets/`, and embeddings under each dataset's `data/<dataset>/embeddings/` directory. A `data/merged_datasets` compatibility link is also created when that path is available. Use `--download-dir .downloads` to retain the large archives for reuse.

Verify the installation and data payload with:

```bash
python scripts/verify_install.py --strict-data
```

Prepared embeddings are included for rerunning embedding-based baselines; they are not needed to recreate the notebook analyses.

### 3. Run the lean notebooks

Start Jupyter from the repository root:

```bash
jupyter lab
```

Open a notebook under `notebooks/` and use **Restart Kernel and Run All Cells**. The notebooks normalize their working directory to the repository root and can also be launched from `notebooks/`.

| Notebook | What it does |
|---|---|
| [`lean_epistasis_analysis.ipynb`](notebooks/lean_epistasis_analysis.ipynb) | Quantifies classical and sign epistasis, reversals, and their relationship to zero-shot scores. |
| [`lean_library_generation.ipynb`](notebooks/lean_library_generation.ipynb) | Compares candidate-library screening and exploration strategies across protein landscapes. |
| [`lean_merged_performance_ratio_plots.ipynb`](notebooks/lean_merged_performance_ratio_plots.ipynb) | Plots controlled-extrapolation and subsampling performance, including PLM-to-one-hot ratios. |
| [`lean_oracle_transfer_learning.ipynb`](notebooks/lean_oracle_transfer_learning.ipynb) | Compares oracle transfer learning with embeddings, delta embeddings, and one-hot encodings. |
| [`lean_similarity_analysis.ipynb`](notebooks/lean_similarity_analysis.ipynb) | Compares one-hot and PLM representation geometry using pairwise distances and reconstruction. |
| [`lean_variance_analysis_merged.ipynb`](notebooks/lean_variance_analysis_merged.ipynb) | Runs train/test-aware statistical tests and summarizes variance explained. |
| [`lean_zero_shot_notebook.ipynb`](notebooks/lean_zero_shot_notebook.ipynb) | Evaluates zero-shot score correlations, ROC performance, and model comparisons. |

The notebooks are independent; they do not need to be run in a particular order. Generated figures are written under `figures/` where an output path is configured.

## Rerun analyses from earlier stages

The full workflow is:

```text
dataset CSV
  -> chunked PLM embeddings
  -> collected mutation-order tensors
  -> baseline training, model selection, or fitness scoring
  -> merged result tables
  -> lean notebooks
```

These steps can require substantial CPU, GPU, memory, and scheduler time.

### Generate embeddings

This example generates PARD3 embeddings at its ten designed positions with ESM2 8M and one local worker. Before running it, place the checkpoint at `models/weights/esm2/esm2_t6_8M_UR50D.pth`; model weights are not tracked by Git.

```bash
python code/embedding_engine.py \
  --action embed_parallel \
  --dataset_file data/pard3/pard3.csv \
  --sequence_colname full_seq \
  --label_column_name activity \
  --model esm2_t6_8M_UR50D \
  --embedding_identifier esm_8m \
  --positions_to_embed 48 52 53 55 56 74 78 80 81 82 \
  --evaluation_path results/pard3/esm_8m \
  --chunk_size 500 \
  --internal_batch_size 20 \
  --execute_after_parallel_generation \
  --job_executing_type local \
  --n_workers 1
```

The positions are one-based protein positions. For an LSF cluster, change `--job_executing_type local` to `lsf` and select a suitable worker count. Each worker loads its own model copy.

The command produces chunked outputs. Follow the collection and merging examples in [EXAMPLE_RUNS.md](docs/EXAMPLE_RUNS.md#embedding-collection-and-feature-preparation) to create tensors suitable for training. Model-specific roles and inputs are summarized in [CODE_REFERENCE.md](docs/CODE_REFERENCE.md#core-embedding-and-model-modules).

The prepared embedding archive is installed by `scripts/download_data.sh`, so these tensors do not need to be recomputed unless you want to regenerate them.

### Rerun baseline and fitness experiments

List the reproducible experiment configurations:

```bash
bash scripts/run_baseline_experiments.sh list
```

Run one controlled-extrapolation baseline:

```bash
bash scripts/run_baseline_experiments.sh \
  pard3-controlled-regression-embeddings
```

Run a focused zero-shot fitness calculation:

```bash
FITNESS_DATASETS="pard3" FITNESS_MODELS="esm_8m" \
  bash scripts/run_baseline_experiments.sh fitness-pll
```

The runner uses the `esm_env` Conda environment by default; set `BASELINE_CONDA_ENV` to override it. Experiment names, output paths, quick-test settings, and classification caveats are documented in [BASELINE_EXPERIMENTS.md](docs/BASELINE_EXPERIMENTS.md).

## Documentation

| Document | Use it for |
|---|---|
| [BASELINE_EXPERIMENTS.md](docs/BASELINE_EXPERIMENTS.md) | Prepared baseline and zero-shot experiment runner, outputs, and resource notes. |
| [EXAMPLE_RUNS.md](docs/EXAMPLE_RUNS.md) | Concrete command or import examples for every module under `code/`. |
| [CODE_REFERENCE.md](docs/CODE_REFERENCE.md) | Concise descriptions of each component and the embedding-to-analysis workflow. |

## Repository layout

```text
code/          embedding, scoring, training, collection, and analysis modules
data/          downloaded datasets, cached features, and prepared embeddings
merged_datasets/  prepared tables consumed by the lean notebooks
models/        local ESM2/ProGen implementations and optional checkpoints
notebooks/     seven lean analysis notebooks
docs/          detailed references and examples
scripts/       data download, experiment runner, and installation verification
```

## Runtime notes

- Run commands from the repository root unless a document says otherwise.
- Large embedding and training runs commonly require a CUDA GPU and substantial RAM/VRAM.
- LSF execution requires `bsub`; local worker execution is available for embedding jobs.
- ESM2 and ProGen use local checkpoints under `models/weights/`; checkpoint files are not tracked by Git.
- SaProt, ProtBERT, Ankh, and other Hugging Face models require network access or an existing model cache.
- `mine_foldseek_tokens.py` requires the Foldseek executable.
- Downloaded data, model outputs, and generated figures are intentionally excluded from Git.
