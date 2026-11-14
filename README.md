# Simple Baselines

```text
  ___ ___ __  __ ___ _    ___
 / __|_ _|  \/  | _ \ |  | __|
 \__ \| || |\/| |  _/ |__| _|
 |___/___|_|  |_|_| |____|___|

  ___   _   ___ ___ _    ___ _  _ ___ ___
 | _ ) /_\ / __| __| |  |_ _| \| | __/ __|
 | _ \/ _ \\__ \ _|| |__ | || .` | _|\__ \
 |___/_/ \_\___/___|____|___|_|\_|___|___/
```

Code and data accompanying **[Simple baselines rival protein language models in mutation-dense design tasks](https://www.biorxiv.org/content/10.64898/2026.05.01.722313v1)**. The repository provides seven analysis notebooks, baseline and training code, local pLM implementations, and prepared-data downloads. The broader research codebase is `fitness_learning`. **The notebooks reproduce the paper's figures directly; the full computational reproduction workflow is documented below.**

## Installation and data download

### 1. Install

Python **3.10 or 3.11** is required.

```bash
git clone <repository-url> simple_baselines
cd simple_baselines
./install.sh
source .venv/bin/activate
```

The installer creates `.venv`, installs [requirements.txt](requirements.txt), installs the modules under `code/`, and runs a smoke test.

To use an existing Conda environment instead:

```bash
./install.sh --no-venv --skip-deps --python "$CONDA_PREFIX/bin/python"
```

### 2. Download the prepared data

From the repository root:

```bash
bash scripts/download_data.sh
```

The script downloads two public archives: (1) the datasets and (2) the merged analysis tables together with prepared embeddings. It installs:

- Datasets under `data/`.
- **Merged analysis tables under root-level `merged_datasets/`.** These are the only data files required to reproduce the paper's figures with the notebooks.
- Prepared example embeddings for **HIS2, AAV, and PARD3** under each dataset's `data/<dataset>/embeddings/` directory. They are useful for rerunning embedding-based baselines, but **not required for figure reproduction**. Embeddings for all datasets can be regenerated using the workflow below.
- A `data/merged_datasets` compatibility link, when that path is available.

To retain the large downloaded archives for reuse, pass `--download-dir .downloads` to the downloader.

Verify the installation and downloaded payload with:

```bash
python scripts/verify_install.py --strict-data
```

## Reproduce all paper figures: analysis notebooks

**The seven notebooks collectively reproduce all figures in the paper from the downloaded `merged_datasets/` tables.** They summarize the results of **all experiments reported in the paper**, including the analyses of previously computed predictions, trained models, and embeddings. Running these notebooks does **not** require regenerating embeddings, retraining models, or recomputing zero-shot scores.

Start Jupyter from the repository root:

```bash
jupyter lab
```

Open a notebook in `notebooks/` and choose **Restart Kernel and Run All Cells**. The notebooks are independent and can be run in any order; their order below follows the progression of the paper's results. They normalize their working directory to the repository root and can also be launched from `notebooks/`.

| Notebook | Analysis reproduced |
|---|---|
| [`lean_zero_shot_notebook.ipynb`](notebooks/lean_zero_shot_notebook.ipynb) | Evaluates zero-shot mutation-effect/fitness scoring against measured activity: rank correlations, ROC performance for distinguishing functional variants, and comparisons between pLM and conventional scores. |
| [`lean_epistasis_analysis.ipynb`](notebooks/lean_epistasis_analysis.ipynb) | Characterizes classical and sign epistasis and fitness-effect reversals; compares these landscape properties with changes in zero-shot model scores. |
| [`lean_merged_performance_ratio_plots.ipynb`](notebooks/lean_merged_performance_ratio_plots.ipynb) | Summarizes supervised prediction across controlled lower-to-higher mutation-order extrapolation and random subsampling; compares pLM-based models with one-hot baselines, including PLM-to-one-hot performance ratios. |
| [`lean_variance_analysis_merged.ipynb`](notebooks/lean_variance_analysis_merged.ipynb) | Quantifies explained variance and performs train/test-aware statistical comparisons of model performance across the merged benchmark results. |
| [`lean_oracle_transfer_learning.ipynb`](notebooks/lean_oracle_transfer_learning.ipynb) | Analyzes oracle transfer-learning experiments and contrasts full pLM embeddings, delta embeddings, and one-hot sequence encodings. |
| [`lean_similarity_analysis.ipynb`](notebooks/lean_similarity_analysis.ipynb) | Examines what pLM representations add beyond mutation identity by comparing pairwise representation distances and reconstructing pLM embeddings from one-hot encodings. |
| [`lean_library_generation.ipynb`](notebooks/lean_library_generation.ipynb) | Evaluates computational candidate-library selection and exploration strategies across protein fitness landscapes, comparing their ability to prioritize useful sequences. |

Generated figures are saved under `figures/` wherever an output path is configured.

## Reproduce the underlying experiments

The notebooks above reproduce the paper's figures directly from the prepared merged tables. **To independently recompute the results that feed those tables**, follow the full pipeline below, then rerun the same notebooks on the regenerated results:

```text
dataset CSV
  -> chunked PLM embeddings
  -> collected mutation-order tensors
  -> baseline training, model selection, or fitness scoring
  -> merged result tables
  -> lean notebooks
```

These stages can require substantial CPU, GPU, RAM/VRAM, and scheduler time. Prepared embeddings (including the HIS2, AAV, and PARD3 examples) are already included in the downloaded data, so you can reuse them where applicable instead of starting from the embedding stage. All embeddings can also be recomputed from scratch.

### 1. Generate embeddings

For example, generate PARD3 embeddings at its ten designed positions with ESM2 8M and one local worker. First place the model checkpoint at `models/weights/esm2/esm2_t6_8M_UR50D.pth` (model weights are not tracked by Git):

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

The positions are **one-based protein positions**. For an LSF cluster, switch `--job_executing_type local` to `lsf` and choose a suitable worker count; **each worker loads its own copy of the model**.

This produces chunked outputs. Follow the collection and merging examples in [EXAMPLE_RUNS.md](docs/EXAMPLE_RUNS.md#embedding-collection-and-feature-preparation) to build the mutation-order tensors used for training. See [CODE_REFERENCE.md](docs/CODE_REFERENCE.md#core-embedding-and-model-modules) for model-specific roles and required inputs.

### 2. Rerun baseline training and fitness experiments

List the available reproducible experiment configurations:

```bash
bash scripts/run_baseline_experiments.sh list
```

Run a controlled-extrapolation baseline:

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

### 3. Collect results and rerun the analysis

Collect the generated outputs into mutation-order tensors and merged result tables using the workflows documented in [EXAMPLE_RUNS.md](docs/EXAMPLE_RUNS.md) and [CODE_REFERENCE.md](docs/CODE_REFERENCE.md). Once the corresponding merged tables have been reconstructed, rerun the [analysis notebooks](#reproduce-all-paper-figures-analysis-notebooks) to regenerate the figures from your own experiment outputs.

## Documentation

| Document | Use it for |
|---|---|
| [BASELINE_EXPERIMENTS.md](docs/BASELINE_EXPERIMENTS.md) | Prepared baseline and zero-shot experiment runner, outputs, and resource notes. |
| [EXAMPLE_RUNS.md](docs/EXAMPLE_RUNS.md) | Concrete command or import examples for every module under `code/`. |
| [CODE_REFERENCE.md](docs/CODE_REFERENCE.md) | Concise descriptions of each component and the embedding-to-analysis workflow. |

## Repository layout

```text
code/             embedding, scoring, training, collection, and analysis modules
data/             downloaded datasets, cached features, and prepared embeddings
merged_datasets/  prepared tables consumed by the lean notebooks
models/           local ESM2/ProGen implementations and optional checkpoints
notebooks/        seven lean analysis notebooks
docs/             detailed references and examples
scripts/          data download, experiment runner, and installation verification
```

## Runtime notes

- Run commands from the repository root unless a document says otherwise.
- Large embedding and training runs commonly require a CUDA GPU and substantial RAM/VRAM.
- LSF execution requires `bsub`; local worker execution is available for embedding jobs.
- ESM2 and ProGen use local checkpoints under `models/weights/`; checkpoint files are not tracked by Git.
- SaProt, ProtBERT, Ankh, and other Hugging Face models require network access or an existing model cache.
- `mine_foldseek_tokens.py` requires the Foldseek executable.
- Downloaded data, model outputs, and generated figures are intentionally excluded from Git.

## Citation

If you use this repository, please cite:

> Talpir, I., & Fleishman, S. J. (2026). *Simple baselines rival protein language models in mutation-dense design tasks*. bioRxiv. https://doi.org/10.64898/2026.05.01.722313
