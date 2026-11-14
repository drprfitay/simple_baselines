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

Code and data accompanying the preprint **[Simple baselines rival protein language models in dense combinatorial protein landscapes](https://doi.org/10.64898/2026.05.01.722313)** by Itay Talpir and Sarel J. Fleishman. The repository provides seven analysis notebooks, baseline and training code, local pLM implementations, and prepared-data downloads; 

The prepared-data workflow reproduces the paper's figures without rerunning model predictions. The available stages for independently recomputing the underlying experiments are documented separately below.

## Installation and data download

### System requirements

- **Analysis notebooks:** Python 3.10 or 3.11 and the packages in [requirements.txt](requirements.txt). The dependency set records testing with Python 3.10; the installation smoke test also passed on RHEL 9.6 with Python 3.11.5. The notebooks can run on CPU and do not require a GPU.
- **Installation and data download:** a Unix-like shell, Git, and Python with virtual-environment support; `scripts/download_data.sh` additionally requires `curl` and `unzip`.
- **Experiment recomputation:** large pLM embedding, scoring, and full training runs commonly require a CUDA-capable GPU and substantial RAM/VRAM. No single CUDA version is prescribed by this repository. Local embedding workers are supported; LSF execution additionally requires `bsub` and a configured Conda environment.
- **Optional models and features:** ESM2 and ProGen use local checkpoints under `models/weights/`; Hugging Face-backed models require network access or a populated model cache. [`mine_foldseek_tokens.py`](code/mine_foldseek_tokens.py) requires a Foldseek executable.

### 1. Install

Python **3.10 or 3.11** is required.

```bash
git clone https://github.com/drprfitay/simple_baselines.git simple_baselines
cd simple_baselines
bash install.sh
source .venv/bin/activate
```

The installer creates `.venv`, installs [requirements.txt](requirements.txt), installs the modules under `code/`, and runs a smoke test.

To use an existing Conda environment instead:

```bash
bash install.sh --no-venv --skip-deps --python "$CONDA_PREFIX/bin/python"
```

For experiments that use ProGen2, optionally download the small and medium checkpoints (about 3.2 GiB) to `models/weights/progen/`:

```bash
bash scripts/download_progen_weights.sh
```

This checkpoint download is separate from installation and the prepared analysis-data download.

### 2. Download the prepared data

From the repository root:

```bash
bash scripts/download_data.sh
```

The script downloads two public archives: (1) the datasets and (2) the merged analysis tables together with prepared embeddings. It installs:

- Datasets under `data/`.
- **Merged analysis tables under root-level `merged_datasets/`.** These provide the primary inputs to the notebooks.
- Prepared example embeddings for **HIS2, AAV, and PARD3** under each dataset's `data/<dataset>/embeddings/` directory. They are useful for rerunning embedding-based baselines, but **not required for figure reproduction**. Embeddings for all datasets can be regenerated using the workflow below.
- A `data/merged_datasets` compatibility link, when that path is available.

To retain the large downloaded archives for reuse, pass `--download-dir .downloads` to the downloader.

Run the dependency, source-code, repository-structure, notebook-count, and dataset-CSV smoke test with:

```bash
python scripts/verify_install.py --strict-data
```

`--strict-data` checks the eleven canonical `data/<dataset>/<dataset>.csv` files; it does not verify every table under `merged_datasets/`. Executing the minimal notebook demonstration below provides an end-to-end check of its prepared analysis-table inputs.

## Reproduce all paper figures: analysis notebooks

**The seven notebooks collectively reproduce all figures in the paper from the prepared download.** They primarily consume the analysis tables under `merged_datasets/` and summarize the results of **all experiments reported in the paper**, including analyses of previously computed predictions, trained models, and embeddings. `lean_zero_shot_notebook.ipynb`, `lean_epistasis_analysis.ipynb`, and `lean_library_generation.ipynb` additionally read downloaded dataset or PSSM files under `data/`. Running the notebooks does **not** require regenerating embeddings, retraining models, or recomputing zero-shot scores.

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


### Minimal working demonstration

After installation and data download, run the CPU-only variance notebook from the repository root in Jupyter, or execute it non-interactively:

```bash
jupyter nbconvert --to notebook --execute \
  notebooks/lean_variance_analysis_merged.ipynb \
  --output lean_variance_analysis_merged.executed.ipynb \
  --output-dir /tmp \
  --ExecutePreprocessor.timeout=600
```

A successful run completes without notebook errors and writes these six SVG files under `figures/variance_analysis/`:

```text
pvalue_dist_fdr_correction.svg
corrected_all.svg
corrected_muts.svg
fval_all.svg
fval_muts.svg
var_explained_plot.svg
```

## Reproduce the underlying experiments

The notebooks above reproduce the paper's figures directly from the prepared download. **To independently recompute the experiments**, follow the available stages below, integrate the resulting cross-experiment analysis tables, and then rerun the same notebooks:

```text
dataset CSV
  -> chunked PLM embeddings
  -> collected mutation-order tensors
  -> baseline training, model selection, or fitness scoring
  -> experiment-specific result tables
  -> cross-experiment analysis-table integration
  -> lean notebooks
```

These stages can require substantial CPU, GPU, RAM/VRAM, and scheduler time. Prepared embeddings (including the HIS2, AAV, and PARD3 examples) are already included in the downloaded data, so you can reuse them where applicable instead of starting from the embedding stage. All embeddings can also be recomputed from scratch. The repository provides embedding collectors and experiment-specific result collectors, but it does not currently provide one end-to-end command that rebuilds every final notebook input (for example, `merged_datasets/merged.csv` and `merged_datasets/merged_by_subsamples.csv`) from all newly computed outputs.

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

### 3. Collect available results and integrate the analysis tables

Collect embedding chunks into mutation-order tensors and collect experiment-specific result tables using the workflows documented in [EXAMPLE_RUNS.md](docs/EXAMPLE_RUNS.md) and [CODE_REFERENCE.md](docs/CODE_REFERENCE.md). Final cross-experiment integration into all schemas and paths expected under `merged_datasets/` must currently be performed separately; no single repository command implements that last stage. Once the corresponding analysis tables have been assembled, rerun the [analysis notebooks](#reproduce-all-paper-figures-analysis-notebooks) to regenerate the figures from your own experiment outputs.

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
- A clean installation time has not yet been benchmarked, so an installation-time estimate and its reference hardware remain to be supplied.

### Availability and license

The source repository is available at [https://github.com/drprfitay/simple_baselines](https://github.com/drprfitay/simple_baselines). No `LICENSE` file is currently included; a repository license remains an outstanding submission requirement.

## Citation

If you use this repository, please cite:

Talpir, I., & Fleishman, S. J. (2026). *Simple baselines rival protein language models in dense combinatorial protein landscapes*. bioRxiv. https://doi.org/10.64898/2026.05.01.722313
