# Protein fitness simple baselines

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
