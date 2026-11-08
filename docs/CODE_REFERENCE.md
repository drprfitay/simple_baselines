# Code and notebook reference

All commands below assume the repository root as the current directory and an activated environment. Scripts with a CLI expose their complete options through:

```bash
python code/<script>.py --help
```

## Core embedding and model modules

| Component | Purpose | How it is used |
|---|---|---|
| `plm_base.py` | Unified wrappers for bundled ESM2 and ProGen models plus Hugging Face IgBert, ProtBERT, Ankh, and SaProt models; also applies LoRA weights. | Imported by embedding/scoring/training scripts. Call `plm_init(<repo-root>)` before `load_model(...)`. |
| `dataset.py` | PyTorch dataset classes and sequence/token preparation used by embedding and training. | Library module; imported by `embedding_engine.py`. |
| `embedder.py` | Chunk embedding execution, result persistence, t-SNE helpers, and local/LSF worker orchestration. | Library module; imported by `embedding_engine.py`. |
| `embedding_engine.py` | Main CLI for tokenizing/embedding datasets, including chunked parallel execution. | `python code/embedding_engine.py --help`; outputs chunk folders under the requested evaluation path. |
| `trainer.py` | Full, frozen-trunk, or LoRA masked-language-model training over an MSA. | CLI; requires `--plm_name`, `--msa_path`, and `--output_path`. |
| `utils.py` | Dataset registry, labels, metrics, plotting settings, aggregation, and general helpers. Paths resolve from this repository. | Shared library module. |
| `utils_for_analysis.py` | Compatibility re-export of `utils.py` for original analysis scripts/notebooks. | Shared library module. |
| `example_run_jobs.py` | Local and LSF command-launch helpers used by embedding orchestration. | Library module; not a standalone workflow. |

## Embedding collection and feature preparation

| Component | Purpose | Main inputs and outputs |
|---|---|---|
| `collect_gfp_embeddings.py` | Reassembles GFP chunk outputs, optionally selects positions, validates labels/indices, and writes mutation-order tensors. | `--input_csv`, `--model_name`, `--hidden_dim_size`; reads `results/gfp/.../evaluations`, writes `data/gfp/new_embeddings/...`. |
| `collect_protgym_embeddings.py` | Generic ProtGym chunk collector split by mutation order. | `--input_csv`, `--dataset_name`, `--model_name`, `--hidden_dim_size`; writes `data/<dataset>/embeddings/<model>`. |
| `merge_embedding_chunks.py` | Merges `embeddings/y_values/indices_of_nmut_*.pt` files in index order and validates against a dataset CSV. | CLI; use `--dataset` or explicit dataset/folder arguments. |
| `subset_embedding_chunks.py` | Subsets a parent embedding/token cache into child datasets while preserving indices and optional aggregate files. | CLI; requires `--parent_dir` and `--subset_dirs`; supports `--dry_run`. |
| `collect_scoring_features_for_representation_analyses.py` | Builds a single feature tensor from one-hot, full/partial/flat/mean embeddings, activities, and selected positions. | CLI; writes the requested `.pt` feature dictionary. |
| `mine_foldseek_tokens.py` | Runs Foldseek on a PDB/CIF, aligns 3Di tokens to the wild-type sequence, and writes pickle/text inputs used by SaProt. | Positional form: `python code/mine_foldseek_tokens.py [dataset] [csv] [pdb]`; requires Foldseek. |

## Baseline training and model selection

| Component | Purpose | How to run |
|---|---|---|
| `exetnsive_model_selection.py` | Extensive controlled-extrapolation/random-split architecture selection over embeddings or one-hot features. The historical filename typo is preserved for compatibility. | CLI; `--dataset_path` is required. Use `--help` for task/split/architecture controls. |
| `generate_jobs_for_extensive_model_selection.py` | Generates shell jobs and manifests for the extensive model-selection workflow. | CLI; requires output/data/mutation bounds and one task mode (`--regressor` or classifier option). |
| `collect_model_selection_scoring_results.py` | Collects scorer outputs into the five standard aggregate CSV groups. | CLI; requires `--cache_path`; optional `--output_dir`. |
| `train_classifiers_over_embeddings.py` | Trains/evaluates classifiers or regressors over selected embedding models and mutation-order splits. | CLI; requires one or more `--model_name` values. |
| `train_classifiers_over_ohe.py` | Equivalent controlled-extrapolation workflow over one-hot features. | CLI; discovers available mutation orders and writes architecture-specific results under `data/<dataset>/controlled_extrapolation/one_hot/` by default. |
| `train_classifiers_over_subsamples_new.py` | Parameterized GFP-style LLM-vs-OHE subsampling experiment. | CLI; configure dataset, embedding directory/model, sample sizes, iterations, and output CSV. |
| `train_regressors_over_embeddings_subsamples.py` | General LLM-vs-OHE subsampling benchmark for regression or classification. | CLI; writes representation-specific results under `data/<dataset>/subsamples/<representation>/` by default; accepts independent OHE/LLM architectures. |
| `train_classifiers_over_subsamples.py` | Original hard-coded GFP subsampling experiment retained unchanged as a legacy reproduction script. | Direct execution only after confirming its configuration block and expected GFP embeddings. Prefer the `_new.py` script for new runs. |
| `transfer_learning_across_layers.py` | Mutation-order transfer learning over embedding layers, delta embeddings, and pooling configurations for NMT/PARD3. | CLI; requires `--dataset {nmt,pard3}` plus output/model options. |
| `transfer_learning_ohe_grid.py` | Matching one-hot transfer-learning architecture grid for NMT/PARD3. | CLI; requires `--dataset {nmt,pard3}`. |
| `lora_controlled_extrap.py` | LoRA fine-tuning/evaluation across mutation-order train/test splits. | CLI; requires dataset, model, training mutation order, and output path. Dataset defaults resolve to `data/`. |

## Fitness scoring, epistasis, and representation analysis

| Component | Purpose | How to run |
|---|---|---|
| `calculate_fitness_for_plm_multiple_decodings.py` | Computes zero-shot fitness with mutation-mask/PLL, one-mutant, or left-to-right scoring modes. | Configuration-driven script. Review `SCORING_MODES`, model/dataset lists, and batch sizes near the top, then execute directly. |
| `calculate_fitness_for_plm_original_additivity.py` | Computes the original additive masked-position fitness baseline. | Configuration-driven script; review model/dataset constants before direct execution. |
| `detect_epistasis.py` | Binary and classical pairwise epistasis/reversal detection across configured datasets. | Direct execution uses `datasets_to_run` and writes under `data/merged_datasets/linear_baslines`. |
| `representation_geometry.py` | OHE-to-representation reconstruction and geometry analyses with controlled extrapolation. | Configuration-driven direct execution; update the constants (`DATASET_NAME`, `FEATURES_PT`, keys) before a run. |
| `normalization_method_sensitivity.py` | Compares normalization choices in transfer and reconstruction experiments for PARD3/HIS5. | CLI; choose `--mode`, `--dataset_name`, optional caches/output directory. |
| `generate_scoring_for_feature_comp.py` | Creates FuncScape feature-combination scoring jobs/manifests and can collect them. | CLI; point `--funcscape_root` to the external project and provide a feature `.pt`. |
| `mixed_representation_analysis_gen_jobs.py` | Generates mixed-representation scoring jobs for specified feature combinations. | CLI or imported `score_queued`; scoring execution uses FuncScape. |
| `run_feature_comp_analysis.py` | Standalone manifest-driven feature-comparison scorer/collector. | CLI; requires `--manifest` plus either `--job-index` or `--collect`. |

## Visualization and geometry utilities

| Component | Purpose | How to run |
|---|---|---|
| `create_hamming_mds.py` | Creates pairwise Hamming matrices, MDS coordinates, and optional figures from mutation columns or external OHE arrays. | CLI; provide a dataset name/path and optional column/sample/output controls. |
| `tsne_candidate_libraries.py` | Generates t-SNE candidate-library plots and selected-variant tables from embeddings and index sets. | CLI; requires dataset and best/worst/PSSM/PLM index lists. |

## Typical embedding-to-baseline sequence

1. Run `embedding_engine.py` with a dataset CSV and evaluation path.
2. Reassemble chunks with `collect_gfp_embeddings.py`, `collect_protgym_embeddings.py`, or `merge_embedding_chunks.py`.
3. Run `exetnsive_model_selection.py` directly or generate batches with `generate_jobs_for_extensive_model_selection.py`.
4. Collect result tables with `collect_model_selection_scoring_results.py`.
5. Place/merge the final analysis tables under `data/merged_datasets/` for the lean notebooks.

The generator prints or writes the exact scorer commands, making generated jobs the authoritative record of each experiment configuration.

## Lean notebooks

| Notebook | Analysis |
|---|---|
| `lean_epistasis_analysis.ipynb` | Epistasis rates, reversals, classical contributions, and plots. |
| `lean_library_generation.ipynb` | Candidate-library screening and exploration comparisons. |
| `lean_merged_performance_ratio_plots.ipynb` | Aggregated controlled-extrapolation/subsampling performance ratios. |
| `lean_oracle_transfer_learning.ipynb` | Oracle transfer-learning comparisons across OHE, embeddings, and delta embeddings. |
| `lean_similarity_analysis.ipynb` | OHE/embedding geometry, pairwise similarities, and reconstruction quality. |
| `lean_variance_analysis_merged.ipynb` | Statistical tests and variance explained from merged result tables. |
| `lean_zero_shot_notebook.ipynb` | Zero-shot score correlations, ROC analyses, and comparative figures. |

Start Jupyter at the repository root with `jupyter lab`. Each notebook normalizes its working directory to the root and imports helpers from `code/`.

## Outputs and reproducibility notes

- Input datasets and prepared features live under `data/`; generated results should remain in a dataset-specific subdirectory or `results/`.
- Most long workflows are stochastic. Preserve seeds, generated manifests, and command lines with the results.
- ESM2/ProGen can use bundled checkpoints. Hugging Face-backed models need cached files or network access.
- Embedding and scoring batch sizes in the source scripts were tuned for GPU jobs; lower them for smaller GPUs.
- `data/tmp_scripts.txt` is the unedited historical command log and contains old absolute paths. It is useful for recovering experiment flags, but new commands should use paths relative to this repository.
