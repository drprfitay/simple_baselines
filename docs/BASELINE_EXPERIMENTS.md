# HIS2, PARD3, and CASP baseline experiments

The companion command runner is
[`scripts/run_baseline_experiments.sh`](../scripts/run_baseline_experiments.sh).
It contains controlled-extrapolation and random-subsampling experiments for
regression and classification, using both one-hot features and the embeddings
already present under `data/`.

Run commands from any directory; the runner changes to the repository root
before starting an experiment. It uses the `esm_env` conda environment by
default.

## Prepared embeddings

| Dataset | Embedding directories | Controlled training thresholds |
|---|---|---|
| HIS2 | `prot_bert`, `progen2-small` | 1 through 9 |
| PARD3 | `esm_35m`, `esm_150m`, `esm_650m` | 1 through 9 |
| CASP | `saprot`, `saprot_aav_sites` | 1 through 10 |

The controlled commands use mean-pooled embeddings. HIS2 and PARD3 have
prepared mutation-order chunks through order 10; CASP has chunks through order
11. The scripts treat `n_end` as exclusive, leaving the final order available
for held-out testing.

## Usage

List all experiment names:

```bash
bash scripts/run_baseline_experiments.sh list
```

Run one experiment:

```bash
bash scripts/run_baseline_experiments.sh \
  pard3-controlled-regression-embeddings
```

Run a one-hot classifier:

```bash
bash scripts/run_baseline_experiments.sh \
  casp-controlled-classification-one-hot
```

Run a subsampling experiment:

```bash
bash scripts/run_baseline_experiments.sh \
  his2-subsamples-regression
```

Run every experiment sequentially only when enough time and memory are
available:

```bash
bash scripts/run_baseline_experiments.sh all
```

Run one of the four zero-shot fitness calculations:

```bash
bash scripts/run_baseline_experiments.sh fitness-classic
bash scripts/run_baseline_experiments.sh fitness-l2r
bash scripts/run_baseline_experiments.sh fitness-single-mutations
bash scripts/run_baseline_experiments.sh fitness-pll
```

`fitness-classic` runs the original additive masked-position implementation.
The other names select `left_to_right`, `one_mut`, and `mutation_mask`,
respectively, in `calculate_fitness_for_plm_multiple_decodings.py`. NMT is
excluded from the default dataset lists because it is not in the public data
download. Each decoding mode writes a separate result file, and the runner
resumes any canonical columns already present in that file.

Fitness runs use each calculator's default datasets and models. Restrict a run
with space-separated environment variables; this is also useful for a focused
smoke test:

```bash
FITNESS_DATASETS="pard3" FITNESS_MODELS="esm_8m" \
  bash scripts/run_baseline_experiments.sh fitness-pll
```

To use another conda environment:

```bash
BASELINE_CONDA_ENV=my_environment \
  bash scripts/run_baseline_experiments.sh pard3-subsamples-classification
```

## Experiment-name pattern

Controlled-extrapolation names use:

```text
<dataset>-controlled-<regression|classification>-<embeddings|one-hot>
```

Subsampling names use:

```text
<dataset>-subsamples-<regression|classification>
```

Each subsampling command evaluates the one-hot baseline and every embedding
listed for that dataset in a single run. It uses training sample sizes 25, 50,
100, 200, 500, and 1,000, with 30 random repetitions per size.

## Output locations

Controlled embedding outputs are written to:

```text
data/<dataset>/controlled_extrapolation/<task>/embeddings/
```

Controlled one-hot outputs are written to:

```text
data/<dataset>/controlled_extrapolation/<task>/one_hot/
```

New one-hot controlled files use names such as:

```text
one_hot_evaluation_train_on_2_mlp.csv
```

Subsampling outputs are separated by representation under:

```text
data/<dataset>/subsamples/collected_results/
```

The CASP subsampling commands explicitly map the two embedding inputs to the
distinct output directories `saprot/` and `saprot_aav_sites/`.

Fitness outputs are written beneath:

```text
results/<dataset>/fitness_results/
```

The multiple-decoding outputs are:

| Scoring mode | File | Columns |
|---|---|---|
| `one_mut` | `new_om_normed_fitness_all.csv` | `esm_8m`, `esm_35m`, `esm_150m`, `esm_650m`, `esm_3b`, `progen2-medium`, `progen2-small`, `saprot` |
| `mutation_mask` | `pll_normed_fitness_all.csv` | Same canonical model columns as `one_mut` |
| `left_to_right` | `new_l2r_normed_fitness_all.csv` | `progen2-medium_left_to_right`, `progen2-small_left_to_right` |

With resume enabled, legacy suffixed columns such as `saprot_single_mut` and
`saprot_pll` are normalized to `saprot`; columns outside the canonical schema
are removed when the file is rewritten.

## Quick tests

The runner contains the full experiment settings. For a quick manual smoke
test of subsampling, copy the corresponding command from the runner and replace:

```text
--n_samples 25 50 100 200 500 1000 --iters 30
```

with:

```text
--n_samples 100 --iters 1
```

For a quick controlled-extrapolation test, replace the mutation range with:

```text
--n_start 1 --n_end 2
```

This trains only the threshold-1 model while retaining all higher mutation
orders as tests.

## Classification-label caveat

The current controlled-extrapolation scripts convert continuous activity to
binary labels at the activity median and then invert the labels. The
subsampling script instead uses the activity mean by default and treats higher
activity as class 1. Consequently, classification metrics from the two
protocols do not currently use identical class definitions.

Use `--classification_threshold` to select another subsampling threshold. The
subsampling CLI does not currently provide a label-inversion option.

## Resource considerations

The merged HIS2 embeddings are approximately 3.5 GB per model, and the merged
CASP embeddings are approximately 2.3 GB per model. Mean pooling reduces the
features used for training, but loading each source tensor still requires
several gigabytes of memory. Avoid running multiple large experiments on the
same low-memory node.
