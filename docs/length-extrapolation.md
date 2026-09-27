# Test length extrapolation of the chunk aggregator

This experiment asks whether an aggregator trained on natural proteins shorter
than 1,500 residues still improves reconstruction on held-out proteins of
1,500–2,046 residues. Every evaluated protein has a full-sequence ESMC teacher.
The experiment measures evidence for or against this limited extrapolation; it
does not assume that extrapolation succeeds.

Run commands from the repository root. Use the server environment described in
[the aggregation workflow](aggregation.md). Preparation requires only Python's
standard library; feature extraction requires ESMC; training and evaluation
require PyTorch. SVG plots additionally require matplotlib, included in
`requirements-server.txt`.

## Data and isolation rules

Lengths count cleaned amino-acid residues, after uppercasing and removing a
terminal stop. The 2,046-residue ceiling retains the existing allowance for two
ESMC special tokens. Exactly 1,500 residues belongs to the longer cohort.

The default scope is the existing NRPS, modular PKS, and HYBRID candidate filter.
The local annotations give 1,153 target candidates below 1,500 residues (715 NRPS,
435 PKS, 3 HYBRID), and 706 at 1,500–2,046 residues (203 NRPS, 486 PKS, 17 HYBRID).
These are candidate counts before grouped splitting, not final training/test
counts. Rule-based labels are not verified domain architectures.

Groups are built on the complete original input using BGC IDs, exact duplicate
sequences, and optional external homology clusters. The original approximately
80/10/10 group allocation runs **before** length filtering. Each group stays in
one split, even when it contains both short and longer proteins.

| Assigned split | Length <1,500 | Length 1,500–2,046 |
| --- | --- | --- |
| train | Parameter updates | Excluded |
| validation | Checkpoint selection and early stopping | Excluded |
| test | Short-protein test cohort | Length-extrapolation test cohort |

Proteins above 2,046 residues contribute no teacher samples or crops. No long
protein is truncated to become a training sample. Every retained sample is a
complete natural protein (`source=native`). All chunk views inherit its split.
Excluded parents retain their group and split in `parents.jsonl` and their reason
in the `experiment_role` column of `selection.tsv`.

Consequently, not all 706 longer candidates enter a single test run. Moving longer
training-group proteins to the test split would break group isolation. Inspect
`dataset.json` for parent/group/family counts in `summary.length_cohorts` and
exclusion counts in `summary.parent_roles`. The command fails when target train,
validation, short-test, or long-test samples are missing. It does not silently
relax isolation or search for a seed using model performance.

For research results, supply MMseqs2 clusters as described in
[Grouped splits and homology](aggregation.md#grouped-splits-and-homology). Without
`--clusters`, `homology_clustered=false`: only BGC and exact-duplicate isolation
are guaranteed. Sequence clustering does not prove that all shared local domains
or remote homologs have been separated.

## Prepare and cache once

After producing `core_homology_cluster.tsv` on the complete input FASTA:

```bash
python -m bgc_aggregation prepare \
  --protocol length-extrapolation \
  --length-cutoff 1500 \
  --scope target \
  --clusters data/processed/core_homology_cluster.tsv \
  --output data/processed/length1500

python -m bgc_aggregation cache \
  --dataset data/processed/length1500 \
  --output data/embeddings/length1500 \
  --model esmc_600m
```

This protocol defaults to `--crops-per-parent 0` and `--chunk-views fixed`.
A nonzero crop count is rejected. Standard preparation retains its previous
four-crop, multiscale defaults. Use a fresh output directory for this experiment;
existing standard datasets/checkpoints are not relabeled as length experiments.

Fixed views have 512-residue nonoverlapping cores, at most 64 context residues on
either side by default, and zero boundary offset. Only the cores are pooled.
Training, validation, and primary testing use this same partition.

| Protein length | Number of fixed chunks |
| --- | --- |
| <1,500 | At most 3 |
| 1,500–1,536 | 3 |
| 1,537–2,046 | 4 |

Thus the 1,537–2,046 subset tests both greater protein length and more chunks than
any fixed-view training protein. The 1,500–1,536 subset separates a small length
increase from an increase in chunk count. Neither subset is used for early
stopping or hyperparameter tuning.

The cutoff can be changed with `--length-cutoff` before preparing a new dataset.
It must be between 2 and 2,046; cohort boundaries and training restrictions follow
it. The 3-to-4-chunk interpretation above specifically describes cutoff 1,500.

## Run training and evaluation with one script

After completing feature extraction, the server instructions' steps 5 and 6 can
be run together from the active PyTorch environment:

```bash
python script/run_length_extrapolation.py
```

The default cache is `data/embeddings/length1500_no_mmseqs`, matching the
no-MMseqs2 server walkthrough. For the `length1500` paths in this guide, use:

```bash
python script/run_length_extrapolation.py \
  --cache data/embeddings/length1500 \
  --prefix length1500
```

The script runs Transformer and MLP training sequentially for seeds 42, 43, and
44, using fixed views, native target proteins, 100 maximum epochs, batch size 64,
learning rate 1e-4, and patience 10. After all six training calls succeed, it
evaluates all six best checkpoints on the test split, with 2,000 bootstrap draws,
bootstrap seed 42, and SVG plots. Weighted means are included automatically by
the evaluator. Relative cache paths and all output paths refer to the project
root, even when the script is invoked from another directory.

Runs, results, and logs use `<prefix>_<architecture>_s<seed>` names in `runs/`,
`results/`, and `logs/`. Logs append on reruns and are ignored by Git. A failed
command stops the script before any remaining jobs start. Matching existing
`run.json`, `last.pt`, and `best.pt` enable automatic checkpoint resume. Completed
training runs return without further epochs, and evaluation can be repeated;
existing evaluation files for the same run are replaced. Mismatched or incomplete
run directories are preserved and cause an error; use a new prefix for a different
experiment. The script does not generate data or ESMC features.

To inspect short validation results before deciding to start final testing:

```bash
python script/run_length_extrapolation.py --stage train
python script/run_length_extrapolation.py --stage evaluate
```

Both invocations must use the same cache and prefix when overriding defaults.
`--device cuda:0` selects a GPU; `--no-plots` omits SVGs and the matplotlib
requirement. The training environment is taken from the Python interpreter used
to launch the script. The explicit commands below remain available for individual
runs and custom experiments.

## Train the model and control

```bash
python -m bgc_aggregation train \
  --cache data/embeddings/length1500 \
  --views fixed --seed 42 \
  --output runs/length1500_transformer_s42

python -m bgc_aggregation train \
  --cache data/embeddings/length1500 \
  --architecture mlp --views fixed --seed 42 \
  --output runs/length1500_mlp_s42
```

The Transformer architecture and loss are unchanged: two residual encoder
layers, training-variance-normalized MSE plus 0.5 cosine distance. The MLP receives
only the weighted mean. The untrained weighted chunk mean is evaluated alongside
every checkpoint, so it needs no training run. Epoch zero remains eligible as
the best checkpoint; retaining it is not evidence of learned improvement.

For length-extrapolation caches, training defaults to fixed views and native
samples. `--sources crop` and `--init-from` are rejected. The aggregator starts
from random weights; the pretrained ESMC remains frozen. Teacher statistics are
computed only from the short training cohort. Checkpoint selection always uses
short held-out target proteins.

`run.json` and checkpoints record the experiment contract plus `training_inputs`
and `validation_inputs`: protein/group counts, length range, chunk-count range,
and view mode. `--resume` restores the original protocol, sampling mode, optimizer,
and random state; it cannot turn a fixed-view run into a multiscale run. It still
requires the original cache and run directory.

Repeat each trainable method with seeds 42, 43, and 44 while keeping preparation
seed, splits, cache, architecture, and hyperparameters fixed. Training seeds change
initialization and sampling, not protein/group membership. Choose methods using
short validation results, then freeze the planned comparisons before final testing.

## Evaluate the hypothesis

```bash
python -m bgc_aggregation evaluate \
  --cache data/embeddings/length1500 \
  --checkpoint runs/length1500_transformer_s42/best.pt \
  --split test --bootstrap-replicates 2000 --bootstrap-seed 42 \
  --plots --output results/length1500_transformer_s42
```

Repeat for the MLP and other predeclared training seeds. `--split validation` is
available for development and contains only short proteins. Avoid repeatedly
selecting changes from long-test performance. Evaluation calls the aggregator on
all held-out lengths; `embed` is not an evaluation substitute because it uses
direct ESMC embeddings for sequences within the teacher limit.

Outputs:

| Artifact | Contents |
| --- | --- |
| `metrics.json` | Existing metrics plus `length_extrapolation` with cohort/bin/family results and experiment contract |
| `per_sample.tsv` | Per-protein reconstruction errors, group, family, length, chunk count, and view |
| `length_summary.tsv` | Raw MSE, cosine distances, relative baseline improvement, paired confidence intervals, and group counts |
| `length_error.svg` | Binned raw MSE for aggregator and weighted-mean baseline; ALL, NRPS, and PKS panels |
| `length_gain.svg` | Relative MSE reduction with group-bootstrap intervals; the same panels |

Omit `--plots` to avoid the matplotlib dependency. Figures are standalone SVGs;
the TSV/JSON retain the numeric results and sample counts, including HYBRID.

Primary cohort names are `short`, `near_boundary_short` (1,024–1,499), `long`,
`long_within_chunk_range`, and `long_beyond_chunk_range`. The latter two compare
test chunk counts to the **maximum available training chunk count**, which is
recorded in the checkpoint. For multiscale sampling this is an upper bound across
all cached training views, not a claim that every view was sampled before the
selected checkpoint. With fixed views and training examples above 1,024
aa, they correspond to three versus four chunks in the longer cohort.

Default nonoverlapping length bins are 1–512, 513–1,023, 1,024–1,499,
1,500–1,536, 1,537–1,792, and 1,793–2,046. Each cohort and bin is also reported by
family. Empty buckets are explicit; sparse HYBRID results should remain
exploratory. Do not let the different NRPS/PKS proportions on either side of the
cutoff masquerade as a pure length effect.

The primary error is raw MSE. Relative improvement is
`1 - mean(model_raw_mse) / mean(baseline_raw_mse)` on the same proteins; positive
values favor the aggregator. A zero baseline denominator gives null, not an
infinite or fabricated improvement. Normalized loss is still reported, but raw
MSE is preferable when comparing models with different training variance scales.

Confidence intervals use a **paired group bootstrap**: sample `group_id` values
with replacement, include every protein in each sampled group, and recompute
protein-weighted mean errors. Sampling a large group repeats all its members;
the method does not treat those proteins as independent replicates. The 95%
percentile interval of model-minus-baseline raw MSE determines the reported
`evidence`:

- `improved`: the entire interval is below zero.
- `worse`: the entire interval is above zero.
- `inconclusive`: the interval includes zero.
- `insufficient_groups`: fewer than two groups; no confidence interval is reported.

These labels describe one cohort's reconstruction difference, not a global
verdict that the biological hypothesis is true. Intervals are not corrected for
multiple comparisons and do not include variability across training seeds;
report all planned seeds separately. A two-group interval can still be unstable.

For each cohort, neighbor retention compares predicted and teacher queries
against the same gallery of all selected held-out deployment-view teachers.
Candidates sharing the query group are excluded; k is reduced when necessary.
Short and long queries therefore use the same gallery. Raw and training-mean-
centered cosine versions are reported, together with eligible-query/gallery
counts. These are preservation metrics, not functional annotation accuracy.

`--all-views` adds alternate-view reconstruction rows to the generic report, but
the dedicated length report and neighbor metrics always use view zero. Thus
adding evaluation augmentations cannot change the primary extrapolation result.

## Multiscale augmentation comparison

Multi-scale training can expose short proteins to more chunks than the longer
fixed-view test proteins. It answers a related question about training with
augmentation, rather than the same unseen-chunk-count question as the main run.

To compare it efficiently, prepare a separate experiment with
`--chunk-views multiscale`, using the same inputs, clusters, cutoff, and preparation
seed, then cache it. Group/sample membership remains identical to fixed
preparation. That cache can support both `train --views fixed` (still the default
for the length protocol) and `train --views multiscale`, including the MLP control.
This is also a way to plan all comparisons using one shared cache from the start.
A fixed-only cache rejects `--views multiscale` instead of silently running a
fixed experiment under the wrong label. Validation and primary test views remain
fixed in both cases. Each run still starts from random aggregator weights.

## Scope of the conclusion

Stable improvement on the short and longer test cohorts, especially the four-
chunk subset, supports extrapolation within this tested interval. Improvement on
short proteins but degradation on longer ones indicates a possible generalization
failure. No improvement in either cohort does not isolate length as the cause.
Class/domain composition and homology novelty remain possible confounders; inspect
family-specific results and domain-matched subsets when annotations are available.

Success here does not establish reconstruction of 18,447-residue, 37-chunk proteins,
nor prove biological retrieval accuracy. Those claims require further experiments.
The code supplies the experiment and evidence; real ESMC feature extraction and
server training must run before drawing a scientific conclusion.
