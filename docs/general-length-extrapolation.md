# General-protein length-extrapolation training

This workflow reuses the frozen ESMC encoder, chunk cache, residual MLP/BiLSTM/
Transformer, loss, checkpointing, and evaluation code. It adds a separate TSV
importer and experiment contract, `general-length-extrapolation`. No MMseqs2
installation or sequence-homology clustering is required.

The experiment trains on general proteins of 500–1,499 aa, selects checkpoints
using short validation proteins, and tests on short held-out proteins plus the
entire supplied 1,500–2,000 aa file. It measures length extrapolation with possible
homology overlap. A homology rate below 5% is an unverified working assumption;
the pipeline neither assumes that value in its calculations nor certifies it.

## Inputs and grouping

The default inputs are:

- `data/raw/general/protein_sequences_500_1499aa_50000.tsv`
- `data/raw/general/protein_sequences_1500_2000aa.tsv`

Both are tab-separated, with these columns:

```text
accession group other_subgroup organism scientific_name organism_taxon_id species_key_taxon_id genus_taxon_id length_aa reviewed protein_existence protein_name sequence_sha256 uniprot_url sequence
```

The importer checks required columns, complete rows, numeric taxonomy IDs,
uppercase ungapped sequences, stated lengths, SHA256 checksums, and duplicate
accessions/full sequences across both files. `U` and the other residue symbols
accepted by the existing ESMC pipeline remain valid. It rejects bad inputs instead
of silently dropping, truncating, or relabeling proteins. Original annotations are
retained in `parents.jsonl`; taxonomy and provenance needed for evaluation also
survive feature caching.

The supplied TSV's `group` column means Bacteria/Fungi/Other. It is saved as
`taxonomy_group`, not used as a split group or an NRPS/PKS label. The actual
`group_id` is `species:<species_key_taxon_id>`. General proteins have
`family=GENERAL` and `bgc_id=null`.

The short pool is allocated approximately 80/10/10 to train, validation, and
short test. A deterministic greedy allocation keeps each species intact and
balances protein counts and the joint taxonomy/length strata
(Bacteria/Fungi/Other × 500–512, 513–1,023, 1,024–1,499 aa). Exact fractions are
not guaranteed for arbitrary group sizes. Allocation uses only short-pool
metadata, never long-test composition or model scores.

The long file remains entirely in test. Its species may occur in any short
split. Groups are not renamed by split to hide this overlap. Protocol validation
enforces species isolation **within the short pool**, plus distinct proteins and
exact sequences across all rows. The older MIBiG protocol retains its global
BGC/group isolation rules.

For the currently supplied files and preparation seed 42, the actual allocation is:

| Cohort | Proteins | Species groups | Bacteria / Fungi / Other |
| --- | ---: | ---: | --- |
| Train | 40,000 | 14,772 | 20,000 / 10,000 / 10,000 |
| Validation | 5,000 | 1,847 | 2,500 / 1,250 / 1,250 |
| Short test | 5,000 | 1,847 | 2,500 / 1,250 / 1,250 |
| Long test | 10,000 | 8,758 | 5,000 / 2,500 / 2,500 |

The supplied long file currently contains **10,000**, not 20,000 proteins.
Of those, 7,037 have a species present in the actual training split; 2,963 do not.
These counts describe the current input hashes and seed, not hardcoded sample
limits. `dataset.json` records the actual counts, hashes, protocol, and split seed.
Replacing an input requires a new prepared dataset and cache.

## Prepare and extract ESMC features

Run from the repository root. Preparation needs only Python's standard library.
Use the existing [server environment](aggregation.md) for GPU stages; this change
does not require replacing the server's CUDA PyTorch installation.

```bash
python -m bgc_aggregation prepare-general \
  --output data/processed/general_length1500

python -m bgc_aggregation cache \
  --dataset data/processed/general_length1500 \
  --output data/embeddings/general_length1500 \
  --model esmc_600m --device cuda \
  --esm-batch-size 2 --token-budget 4096
```

Use `--short-tsv`, `--long-tsv`, and `--seed` to override input paths or preparation
seed. Preparation requires a new or empty directory. If the local dataset has
already been prepared, use it directly instead of preparing into the same folder.

Each protein supplies one full-sequence teacher and one fixed chunk view:
512-aa nonoverlapping cores, 64-aa flanking context by default, zero offset, and
core-only pooling. No crops or multiscale views enter this simplified protocol.
ESMC is frozen. The model inputs do not include taxonomy or species IDs.

Use a persistent terminal session for extraction, for example:

```bash
tmux new -s general-esmc
```

Activate the server environment and run the cache command inside that session.
Detach with Ctrl-b followed by d; reconnect with
`tmux attach -t general-esmc`. Rerunning the identical cache command reuses completed
per-protein files. A cache created with `--limit` remains incomplete and cannot
be used for training until extraction finishes. Feature extraction is substantially
larger than the earlier MIBiG experiment; measure throughput on the server instead
of extrapolating a precise runtime from GPU memory alone.

## Train all three architectures

The dedicated runner shares logging, command construction, matching-run checks,
and resume logic with the existing MIBiG runner:

```bash
python script/run_general_length_extrapolation.py --stage train
```

Defaults are:

| Setting | Value |
| --- | --- |
| Architectures | Transformer, MLP, bidirectional LSTM |
| Training seeds | 42, 43, 44 |
| Cache | `data/embeddings/general_length1500` |
| Output prefix | `general_length1500` |
| Batch size / learning rate | 64 / 1e-4 |
| Epoch ceiling / optimizer-update ceiling | 20 / 10,000 |
| Early-stopping patience | 5 validation checks |
| Validation | All general short-validation proteins, fixed views |
| Loss | Training-variance-normalized MSE + 0.5 cosine distance |

Model dimensions, dropout, weight decay, warmup fraction, and residual heads match
the earlier comparison. The aggregator starts from random weights; ESMC retains
its pretrained weights. Teacher mean/variance come only from the training split.

Training stops at the first of the epoch limit, update limit, or early-stopping
criterion. Warmup and cosine decay use
`min(epochs * ceil(training_proteins / batch_size), max_steps)`. With the current
40,000 training proteins, one epoch is 625 updates and the default cap is 16 full
epochs. Validation runs after each full epoch and after a final partial epoch if
the update cap falls inside an epoch. Epoch zero remains eligible as best.

The 10,000-update budget is a starting experiment setting, not evidence that all
methods converge by that point. Inspect short-validation histories before final
testing. A different budget requires a new prefix; resume preserves the original
optimizer, scheduler, sampling, RNG, and budget. For example, a new 20,000-update
comparison also needs a sufficiently large epoch ceiling:

```bash
python script/run_general_length_extrapolation.py --stage train \
  --prefix general_length1500_steps20000 --epochs 40 --max-steps 20000
```

Use the same budget/prefix arguments when evaluating those runs. Keep the split
and feature cache fixed across architectures and training seeds.

To run one architecture/seed:

```bash
python script/run_general_length_extrapolation.py \
  --architectures mlp --seeds 42 --stage train
```

The equivalent individual training command is:

```bash
python -m bgc_aggregation train \
  --cache data/embeddings/general_length1500 \
  --output runs/general_manual_mlp_s42 \
  --architecture mlp --scope all --sources native --views fixed \
  --epochs 20 --max-steps 10000 --patience 5 --seed 42
```

General training/evaluation require `--scope all`; the runner supplies it. The
old runner still defaults to target NRPS/PKS, Transformer+MLP, and its original
100-epoch settings. Existing checkpoints without an update cap remain resumable.

## Evaluate once the comparisons are fixed

```bash
python script/run_general_length_extrapolation.py --stage evaluate
```

Without `--stage`, the runner trains all selected runs before any test evaluation.
It stops on failure. Matching runs resume from `last.pt`; `best.pt` is selected
using short validation only. Long-test results never enter training or early
stopping. Use validation to choose training budgets and model settings, then fix
the comparisons before viewing the long test.

Outputs use `<prefix>_<architecture>_s<seed>`:

| Directory | Files |
| --- | --- |
| `runs/general_length1500_mlp_s42/` | `run.json`, `history.json`, `best.pt`, `last.pt` |
| `results/general_length1500_mlp_s42/` | `metrics.json`, `per_sample.tsv`, `length_summary.tsv`, SVG plots |
| `logs/` | Training and evaluation logs for each run |

The same naming applies to the other architectures/seeds. `run.json` records
the planned update budget and updates per full epoch; checkpoint/history entries
record `global_step` for comparing convergence by updates.

The length report contains short, near-boundary short, long, within-training-
chunk-range, and beyond-training-chunk-range cohorts. At the current lengths,
long proteins split into 1,519 three-chunk and 8,481 four-chunk examples. Additional
cohorts are `long_species_seen_in_training` and
`long_species_unseen_in_training`.

Cohorts and length bins use `by_taxonomy_group` in JSON and `taxonomy_group` in
`length_summary.tsv`. Plots show ALL/Bacteria/Fungi/Other. Per-protein rows retain
species, taxonomy, source-pool provenance, and `species_seen_in_training`.
Length bins are 500–512, 513–1,023, 1,024–1,499, 1,500–1,536, 1,537–1,792,
and 1,793–2,000 aa. The generic report also includes `by_other_subgroup`.

Raw MSE and cosine distance compare each predicted vector with its full-sequence
teacher. The paired baseline is the core-length-weighted mean. Positive
`relative_mse_reduction` favors the learned aggregator. Confidence intervals
resample species groups with protein weighting; the general protocol uses
memory-bounded NumPy bootstrap batches, while the legacy protocol retains its
existing Python bootstrap draws. These intervals do not correct unknown homology
across species or similarity between training and test proteins.

Neighbor retention uses a shared gallery of selected held-out teachers, excludes
same-species candidates, and computes raw/centered scores once per protein before
reusing them across cohorts. The full search remains quadratic in gallery size.
To omit it while retaining reconstruction and confidence intervals:

```bash
python script/run_general_length_extrapolation.py --stage evaluate --skip-neighbors
```

Skipped neighbor metrics are explicitly null with `neighbors_skipped=true`.
Use `--no-plots` to omit SVGs. Evaluation flags do not change training or require
a new prefix. Reports for the same run are replaced when reevaluated.

## Scope and subsequent NRPS/PKS work

Species-disjoint short testing and fixed long testing answer different novelty
questions. A long protein from an unseen training species can still be homologous
to training proteins. Improvement here supports reconstruction in the tested
length interval under this stated protocol; it does not establish sequence-
independent extrapolation, correct embeddings beyond 2,046 aa, or NRPS/PKS
biological retrieval accuracy.

This implementation covers general training and its length experiment. It does
not add cross-dataset fine-tuning: `--init-from` remains rejected for length
experiments, and checkpoint/cache identity checks remain enforced. Later
general-to-MIBiG fine-tuning needs a separate transfer protocol and target-data
evaluation.

Raw inputs, prepared data, caches, runs, logs, and results remain ignored by Git.
Code and documentation are versioned. Run regression tests with:

```bash
python -m unittest discover -s tests -v
```

PyTorch tests use small synthetic features/fake encoders to validate the workflow;
they do not substitute for real ESMC inference on the server.
