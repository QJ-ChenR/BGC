# ESMC chunk aggregation

This workflow trains a small residual Transformer to approximate a full-sequence
ESMC residue-mean embedding from independently encoded chunks. The target use is
long NRPS/PKS proteins. ESMC is frozen throughout; only the aggregator is trained.
All commands below run from the repository root.

## Local versus server requirements

Data preparation, CLI help, and preparation tests use only Python's standard
library. They do not import PyTorch, ESM, NumPy, or Biopython. Python 3.12 is
recommended for the server; the preparation tools also work with Python 3.13.

The training target is the existing server environment with PyTorch
`2.11.0+cu130`, CUDA available, and an NVIDIA L40S with 48 GB memory. Do not
install GPU dependencies on the laptop just to prepare the data.

On the server, activate the environment containing that PyTorch build, then run:

```bash
python -m pip install -r requirements-server.txt
python -m bgc_aggregation check-env
python -m unittest discover -s tests -v
python -m bgc_aggregation check-env --check-esmc --model esmc_600m
```

The requirements pin the existing CUDA PyTorch build rather than allowing ESM's
dependencies to select another version. They pin `esm==3.2.1` and
`transformers==4.48.1` because the adapter targets that ESMC API; newer ESM package
releases have changed the API. This is a proposed server dependency combination,
not a claim that CUDA execution has already been verified on this laptop.

The ESM package declares additional dependencies, including torchvision and
torchtext; this project does not directly use their APIs. If dependency resolution
fails, keep the Torch pin and inspect the conflict rather than downgrading Torch.
Run the actual ESMC smoke check before launching feature extraction.

The smoke check downloads official weights through Hugging Face if needed and
compares single-sequence and padded-batch pooling. Authenticate with Hugging Face
if the selected model repository requires access. `--weights /path/to/model.pth`
can supply an already downloaded official checkpoint instead. No separate
`flash-attn` installation is needed: ESMC uses PyTorch SDPA.

PyTorch numerical tests are not silently substituted with mock training: when Torch
is absent, the numerical/workflow tests are explicitly reported as skipped.
The numerical tests use synthetic features and do not need ESM or model downloads.

## 1. Prepare and inspect the dataset

The existing extraction outputs are the inputs:

- `data/processed/core_genes/core_genes.faa`
- `data/processed/core_genes/core_genes.tsv`

For an initial local preparation check:

```bash
python -m bgc_aggregation prepare \
  --scope all \
  --output data/processed/aggregation_preview
```

`--scope all` retains all natural short proteins, so one cache can support both
target-only training and all-family pretraining. Long-protein crops are generated
only for target NRPS/PKS families. Training itself defaults to `--scope target`.
Use preparation `--scope target` if the all-family comparison is not needed.

The automatic candidate selection uses each protein's `gene_functions` rule hits:

| Family | Rule hits |
| --- | --- |
| NRPS | NRPS, NRPS-like, NRP-metallophore |
| PKS | T1PKS, transAT-PKS, transAT-PKS-like, PKS-like |
| HYBRID | Both NRPS and PKS hits |
| OTHER_PKS | T2PKS, T3PKS, HR-T2PKS without target hits |
| OTHER | No recognized target hits |

Only NRPS, PKS, and HYBRID are target families. This deliberately excludes type II
and type III PKSs from the modular-protein target set. Rule hits are a candidate
selection heuristic, not a verified domain architecture. Inspect `selection.tsv`,
especially NRPS-like and PKS-like calls, and supply reviewed overrides if needed:

```text
sequence_id	family
BGC0000001|16|abyB3	PKS
```

Here the columns must be separated by an actual tab. Use `--labels reviewed.tsv`;
rows not listed retain their automatic family. Product names and BGC-wide labels
are not used to assign a synthase family. Domain boundaries are not inferred.
You can copy `selection.tsv`, edit its `family` column, and pass that reviewed
copy with `--labels`; the other audit columns are ignored by the override reader.

The local 6,831-protein dataset produced 1,859 short target candidates and 2,700
long target candidates with these rules. Nine additional long proteins have
`hglE-KS` hits and are outside the default filter. They are listed in
`dataset.json` under `unselected_long_ids`, and in the selection audit. Review
these PKS-related candidates and use explicit overrides to include them if that
matches the intended biological scope; they are not silently relabeled.

Sequences are uppercased; one terminal `*` is removed. Internal stops, gaps,
empty sequences, and duplicate FASTA identifiers are rejected. Ambiguous residue
symbols supported by the ESM tokenizer are retained, never silently deleted.

### Grouped splits and homology

Every parent protein is assigned to one split before cropping. All proteins from
the same BGC and all exact duplicate sequences are linked. If a homology-cluster
TSV is supplied, those links are added transitively before splitting into
approximately 80% training, 10% validation, and 10% test parents. Large connected
groups can make the proportions or family counts uneven; inspect the summary.

For research evaluation, generate homology clusters on the complete input FASTA.
For example, with MMseqs2 available on the server:

```bash
mmseqs easy-cluster \
  data/processed/core_genes/core_genes.faa \
  data/processed/core_homology \
  data/tmp/mmseqs \
  --min-seq-id 0.4 -c 0.8 --cov-mode 0

python -m bgc_aggregation prepare \
  --scope all \
  --clusters data/processed/core_homology_cluster.tsv \
  --output data/processed/aggregation
```

These clustering thresholds are starting choices, not proof that all remotely
related domains have been separated. The headerless TSV must contain
`representative_id<TAB>member_id`, cover every input protein (including
singletons), and contain no unknown IDs. Without it, preparation still runs but
records `homology_clustered=false`: only BGC and exact-duplicate separation is
guaranteed. Never report that preview split as homology-controlled.

The input tables, split seed, selected scope, crop parameters, file hashes,
family counts, and largest group size are recorded in `dataset.json`.
Preparation requires an empty output directory to avoid silently replacing splits.

### Teachers, crops, and chunk views

Natural proteins of at most 2,046 residues have a full-sequence teacher. The
2,048-token budget reserves BOS and EOS. Longer target proteins contribute up to
four continuous crops per parent, with lengths 1,024, 1,536, 2,046, and 2,046.
The first two cover the termini and the others use seeded internal positions.
Set `--crops-per-parent 0` for a natural-short-protein-only dataset.

Crops inherit their parent's split. Training uses only training-parent crops;
validation/test crops provide isolated-fragment reconstruction checks on held-out
parents. Their teachers do not represent the embedding of the full long protein.
No unrelated fragments are concatenated into artificial teacher proteins.

Each teacher sample has the deployment partition plus unique alternative
partitions with core sizes 128, 256, 512, and 768 and seeded boundary offsets.
The default deployment partition has 512-residue nonoverlapping cores and up to
64 context residues on each side. ESMC reads the context-extended sequence, but
pooling includes only core residues. Every residue therefore contributes exactly
once to the length-weighted mean. All coordinates are zero-based, end-exclusive.

Crop coordinates and positions refer to the isolated teacher fragment during
training. During full-protein inference they refer to the complete protein.
The two terminal features indicate ends of the represented sequence, including
artificial crop ends; they must not be interpreted as verified biological termini.

## 2. Extract frozen features on the server

```bash
python -m bgc_aggregation cache \
  --dataset data/processed/aggregation \
  --model esmc_600m \
  --output data/embeddings/esmc600_aggregation
```

The 600M model produces 1,152-dimensional vectors; `--model esmc_300m` produces
960-dimensional vectors. Use separate cache directories and checkpoints for each.
CUDA extraction uses BF16; pooling and saved vectors use FP32. ESMC runs in
evaluation/inference mode. BOS, EOS, and padding never enter the residue mean.

Defaults are `--esm-batch-size 2 --token-budget 4096`. Batches are length-sorted
and bounded by padded token count. On out-of-memory errors, reduce the batch size
to one and rerun; completed samples are reused. Extraction resumes at complete
teacher samples, using atomic file replacement. A `--limit 10` smoke extraction
creates an explicitly incomplete cache; rerun without the limit before training.

`cache.json` stores the weight-file SHA-256, ESM version, pooling convention,
precision, dataset fingerprint, and hashes of feature files. Changed datasets or
encoder settings require a new cache. A complete cache is sufficient for training
without importing ESM or loading its weights. Feature extraction is expected to
dominate computation; throughput and peak memory need measurement on the server.

## 3. Train the aggregator

```bash
python -m bgc_aggregation train \
  --cache data/embeddings/esmc600_aggregation \
  --output runs/aggregation_target
```

Defaults:

| Component | Configuration |
| --- | --- |
| Backbone | Two bidirectional pre-LN Transformer encoder layers |
| Input/output dimensions | ESMC dimension, inferred from cache |
| Hidden dimension / heads / FFN | 128 / 4 / 512 |
| Dropout | 0.1 |
| Position features | Relative core start/end, core length / 512, two terminal flags |
| Readout | Core-length-weighted mean of encoded chunks |
| Prediction | Original weighted chunk mean plus learned correction |
| Initial correction | Zero output projection and bias |
| Objective | Training-variance-normalized MSE + 0.5 cosine distance |
| Optimizer | AdamW, learning rate 1e-4, weight decay 0.01 |
| Schedule | 5% linear warmup, then cosine decay |
| Batch / maximum epochs / patience | 64 parents / 100 / 10 |
| Gradient clipping | 1.0 |

The aggregator trains in FP32; at this size mixed precision is unnecessary.
Padding is masked in attention and pooling. The model has no fixed chunk-count
position table, but quality beyond the training chunk-count distribution remains
an empirical question.

Each epoch samples one example per parent: first a crop (if applicable), then a
view. Half the view choices explicitly use the deployment partition, and the
remainder sample the available views. Parents with four crops do not receive four
times the training weight. Teacher mean and variance are computed from training
parents only, weighting parents equally and averaging their crops internally.

Validation uses all selected held-out target samples with the deployment
partition; metrics first average samples within each parent. Validation always
targets NRPS/PKS, even for all-family pretraining. The test split is not consulted.
`best.pt` includes the initial baseline (epoch zero) as a candidate, so an
unsuccessful training run can retain the original weighted-mean solution.

Artifacts are `run.json`, `history.json`, `best.pt`, and `last.pt`.
To resume after an interruption:

```bash
python -m bgc_aggregation train \
  --cache data/embeddings/esmc600_aggregation \
  --output runs/aggregation_target \
  --resume runs/aggregation_target/last.pt
```

Resume restores the saved hyperparameters, optimizer, scheduler, epoch, early
stopping state, and RNG state; other training flags do not override the saved
configuration. It resumes from the last completed epoch. Use a new run and
`--init-from` for fine-tuning with a different learning rate or training scope.

## 4. Compare training choices

Use one prepared dataset with `--scope all` and one feature cache for every
comparison, so protein/group membership and teacher vectors remain identical.

```bash
# A: All natural short proteins.
python -m bgc_aggregation train \
  --cache data/embeddings/esmc600_aggregation \
  --scope all --sources native --output runs/all_short

# B: Natural short NRPS/PKS proteins only.
python -m bgc_aggregation train \
  --cache data/embeddings/esmc600_aggregation \
  --scope target --sources native --output runs/target_short

# C: Initialize from all-short pretraining, then adapt to target short proteins.
python -m bgc_aggregation train \
  --cache data/embeddings/esmc600_aggregation \
  --scope target --sources native --learning-rate 3e-5 \
  --init-from runs/all_short/best.pt --output runs/target_finetuned

# D: Target natural short proteins plus target long-protein crops.
python -m bgc_aggregation train \
  --cache data/embeddings/esmc600_aggregation \
  --scope target --sources both --output runs/target_with_crops

# Mean-vector MLP control, using the same residual output convention.
python -m bgc_aggregation train \
  --cache data/embeddings/esmc600_aggregation \
  --architecture mlp --output runs/target_mlp
```

The MLP receives only the weighted mean, not an ordered chunk sequence.
`--no-positions` supplies a Transformer position-feature ablation. Repeat promising
experiments with multiple training `--seed` values while keeping the prepared
dataset and splits fixed. Fine-tuning requires the same cache and architecture;
creating a new split for fine-tuning could expose held-out evaluation proteins.

## 5. Evaluate without using the test set for model selection

Use `--split validation` during method selection. Once the choices are fixed:

```bash
python -m bgc_aggregation evaluate \
  --cache data/embeddings/esmc600_aggregation \
  --checkpoint runs/aggregation_target/best.pt \
  --split test --output results/aggregation_target
```

`metrics.json` and `per_sample.tsv` report raw MSE, normalized MSE, cosine distance,
and combined loss, including the weighted-mean and training-teacher-mean baselines.
Summaries are parent-balanced and stratified by family, native/crop source, length,
core size, and chunk count. A model selected on mixed native/crop validation can
trade off these populations; inspect their separate metrics before choosing it.

Use `--sources native` to compare the same natural-short target test population
across all training methods. Training-variance-normalized losses from different
training scopes have different denominators: compare raw MSE, cosine distance,
and improvement over the common weighted-mean baseline across those runs.

Neighborhood retention compares predicted-query and teacher-query top-k sets in
the same teacher gallery. It uses one deterministic representative sample per
parent, excludes the entire query group, and reduces k if too few candidates
remain. Both raw and training-mean-centered cosine geometries are reported.
`--all-views` also evaluates alternate partitions; neighborhood metrics still use
only the deployment view. Crop-based neighborhoods concern fragments, not the
full long proteins.

These are reconstruction checks. Full-length long proteins have no trusted
in-window teacher. Domain-architecture agreement, biological retrieval labels,
taxonomy/length controls, and full-length chunk-count extrapolation need separate
biological evaluation; this implementation does not claim those have been tested.

## 6. Embed full proteins

```bash
python -m bgc_aggregation embed \
  --fasta data/processed/core_genes/core_genes.faa \
  --checkpoint runs/aggregation_target/best.pt \
  --output data/embeddings/core_proteins_aggregated
```

Proteins up to 2,046 residues use direct full-sequence ESMC pooling. Longer inputs
use 512-residue cores, the checkpoint's context halo, and the learned aggregator.
This target-trained model is intended for long NRPS/PKS proteins; the inference
command does not infer biological families from raw FASTA.

`embeddings.pt` contains:

- `ids`: ordered FASTA identifiers;
- `embeddings`: raw FP32 protein vectors;
- `normalized_embeddings`: L2-normalized vectors for cosine retrieval;
- `weighted_mean_embeddings`: the uncorrected baseline for comparison;
- `metadata`: sequence lengths, chunk counts, and direct/aggregated methods.

`proteins.json` provides readable metadata. Per-protein shards support resuming
the same command. The model-weight hash, precision, pooling convention, and
checkpoint must match training. Do not combine embeddings from different ESMC
models or independently normalized coordinate systems in one retrieval index.

## Implementation references

- [Pinned ESMC model and output API](https://github.com/evolutionaryscale/esm/blob/v3.2.1/esm/models/esmc.py)
- [Pinned ESMC model dimensions and weight names](https://github.com/evolutionaryscale/esm/blob/v3.2.1/esm/pretrained.py)
- [Pinned attention implementation](https://github.com/evolutionaryscale/esm/blob/v3.2.1/esm/layers/attention.py)
- [PyTorch 2.11 SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html)
