# BGC Retrieval System — Project Design

Retrieval system for Biosynthetic Gene Clusters (BGCs) using MiBIG data and
modern embedding methods. Design document — written before implementation.

## 0. Central design decision

A BGC is an **ordered set of genes** (typically 5–50 CDS) plus metadata, not a
single sequence. Protein language models (ESM) embed single proteins. The
pipeline is therefore hierarchical:

```
BGC → CDS/proteins → per-protein embeddings → aggregation → one BGC vector
```

The aggregation step is the main experimental axis. The architecture treats
"embedder" as a swappable plugin behind a common interface; clustering,
evaluation, visualization and retrieval are embedder-agnostic.

## 1. Architecture (layered pipeline)

```
┌──────────────────────────────────────────────────────────────┐
│ 1. ACQUISITION   download MiBIG release (JSON + GenBank),    │
│                  pin version, checksum, never modify raw/    │
├──────────────────────────────────────────────────────────────┤
│ 2. PARSING/ETL   GenBank+JSON → canonical schema:            │
│                  bgc_id, class, taxonomy, compounds (SMILES),│
│                  genes [(id, product, aa_seq, kind, order)]  │
│                  → data/processed/bgcs.parquet + FASTA       │
├──────────────────────────────────────────────────────────────┤
│ 3. EMBEDDING     plugin interface `BGCEmbedder`              │
│                  baselines: class-composition, Pfam TF-IDF,  │
│                  pfam2vec; PLM: ESM-2 (+pooling variants)    │
│                  → data/embeddings/{name}.npz                │
├──────────────────────────────────────────────────────────────┤
│ 4. ANALYSIS      clustering (HDBSCAN/k-means),               │
│                  visualization (UMAP/t-SNE, interactive)     │
├──────────────────────────────────────────────────────────────┤
│ 5. EVALUATION    class-label metrics, retrieval metrics,     │
│                  compound-similarity correlation, GCF check  │
├──────────────────────────────────────────────────────────────┤
│ 6. RETRIEVAL     FAISS index + query pipeline                │
│                  (query GenBank → parse → embed → k-NN)      │
├──────────────────────────────────────────────────────────────┤
│ CROSS-CUTTING    YAML configs, CLI (typer), logging, seeds,  │
│                  pytest, conda env, cached artifacts on disk │
└──────────────────────────────────────────────────────────────┘
```

Key contracts:

- **Canonical schema** between parsing and everything else. Parquet table of
  BGCs + one FASTA of all proteins (headers `BGC0000001|gene_idx|locus_tag`).
- **Embedder interface**: `embed(bgcs) -> (ids, matrix[N, d])`, registered by
  name, selected via config. Each run writes `{name}.npz` + a JSON sidecar
  recording config, model version, git commit, date.
- **Stage caching**: every stage reads files written by the previous stage, so
  stages are re-runnable and debuggable independently.

## 2. Directory structure

```
BGC/
├── environment.yml
├── pyproject.toml              # src-layout package: pip install -e .
├── README.md
├── DESIGN.md
├── configs/
│   ├── data.yaml               # MiBIG version, URLs, filters
│   ├── embed/                  # one file per embedder
│   │   ├── class_composition.yaml
│   │   ├── pfam_tfidf.yaml
│   │   ├── pfam2vec.yaml
│   │   ├── esm2_35m_mean.yaml
│   │   └── esm2_650m_weighted.yaml
│   └── eval.yaml
├── data/                       # gitignored, regenerable
│   ├── raw/mibig_4.0/          # untouched downloads
│   ├── processed/              # bgcs.parquet, proteins.faa, pfam hits
│   └── embeddings/             # {embedder}.npz + .meta.json
├── src/bgc_retrieval/
│   ├── data/       (download.py, parse_mibig.py, schema.py)
│   ├── features/   (pfam_annotation.py)
│   ├── embedders/  (base.py, baseline.py, pfam.py, plm.py, registry)
│   ├── analysis/   (cluster.py, project.py, plots.py)
│   ├── eval/       (labels.py, retrieval_metrics.py, chem.py, benchmark.py)
│   ├── retrieval/  (index.py, query.py)
│   └── cli.py      # bgc download|parse|embed|evaluate|visualize|query
├── notebooks/                  # exploration only; logic lives in src/
├── tests/                      # pytest + 2–3 tiny fixture BGCs
└── results/                    # figures, metric tables per run
```

Conventions: raw data immutable; notebooks never define pipeline logic;
every artifact traceable to a config + code version.

## 3. Milestones

| # | Milestone | Deliverable / exit criterion |
|---|-----------|------------------------------|
| M0 | Scaffold | conda env, package skeleton, CI-able pytest, CLI stub |
| M1 | Data | MiBIG parsed → bgcs.parquet + proteins.faa; EDA notebook (class/taxon/gene-count distributions); parser unit tests |
| M2 | Baselines | class-composition + Pfam TF-IDF embeddings; UMAP colored by class; eval harness v1 (silhouette, kNN accuracy) |
| M3 | PLM embeddings | ESM-2 per-protein embedding cache; mean-pooled BGC vectors; same eval — first baseline-vs-PLM comparison |
| M4 | Evaluation deep-dive | retrieval metrics (mAP, P@k), compound Tanimoto correlation, taxonomy-controlled analysis; benchmark table across all embedders |
| M5 | Retrieval system | FAISS index + `bgc query my_cluster.gbk --top-k 10` end-to-end |
| M6 | Extensions (optional) | weighted/order-aware pooling, contrastive fine-tuning, BiG-SLiCE comparison, small web UI |

M2 before M3 is deliberate: the eval harness must exist and be trusted on
cheap embeddings before spending GPU hours on ESM.

## 4. MiBIG data & metadata

Source: https://mibig.secondarymetabolites.org (v4.0, ~3,000 curated entries;
verify current release at download time and pin it in `configs/data.yaml`).
Downloads: JSON metadata archive + GenBank archive, one file per BGC
(`BGC0000001` …).

Per entry:
- **Sequence** (GenBank): full nucleotide locus, CDS features with
  translations, gene/locus_tag/product qualifiers; antiSMASH-style gene
  `kind` annotations (core biosynthetic / additional / transport / regulatory)
  for many entries.
- **biosyn_class** (JSON): NRP, Polyketide, RiPP, Terpene, Saccharide,
  Alkaloid, Other — multi-label. Primary evaluation label.
- **Compounds**: name, SMILES, chemical activities (antibacterial, cytotoxic…),
  molecular targets. SMILES → chemistry-based ground truth.
- **Taxonomy**: organism name + NCBI taxid → confound control.
- **Completeness / status / evidence**: filter to complete, non-retired
  entries with experimental evidence for the eval set.
- **Publications, cross-links** (PubMed, DOI): provenance in retrieval output.

Companion resources (optional, later): antiSMASH-DB and BiG-FAM (GCF
memberships usable as an external clustering ground truth); BiG-SCAPE run
locally on MiBIG gives GCF labels for evaluation.

## 5. Embedding approaches (simple → advanced)

1. **Gene-class composition** (trivial baseline): counts/fractions of gene
   kinds, gene count, cluster length, GC. ~10 dims. Sanity floor — anything
   fancier must beat it.
2. **Pfam domain TF-IDF** (strong classical baseline): pyhmmer scan of all
   proteins vs Pfam-A → bag-of-domains per BGC → TF-IDF (+ optional truncated
   SVD to ~256 dims). This is the representation family behind
   BiG-SLiCE/BiG-FAM; interpretable and fast.
3. **pfam2vec**: word2vec over ordered domain "sentences" per BGC, averaged.
   Adds domain co-occurrence semantics with negligible compute.
4. **ESM-2 mean pooling** (main PLM method): per-protein mean-pooled ESM-2
   embeddings, averaged (optionally length-weighted) over the BGC's genes.
   Dev model: `esm2_t12_35M` (480d); final: `esm2_t33_650M` (1280d).
   Cache per-protein embeddings once; pooling variants are then free.
5. **Weighted pooling**: same per-protein vectors, weight core biosynthetic
   genes higher (from gene-kind annotations); or embed only core genes.
   Tests the hypothesis that transport/regulatory genes are noise.
6. **Alternative PLMs**: ProtT5-XL, ESM-C — same interface, ablation.
7. **Order-aware / learned aggregation** (stretch): small transformer or
   attention pooling over the gene-embedding sequence, trained contrastively
   (positives = same class or same GCF). First method that uses synteny.
8. **DNA language models** (stretch, likely weaker): Nucleotide
   Transformer / HyenaDNA on cluster DNA — captures intergenic signal but
   context length and noise are problematic; include only as a comparison.

All produce `(N, d)` matrices behind the same interface → identical
downstream treatment. Compute: ~3,000 BGCs × ~30 genes ≈ 90k proteins;
ESM-2 650M ≈ a few GPU-hours once (35M model runs on CPU/laptop GPU).

## 6. Evaluation strategy

No single ground truth for "biologically related" — triangulate:

1. **Biosynthetic class** (primary, cheap): kNN classification accuracy /
   macro-F1 (multi-label aware); silhouette + ARI/NMI of clustering vs class;
   UMAP visual check. Caveat: classes are coarse (7 labels).
2. **Retrieval metrics** (matches the end goal): leave-one-out — each BGC
   queries the rest; relevance = shared class (v1) or shared GCF (v2).
   Report Precision@k (k=1,5,10), mAP, nDCG.
3. **Compound similarity** (strongest biology): Morgan fingerprints from
   SMILES (RDKit) → Tanimoto similarity between BGC pairs; Spearman
   correlation with embedding cosine similarity; enrichment: do top-k
   retrieved BGCs make significantly more similar compounds than random?
   Independent of the class labels.
4. **GCF agreement**: BiG-SCAPE (or BiG-FAM memberships) on MiBIG → gene
   cluster families; ARI between embedding clusters and GCFs; retrieval with
   GCF as relevance. Compares directly against the established tool.
5. **Confound control — taxonomy**: PLM embeddings may encode phylogeny, not
   biosynthesis. Check kNN class accuracy restricted to cross-genus /
   cross-phylum neighbors; if accuracy collapses, embeddings capture taxonomy
   rather than function.

Protocol: fixed eval set (complete entries only), fixed seeds, one benchmark
script producing a per-embedder metric table (CSV + markdown in results/).
Every embedder — including trivial baselines — goes through the identical
harness.

## 7. Implementation roadmap

1. **Scaffold (M0)**: environment.yml (python 3.11; biopython, pandas,
   pyarrow, scikit-learn, umap-learn, hdbscan, matplotlib, plotly, rdkit,
   pyhmmer, torch, fair-esm or huggingface transformers, faiss-cpu, typer,
   pyyaml, pytest, ruff). `pip install -e .`, CLI stub, pre-commit, git init.
2. **Download (M1)**: `bgc download` — fetch pinned MiBIG JSON+GenBank
   archives to data/raw/, record version + sha256.
3. **Parse (M1)**: `bgc parse` — GenBank+JSON → bgcs.parquet + proteins.faa;
   handle multi-locus entries, missing translations (translate from DNA),
   retired entries; unit tests on 2–3 fixture BGCs.
4. **EDA (M1)**: notebook — class balance, taxonomy, genes/BGC, protein
   lengths; informs pooling and filter choices.
5. **Baseline embedders (M2)**: class-composition, then Pfam pipeline
   (pyhmmer + Pfam-A) → TF-IDF embedder.
6. **Eval harness v1 (M2)**: labels, kNN accuracy, silhouette, UMAP plots;
   `bgc evaluate --embedding X` → metrics table. Verify Pfam TF-IDF beats
   the trivial baseline (harness sanity check).
7. **ESM embeddings (M3)**: per-protein embedding cache (chunked npz/hdf5,
   resumable); mean-pool → BGC vectors; run harness; compare. Start with the
   35M model end-to-end, then swap in 650M.
8. **Pooling ablations (M3/M6)**: weighted / core-only pooling from the same
   protein cache.
9. **Eval deep-dive (M4)**: retrieval metrics, RDKit Tanimoto correlation,
   taxonomy-controlled analysis, optional BiG-SCAPE GCF run; final benchmark
   table + figures.
10. **Retrieval (M5)**: FAISS index (exact IndexFlatIP on 3k vectors,
    cosine/normalized) + `bgc query cluster.gbk --top-k 10` — parse → embed →
    search → table with class, organism, compounds, similarity.
11. **Extensions (M6)**: contrastive aggregation, ESM-C/ProtT5, web UI
    (streamlit), scale-up to antiSMASH-DB queries.

## Reproducibility practices

- Pinned MiBIG release + checksums; raw data immutable.
- All parameters in YAML configs; runs are `cli + config`, no notebook state.
- Seeds fixed everywhere (numpy, torch, umap).
- Embedding sidecar metadata: model id, config hash, git commit, date.
- src-layout package + pytest + ruff + pre-commit; notebooks for exploration
  only.
