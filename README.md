# BGC

A project for biosynthetic gene cluster (BGC) analysis and retrieval using MIBiG data.
The current implementation extracts core biosynthetic genes from annotated GenBank files.
See [DESIGN.md](DESIGN.md) for the proposed architecture and research roadmap.

## Getting started

Clone the repository and enter the project directory:

```bash
git clone git@github.com:QJ-ChenR/BGC.git
cd BGC
```

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate BGC
```

If the `BGC` environment already exists, activate it with `conda activate BGC`.
The environment includes Python 3.12 and Biopython, which is imported in Python as `Bio`.

## Input data

Place the MIBiG 4.0 GenBank files in `data/raw/mibig_gbk_4.0/`, with one `.gbk` file per entry.
The script searches this directory directly, without recursing into subdirectories.
Raw data and generated outputs are stored under `data/` and excluded from Git;
each collaborator needs to prepare the input data locally.

## Extracting core genes

Run the following command from the project root:

```bash
python script/extract_core_gene.py
```

To use custom input and output directories:

```bash
python script/extract_core_gene.py --input-dir /path/to/gbk --output-dir /path/to/output
```

The script selects coding sequence (`CDS`) features whose `gene_kind` qualifier
contains the exact value `biosynthetic`. Files without this label are skipped.
Features labeled `biosynthetic-additional` are excluded.
This selection follows the existing antiSMASH core classification and does not
necessarily capture every experimentally established essential biosynthetic gene.
A missing core label does not establish that a cluster lacks core biosynthetic genes.

### Outputs

Results are written to `data/processed/core_genes/` by default:

| File | Contents |
| --- | --- |
| `core_genes.faa` | Protein sequences from existing CDS translation annotations |
| `core_genes.fna` | CDS nucleotide sequences in coding orientation, supporting reverse strands and joined locations |
| `core_genes.tsv` | BGC and gene identifiers, coordinates, functional annotations, and sequence lengths |

FASTA identifiers use the format `BGC_ID|CDS_index|gene_identifier`.
The CDS index is 1-based within each input file and counts all CDS features,
including those excluded from the output. The gene identifier uses `locus_tag`,
then `gene`, then `protein_id`, falling back to `CDS_<index>` if none is available.

Core CDS features without a translation annotation are included only in the
nucleotide FASTA and annotation table; their count is reported in the run summary.
Protein translations are not inferred when the annotation is missing.

In the TSV, `start` and `end` describe the 1-based, inclusive bounding range.
The `location_0based` field preserves the full Biopython location representation,
including strand, joined segments, and fuzzy boundaries, using 0-based,
end-exclusive coordinates.

Rerunning the script overwrites files with the same names in the output directory.

## Validation

The script was validated on a local MIBiG 4.0 dataset:

| Metric | Count |
| --- | ---: |
| Input GBK files | 2,636 |
| Files containing core labels | 2,269 |
| Extracted core CDS features | 6,831 |
| Files skipped because no core label was present | 367 |
| Selected CDS features missing a translation | 0 |

Validation checked identifier and sequence-length consistency across the three
outputs, selection of `abyB1`, `abyB2`, and `abyB3` from `BGC0000001`,
and correct reverse-complement extraction for a reverse-strand CDS.
