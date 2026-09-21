#!/usr/bin/env python3
"""Extract CDS features with an exact gene_kind value of biosynthetic from MIBiG GBK files.

Dependency: python -m pip install biopython
Usage: python script/extract_core_gene.py
Custom paths: python script/extract_core_gene.py --input-dir PATH --output-dir PATH

Write core_genes.faa (existing protein translations), core_genes.fna (CDS nucleotide
sequences), core_genes.tsv (annotations), and no_core_files.tsv (a manual review
list of files without core labels).
Skip entries without core labels; overwrite outputs with the same names on each run.
TSV start/end coordinates are 1-based and inclusive; location_0based preserves
Biopython's full location representation (0-based, end-exclusive), including
compound locations and fuzzy boundaries.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
except ImportError as exc:
    raise SystemExit("Biopython is required. Install it with: python -m pip install biopython") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "sequence_id", "bgc_id", "record_id", "gene_index", "gene", "locus_tag",
    "protein_id", "start", "end", "strand", "location_0based", "gene_kind",
    "product", "gene_functions", "nt_length", "aa_length", "source_file",
]
NO_CORE_FIELDS = [
    "bgc_id", "source_file", "record_ids", "organism", "description",
    "mibig_labels", "total_cds", "additional_cds", "unlabeled_cds",
    "gene_kind_counts", "reason", "source_path",
]


def extract_core_genes(input_dir: Path, output_dir: Path) -> dict[str, int]:
    """Extract all core CDS in filename order; gene_index is the 1-based CDS index within each file."""
    gbk_files = sorted(input_dir.glob("*.gbk"))
    if not gbk_files:
        raise ValueError(f"Input directory does not exist or contains no .gbk files: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {"files": len(gbk_files), "files_with_core": 0,
              "core_genes": 0, "missing_translation": 0}

    with (
        (output_dir / "core_genes.faa").open("w", encoding="utf-8") as proteins,
        (output_dir / "core_genes.fna").open("w", encoding="utf-8") as nucleotides,
        (output_dir / "core_genes.tsv").open("w", encoding="utf-8", newline="") as table,
        (output_dir / "no_core_files.tsv").open("w", encoding="utf-8", newline="") as no_core_table,
    ):
        writer = csv.DictWriter(table, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        no_core_writer = csv.DictWriter(no_core_table, fieldnames=NO_CORE_FIELDS, delimiter="\t")
        no_core_writer.writeheader()

        for gbk_file in gbk_files:
            bgc_id = gbk_file.stem
            gene_index = 0
            file_core_count = 0
            record_ids = []
            organisms = set()
            descriptions = set()
            mibig_labels = set()
            kind_counts = Counter()
            unlabeled_cds = 0
            with gbk_file.open(encoding="utf-8") as source:
                for record in SeqIO.parse(source, "genbank"):
                    record_ids.append(record.id)
                    if record.annotations.get("organism"):
                        organisms.add(record.annotations["organism"])
                    if record.description:
                        descriptions.add(record.description)
                    for feature in record.features:
                        if (feature.type == "subregion"
                                and "mibig" in feature.qualifiers.get("aStool", [])):
                            mibig_labels.update(feature.qualifiers.get("label", []))
                        if feature.type != "CDS":
                            continue
                        gene_index += 1
                        qualifiers = feature.qualifiers
                        kinds = {kind for kind in qualifiers.get("gene_kind", []) if kind}
                        kind_counts.update(kinds)
                        if not kinds:
                            unlabeled_cds += 1
                        if "biosynthetic" not in qualifiers.get("gene_kind", []):
                            continue
                        if feature.location is None:
                            raise ValueError(f"CDS {gene_index} in {gbk_file.name} has no valid location")

                        gene = qualifiers.get("gene", [""])[0]
                        locus_tag = qualifiers.get("locus_tag", [""])[0]
                        protein_id = qualifiers.get("protein_id", [""])[0]
                        label = locus_tag or gene or protein_id or f"CDS_{gene_index}"
                        # FASTA identifiers cannot contain whitespace; the index distinguishes duplicate gene names.
                        label = "_".join(label.split())
                        sequence_id = f"{bgc_id}|{gene_index}|{label}"
                        product = qualifiers.get("product", [""])[0]
                        description = " ".join(product.split())

                        # extract handles complement and join automatically, preserving the CDS coding orientation.
                        dna = feature.extract(record.seq)
                        SeqIO.write(SeqRecord(dna, id=sequence_id, description=description),
                                    nucleotides, "fasta")
                        translation = "".join(qualifiers.get("translation", [""])[0].split())
                        if translation:
                            SeqIO.write(
                                SeqRecord(Seq(translation), id=sequence_id, description=description),
                                proteins, "fasta",
                            )
                        else:
                            # Do not infer translations, to avoid errors from incomplete CDS or special translation rules.
                            counts["missing_translation"] += 1

                        writer.writerow({
                            "sequence_id": sequence_id,
                            "bgc_id": bgc_id,
                            "record_id": record.id,
                            "gene_index": gene_index,
                            "gene": gene,
                            "locus_tag": locus_tag,
                            "protein_id": protein_id,
                            "start": int(feature.location.start) + 1,
                            "end": int(feature.location.end),
                            "strand": {1: "+", -1: "-"}.get(feature.location.strand, "."),
                            "location_0based": str(feature.location),
                            "gene_kind": "biosynthetic",
                            "product": product,
                            "gene_functions": " | ".join(qualifiers.get("gene_functions", [])),
                            "nt_length": len(dna),
                            "aa_length": len(translation) if translation else "",
                            "source_file": gbk_file.name,
                        })
                        file_core_count += 1
            counts["core_genes"] += file_core_count
            if file_core_count:
                counts["files_with_core"] += 1
            else:
                no_core_writer.writerow({
                    "bgc_id": bgc_id,
                    "source_file": gbk_file.name,
                    "record_ids": " | ".join(record_ids),
                    "organism": " | ".join(sorted(organisms)),
                    "description": " | ".join(sorted(descriptions)),
                    "mibig_labels": " | ".join(sorted(mibig_labels)),
                    "total_cds": gene_index,
                    "additional_cds": kind_counts["biosynthetic-additional"],
                    "unlabeled_cds": unlabeled_cds,
                    "gene_kind_counts": " | ".join(
                        f"{kind}:{count}" for kind, count in sorted(kind_counts.items())
                    ),
                    "reason": "no_biosynthetic_label",
                    "source_path": str(gbk_file.resolve()),
                })

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path,
                        default=PROJECT_ROOT / "data/raw/mibig_gbk_4.0",
                        help="Directory containing .gbk files (non-recursive search)")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data/processed/core_genes",
                        help="Output directory; existing files with the same names will be overwritten")
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    if not any(args.input_dir.glob("*.gbk")):
        parser.error(f"Input directory contains no .gbk files: {args.input_dir}")
    counts = extract_core_genes(args.input_dir, args.output_dir)
    print(f"GBK files scanned: {counts['files']}")
    print(f"Files with core labels: {counts['files_with_core']}")
    print(f"Files skipped without core labels: {counts['files'] - counts['files_with_core']}")
    print(f"Core CDS extracted: {counts['core_genes']}")
    print(f"Core CDS missing translation: {counts['missing_translation']} (nucleotide sequences and annotations only)")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Manual review list of files without core labels: {(args.output_dir / 'no_core_files.tsv').resolve()}")


if __name__ == "__main__":
    main()
