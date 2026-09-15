#!/usr/bin/env python3
"""Report counts and lengths of extracted core CDS and protein sequences.

Usage: python script/core_gene_stats.py [--input-dir PATH]
Requires Biopython (available in the BGC conda environment).
Lengths count FASTA sequence symbols, excluding headers and whitespace.
Nucleotide and protein records are counted separately because some CDS
features may lack a protein translation. Empty files report N/A lengths.
"""

import argparse
from pathlib import Path

try:
    from Bio import SeqIO
except ImportError as exc:
    raise SystemExit("Biopython is required. Activate the BGC conda environment.") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def summarize_fasta(path: Path) -> dict:
    """Summarize a FASTA in one pass and retain all IDs tied for maximum length."""
    count = 0
    total_length = 0
    max_length = None
    longest_ids = []
    with path.open(encoding="utf-8") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            length = len(record.seq)
            count += 1
            total_length += length
            if max_length is None or length > max_length:
                max_length = length
                longest_ids = [record.id]
            elif length == max_length:
                longest_ids.append(record.id)
    return {
        "count": count,
        "mean_length": total_length / count if count else None,
        "max_length": max_length,
        "longest_ids": longest_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path,
                        default=PROJECT_ROOT / "data/processed/core_genes",
                        help="Directory containing core_genes.fna and core_genes.faa")
    args = parser.parse_args()
    inputs = [("CDS", "bp", args.input_dir / "core_genes.fna"),
              ("Protein", "aa", args.input_dir / "core_genes.faa")]
    for _, _, path in inputs:
        if not path.is_file():
            parser.error(f"Missing input: {path}. Run extract_core_gene.py first "
                         "or specify --input-dir.")

    print("Sequence_type\tCount\tMean_length\tMax_length\tUnit")
    for sequence_type, unit, path in inputs:
        stats = summarize_fasta(path)
        mean = f"{stats['mean_length']:.2f}" if stats["count"] else "N/A"
        maximum = str(stats["max_length"]) if stats["count"] else "N/A"
        print(f"{sequence_type}\t{stats['count']}\t{mean}\t{maximum}\t{unit}")
        if stats["longest_ids"]:
            print(f"# Longest {sequence_type} ID(s): " + ", ".join(stats["longest_ids"]))


if __name__ == "__main__":
    main()
