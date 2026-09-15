#!/usr/bin/env python3
"""提取 MIBiG GBK 中 gene_kind 精确为 biosynthetic 的 CDS。

依赖：python -m pip install biopython
运行：python script/extract_core_gene.py
自定义路径：python script/extract_core_gene.py --input-dir PATH --output-dir PATH

输出 core_genes.faa（已有蛋白翻译）、core_genes.fna（CDS 核酸）和
core_genes.tsv（注释）。跳过没有 core 标签的条目；每次运行覆盖同名输出。
TSV 的 start/end 是 1-based、两端包含的范围；location_0based 保留
Biopython 的完整位置表示（0-based、右端不包含），包括分段和模糊边界。
"""

import argparse
import csv
from pathlib import Path

try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
except ImportError as exc:
    raise SystemExit("缺少 Biopython，请先运行：python -m pip install biopython") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "sequence_id", "bgc_id", "record_id", "gene_index", "gene", "locus_tag",
    "protein_id", "start", "end", "strand", "location_0based", "gene_kind",
    "product", "gene_functions", "nt_length", "aa_length", "source_file",
]


def extract_core_genes(input_dir: Path, output_dir: Path) -> dict[str, int]:
    """按文件名排序提取全部 core CDS；gene_index 为文件内 CDS 的 1-based 序号。"""
    gbk_files = sorted(input_dir.glob("*.gbk"))
    if not gbk_files:
        raise ValueError(f"输入目录不存在或没有 .gbk 文件：{input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {"files": len(gbk_files), "files_with_core": 0,
              "core_genes": 0, "missing_translation": 0}

    with (
        (output_dir / "core_genes.faa").open("w", encoding="utf-8") as proteins,
        (output_dir / "core_genes.fna").open("w", encoding="utf-8") as nucleotides,
        (output_dir / "core_genes.tsv").open("w", encoding="utf-8", newline="") as table,
    ):
        writer = csv.DictWriter(table, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()

        for gbk_file in gbk_files:
            bgc_id = gbk_file.stem
            gene_index = 0
            file_core_count = 0
            with gbk_file.open(encoding="utf-8") as source:
                for record in SeqIO.parse(source, "genbank"):
                    for feature in record.features:
                        if feature.type != "CDS":
                            continue
                        gene_index += 1
                        qualifiers = feature.qualifiers
                        if "biosynthetic" not in qualifiers.get("gene_kind", []):
                            continue
                        if feature.location is None:
                            raise ValueError(f"{gbk_file.name} 的 CDS {gene_index} 缺少有效坐标")

                        gene = qualifiers.get("gene", [""])[0]
                        locus_tag = qualifiers.get("locus_tag", [""])[0]
                        protein_id = qualifiers.get("protein_id", [""])[0]
                        label = locus_tag or gene or protein_id or f"CDS_{gene_index}"
                        # FASTA 标识符不能含空白；序号可区分重复基因名称。
                        label = "_".join(label.split())
                        sequence_id = f"{bgc_id}|{gene_index}|{label}"
                        product = qualifiers.get("product", [""])[0]
                        description = " ".join(product.split())

                        # extract 自动处理 complement 和 join，保留 CDS 的编码方向。
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
                            # 不自行推断翻译，避免不完整 CDS 或特殊翻译规则造成错误。
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

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path,
                        default=PROJECT_ROOT / "data/raw/mibig_gbk_4.0",
                        help="包含 .gbk 文件的目录（不递归搜索）")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data/processed/core_genes",
                        help="结果目录，同名结果会被覆盖")
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f"输入目录不存在：{args.input_dir}")
    if not any(args.input_dir.glob("*.gbk")):
        parser.error(f"输入目录没有 .gbk 文件：{args.input_dir}")
    counts = extract_core_genes(args.input_dir, args.output_dir)
    print(f"扫描 GBK 文件：{counts['files']}")
    print(f"含 core 标签的文件：{counts['files_with_core']}")
    print(f"跳过无 core 标签的文件：{counts['files'] - counts['files_with_core']}")
    print(f"提取 core CDS：{counts['core_genes']}")
    print(f"缺少 translation 的 core CDS：{counts['missing_translation']}（仅输出核酸和注释）")
    print(f"输出目录：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
