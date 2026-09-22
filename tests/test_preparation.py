"""Dependency-free tests for sequence coverage and biological data leakage."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from bgc_aggregation.__main__ import build_parser
from bgc_aggregation.chunking import make_chunks, make_views, position_features
from bgc_aggregation.common import read_fasta, read_jsonl, verify_dataset
from bgc_aggregation.prepare import classify, prepare, split_parents


class ChunkTests(unittest.TestCase):
    def test_every_residue_is_counted_once(self):
        for length in (1, 127, 512, 513, 2046, 18447):
            for size in (128, 256, 512, 768):
                for offset in (0, size // 3):
                    chunks = make_chunks(length, size, 64, offset)
                    coverage = [0] * length
                    for chunk in chunks:
                        self.assertLessEqual(chunk["context_end"] - chunk["context_start"], 2046)
                        self.assertLessEqual(chunk["context_start"], chunk["start"])
                        self.assertGreaterEqual(chunk["context_end"], chunk["end"])
                        for index in range(chunk["start"], chunk["end"]):
                            coverage[index] += 1
                    self.assertEqual(set(coverage), {1})
                    features = position_features(chunks, length)
                    self.assertEqual(features[0][3], 1.0)
                    self.assertEqual(features[-1][4], 1.0)
                    self.assertEqual(len(features[0]), 5)
        self.assertEqual(len(make_chunks(18447)), 37)

    def test_views_are_reproducible_and_include_deployment(self):
        views = make_views(1800, "protein", 42)
        self.assertEqual(views, make_views(1800, "protein", 42))
        self.assertEqual(views[0]["chunks"], make_chunks(1800))
        self.assertGreater(max(len(v["chunks"]) for v in views), len(views[0]["chunks"]))
        self.assertEqual(len(make_views(20, "tiny")), 1)

    def test_invalid_chunk_budgets_fail(self):
        for args in ((0, 512, 64, 0), (1000, 2046, 64, 0), (1000, 512, -1, 0), (1000, 512, 64, 512)):
            with self.assertRaises(ValueError):
                make_chunks(*args)


class PreparationTests(unittest.TestCase):
    def test_classification_uses_cds_rule_hits(self):
        nrps = "biosynthetic (rule-based-clusters) NRPS: AMP-binding"
        pks = "biosynthetic (rule-based-clusters) T1PKS: mod_KS"
        self.assertEqual(classify({"gene_functions": nrps})[0], "NRPS")
        self.assertEqual(classify({"gene_functions": pks})[0], "PKS")
        self.assertEqual(classify({"gene_functions": nrps + " | " + pks})[0], "HYBRID")
        self.assertEqual(classify({"product": "NRPS in a PKS cluster"})[0], "OTHER")
        self.assertEqual(classify({"gene_functions": "biosynthetic-additional (rule-based-clusters) NRPS: A"})[0], "OTHER")
        self.assertEqual(classify({"gene_functions": "biosynthetic (rule-based-clusters) T3PKS: Chal_sti_synt_N"})[0], "OTHER_PKS")

    def test_transitive_bgc_sequence_and_homology_groups(self):
        parents = [{"parent_id": f"p{i}", "bgc_id": f"b{i}", "sequence": "A" * (i + 1)} for i in range(12)]
        parents[1]["bgc_id"] = parents[0]["bgc_id"]
        parents[2]["sequence"] = parents[1]["sequence"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "clusters.tsv"
            path.write_text("".join(f"{'p2' if i == 3 else 'p' + str(i)}\tp{i}\n" for i in range(12)))
            split_parents(parents, path)
            self.assertEqual(len({p["group_id"] for p in parents[:4]}), 1)
            self.assertEqual(len({p["split"] for p in parents[:4]}), 1)
            self.assertEqual({p["split"] for p in parents}, {"train", "validation", "test"})
            path.write_text("p0\tp0\n")
            with self.assertRaisesRegex(ValueError, "cover every"):
                split_parents(parents, path)

    def test_fasta_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "protein.faa"
            path.write_text(">p1 description\nacdx\nuz*\n")
            self.assertEqual(read_fasta(path), {"p1": "ACDXUZ"})
            for contents in (">p1\nAA*AA\n", ">p1\nAA-AA\n", ">p1\nAA\n>p1\nCC\n", "AAA\n", ">p1\n"):
                path.write_text(contents)
                with self.assertRaises(ValueError):
                    read_fasta(path)

    def test_prepare_end_to_end_and_dataset_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fasta, annotations = root / "proteins.faa", root / "annotations.tsv"
            fasta.write_text("".join(f">p{i}\n" + "ACDEFGHIKLMNPQRSTVWY" * (30 + i if i < 12 else 130 + i) + "\n"
                                     for i in range(24)))
            with annotations.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sequence_id", "bgc_id", "gene_functions"], delimiter="\t")
                writer.writeheader()
                for i in range(24):
                    writer.writerow({"sequence_id": f"p{i}", "bgc_id": f"BGC{i}",
                                     "gene_functions": "biosynthetic (rule-based-clusters) NRPS: AMP-binding"})
            args = build_parser().parse_args(["prepare", "--fasta", str(fasta), "--annotations", str(annotations),
                                             "--output", str(root / "prepared")])
            prepare(args)
            metadata = verify_dataset(args.output)
            self.assertFalse(metadata["homology_clustered"])
            parents = {p["parent_id"]: p for p in read_jsonl(args.output / "parents.jsonl")}
            samples = read_jsonl(args.output / "samples.jsonl")
            sources = defaultdict(set)
            for sample in samples:
                parent = parents[sample["parent_id"]]
                self.assertEqual(sample["split"], parent["split"])
                self.assertEqual(sample["group_id"], parent["group_id"])
                self.assertEqual(sample["sequence"], parent["sequence"][sample["crop_start"]:sample["crop_end"]])
                self.assertLessEqual(sample["length"], 2046)
                sources[sample["source"]].add(sample["split"])
            self.assertIn("crop", sources)
            with self.assertRaisesRegex(ValueError, "empty directory"):
                prepare(args)
            with (args.output / "samples.jsonl").open("a") as handle:
                handle.write(json.dumps({"unexpected": True}) + "\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_dataset(args.output)

    def test_cli_help_does_not_import_torch_or_esm(self):
        for command in ([], ["train"], ["cache"], ["embed"]):
            completed = subprocess.run([sys.executable, "-m", "bgc_aggregation", *command, "--help"],
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
