"""Protocol boundaries, group isolation, and statistical interpretation tests."""

import csv
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from bgc_aggregation.__main__ import build_parser
from bgc_aggregation.chunking import make_views
from bgc_aggregation.common import read_jsonl, verify_dataset
from bgc_aggregation.length_evaluation import length_report, paired_group_bootstrap, write_length_table
from bgc_aggregation.prepare import prepare
from bgc_aggregation.protocol import validate_length_entries


class LengthPreparationTests(unittest.TestCase):
    def inputs(self, root):
        fasta, annotations = root / "proteins.faa", root / "annotations.tsv"
        lengths = (800, 1499, 1500, 1536, 1537, 1792, 1793, 2046, 2047)
        with fasta.open("w") as fa, annotations.open("w", newline="") as tsv:
            writer = csv.writer(tsv, delimiter="\t")
            writer.writerow(["sequence_id", "bgc_id", "gene_functions"])
            for group in range(20):
                for number, length in enumerate(lengths):
                    index = group * len(lengths) + number
                    prefix = f"{index:08b}".translate(str.maketrans("01", "AC"))
                    fa.write(f">p{index}\n{prefix}{'D' * (length - len(prefix))}\n")
                    family = "NRPS" if group % 2 else "T1PKS"
                    writer.writerow([f"p{index}", f"BGC{group}",
                                     f"biosynthetic (rule-based-clusters) {family}: domain"])
        return fasta, annotations

    def args(self, root, output="prepared", extra=()):
        fasta, annotations = self.inputs(root)
        return build_parser().parse_args([
            "prepare", "--fasta", str(fasta), "--annotations", str(annotations),
            "--protocol", "length-extrapolation", "--output", str(root / output), *extra])

    def test_group_first_cutoff_before_views_and_native_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.args(root)
            prepare(args)
            metadata = verify_dataset(args.output)
            parents = read_jsonl(args.output / "parents.jsonl")
            samples = read_jsonl(args.output / "samples.jsonl")
            self.assertEqual(metadata["experiment"]["chunk_views"], "fixed")
            self.assertEqual(metadata["crops_per_parent"], 0)
            self.assertEqual(set(metadata["summary"]["length_cohorts"]),
                             {"train", "validation", "test_short", "test_long"})
            for sample in samples:
                self.assertEqual(sample["source"], "native")
                self.assertEqual(len(sample["views"]), 1)
                self.assertEqual(sample["views"][0]["offset"], 0)
                expected = (sample["length"] + 511) // 512
                self.assertEqual(len(sample["views"][0]["chunks"]), expected)
                if sample["split"] != "test":
                    self.assertLess(sample["length"], 1500)
                    self.assertLessEqual(expected, 3)
            test_lengths = {s["length"] for s in samples if s["split"] == "test"}
            self.assertTrue({1499, 1500, 1536, 1537, 1792, 1793, 2046} <= test_lengths)
            self.assertNotIn(2047, test_lengths)
            by_group = {}
            for parent in parents:
                self.assertEqual(by_group.setdefault(parent["group_id"], parent["split"]), parent["split"])
            self.assertTrue(any(p["experiment_role"] == "excluded_long_train" for p in parents))
            self.assertTrue(any(p["experiment_role"] == "excluded_above_teacher_limit" for p in parents))
            # Changing view augmentation must not change split or cohort membership.
            augmented = self.args(root, "augmented", ["--chunk-views", "multiscale"])
            prepare(augmented)
            augmented_samples = read_jsonl(augmented.output / "samples.jsonl")
            self.assertEqual([(s["sample_id"], s["split"]) for s in samples],
                             [(s["sample_id"], s["split"]) for s in augmented_samples])
            self.assertTrue(any(len(s["views"]) > 1 for s in augmented_samples))

    def test_invalid_protocol_options_and_empty_long_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for extra in (["--crops-per-parent", "1"], ["--length-cutoff", "2047"]):
                with self.assertRaises(ValueError):
                    prepare(self.args(root, extra=extra))
            args = self.args(root)
            prepare(args)
            metadata = verify_dataset(args.output)
            samples = read_jsonl(args.output / "samples.jsonl")
            samples = [s for s in samples if s["length"] < 1500]
            with self.assertRaisesRegex(ValueError, "test_long"):
                validate_length_entries(samples, metadata["experiment"])

    def test_cache_contract_rejects_crops_long_training_and_group_leakage(self):
        experiment = {"length_cutoff": 1500}
        entries = [{"parent_id": str(i), "group_id": str(i), "bgc_id": str(i), "family": "NRPS",
                    "source": "native", "length": length, "split": split}
                   for i, (split, length) in enumerate([
                       ("train", 1499), ("validation", 1499), ("test", 1499), ("test", 1500)])]
        validate_length_entries(entries, experiment)
        for field, value, message in (("length", 1500, "shorter"), ("source", "crop", "natural")):
            changed = deepcopy(entries)
            changed[0][field] = value
            with self.assertRaisesRegex(ValueError, message):
                validate_length_entries(changed, experiment)
        for field in ("group_id", "bgc_id"):
            changed = deepcopy(entries)
            changed[-1][field] = changed[0][field]
            with self.assertRaisesRegex(ValueError, "leakage"):
                validate_length_entries(changed, experiment)


class LengthStatisticsTests(unittest.TestCase):
    def row(self, identifier, group, length=1800, model=1.0, baseline=2.0, family="NRPS"):
        return {"parent_id": identifier, "group_id": group, "length": length,
                "chunks": (length + 511) // 512, "family": family, "view": 0,
                "model_raw_mse": model, "baseline_raw_mse": baseline,
                "model_cosine_distance": 0.01, "baseline_cosine_distance": 0.02}

    def test_bootstrap_is_paired_grouped_reproducible_and_handles_zero_baseline(self):
        rows = [self.row("a", "g1"), self.row("b", "g1"), self.row("c", "g2")]
        result = paired_group_bootstrap(rows, 100, 42)
        self.assertEqual(result, paired_group_bootstrap(rows, 100, 42))
        self.assertEqual(result["groups"], 2)
        self.assertEqual(result["raw_mse_difference_ci95"], [-1.0, -1.0])
        self.assertEqual(result["relative_mse_reduction_ci95"], [0.5, 0.5])
        self.assertEqual(result["evidence"], "improved")
        self.assertEqual(paired_group_bootstrap(rows[:2], 100)["evidence"], "insufficient_groups")
        worse = [dict(row, model_raw_mse=3.0) for row in rows]
        self.assertEqual(paired_group_bootstrap(worse, 100)["evidence"], "worse")
        zeros = [dict(row, model_raw_mse=0.0, baseline_raw_mse=0.0) for row in rows]
        zero_result = paired_group_bootstrap(zeros, 100)
        self.assertIsNone(zero_result["relative_mse_reduction"])
        self.assertIsNone(zero_result["relative_mse_reduction_ci95"])
        self.assertEqual(zero_result["evidence"], "inconclusive")
        self.assertIsNone(paired_group_bootstrap([], 100))

    def test_bootstrap_retains_protein_weighting_when_group_sizes_differ(self):
        rows = [self.row(str(i), "big", model=0.0, baseline=1.0) for i in range(9)]
        rows.append(self.row("small", "small", model=10.0, baseline=1.0))
        result = paired_group_bootstrap(rows, 300, 7)
        self.assertEqual(result["raw_mse_difference"], 0.0)
        self.assertEqual(result["raw_mse_difference_ci95"], [-1.0, 9.0])
        self.assertEqual(result["evidence"], "inconclusive")

    def test_cohorts_bins_families_and_augmented_views_do_not_change_primary_result(self):
        lengths = (1023, 1024, 1499, 1500, 1536, 1537, 1792, 1793, 2046)
        rows = [self.row(str(i), str(i), length, family="NRPS" if i % 2 else "PKS")
                for i, length in enumerate(lengths)]
        experiment = {"protocol": "length-extrapolation", "length_cutoff": 1500}
        training = {"max_chunks": 3, "views": "fixed"}
        report, cohorts = length_report(rows, experiment, training, "test", 50)
        self.assertEqual(len(cohorts["short"]), 3)
        self.assertEqual(len(cohorts["long"]), 6)
        self.assertEqual(len(cohorts["long_within_chunk_range"]), 2)
        self.assertEqual(len(cohorts["long_beyond_chunk_range"]), 4)
        self.assertEqual(report["length_bins"]["1024-1499"]["overall"]["proteins"], 2)
        self.assertEqual(report["length_bins"]["1500-1536"]["overall"]["proteins"], 2)
        augmented = rows + [dict(r, view=1, chunks=20, model_raw_mse=1000.0) for r in rows]
        self.assertEqual(report, length_report(augmented, experiment, training, "test", 50)[0])
        with tempfile.TemporaryDirectory() as temporary:
            write_length_table(report, temporary)
            with (Path(temporary) / "length_summary.tsv").open() as handle:
                table = list(csv.DictReader(handle, delimiter="\t"))
            self.assertTrue(any(r["cohort"] == "long" and r["family"] == "PKS" for r in table))
        # Multiscale training can already have seen more than four chunks.
        _, augmented_cohorts = length_report(rows, experiment, {"max_chunks": 12, "views": "multiscale"}, "test", 50)
        self.assertFalse(augmented_cohorts["long_beyond_chunk_range"])

    def test_fixed_view_chunk_boundary(self):
        self.assertEqual(len(make_views(1499, "a", mode="fixed")[0]["chunks"]), 3)
        self.assertEqual(len(make_views(1536, "b", mode="fixed")[0]["chunks"]), 3)
        self.assertEqual(len(make_views(1537, "c", mode="fixed")[0]["chunks"]), 4)


if __name__ == "__main__":
    unittest.main()
