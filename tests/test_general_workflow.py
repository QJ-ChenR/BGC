"""General TSV import, scoped isolation, runner budgets, and CPU workflow checks."""

import csv
import hashlib
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from bgc_aggregation.__main__ import build_parser
from bgc_aggregation.common import read_json, read_jsonl, verify_dataset
from bgc_aggregation.general_prepare import TSV_FIELDS, prepare_general, read_general_table, split_short_pool
from bgc_aggregation.protocol import validate_length_entries
from script import run_general_length_extrapolation as runner


def inputs(root):
    paths = (root / "short.tsv", root / "long.tsv")
    for path, lengths in zip(paths, ((500, 800, 1499) * 20, (1500, 1536, 1537, 2000))):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, TSV_FIELDS, delimiter="\t")
            writer.writeheader()
            for i, length in enumerate(lengths):
                prefix = f"{i:012b}".translate(str.maketrans("01", "AC"))
                sequence = prefix + "D" * (length - len(prefix))
                if i == 0:
                    sequence = sequence[:-1] + "U"
                species = i // 2 + 1 if path == paths[0] else (1, 2, 999, 1000)[i]
                writer.writerow({
                    "accession": f"{path.stem}{i}", "group": ("Bacteria", "Fungi", "Other")[i % 3],
                    "other_subgroup": "Animals" if i % 3 == 2 else "",
                    "organism": f"Organism {species}", "scientific_name": f"Species {species}",
                    "organism_taxon_id": str(species), "species_key_taxon_id": str(species),
                    "genus_taxon_id": str(species + 100), "length_aa": length, "reviewed": "false",
                    "protein_existence": "Inferred from homology", "protein_name": "example",
                    "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                    "uniprot_url": "https://www.uniprot.org/uniprotkb/example", "sequence": sequence})
    return paths


def prepared(root):
    short, long = inputs(root)
    args = build_parser().parse_args(["prepare-general", "--short-tsv", str(short),
                                      "--long-tsv", str(long), "--output", str(root / "prepared")])
    prepare_general(args)
    return args


class GeneralPreparationTests(unittest.TestCase):
    def setUp(self):
        output = redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def test_fixed_long_test_species_isolation_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = prepared(root)
            metadata = verify_dataset(args.output)
            rows = read_jsonl(args.output / "samples.jsonl")
            self.assertEqual(metadata["summary"]["short_pool"], 60)
            self.assertEqual(metadata["summary"]["long_test"], 4)
            self.assertFalse(metadata["homology_clustered"])
            self.assertEqual({r["split"] for r in rows if r["provided_split"] == "long_test"}, {"test"})
            short = [r for r in rows if r["provided_split"] == "short_pool"]
            species = {}
            for row in short:
                self.assertEqual(species.setdefault(row["group_id"], row["split"]), row["split"])
                self.assertEqual(row["group_id"], "species:" + row["species_key_taxon_id"])
                self.assertIsNone(row["bgc_id"])
                self.assertEqual(row["family"], "GENERAL")
            self.assertEqual({r["split"] for r in short}, {"train", "validation", "test"})
            long_shared = next(r for r in rows if r["parent_id"] == "long0")
            self.assertIn(long_shared["group_id"], species)
            original = {r["parent_id"]: r["split"] for r in short}
            split_short_pool(list(reversed(short)), seed=42)
            self.assertEqual(original, {r["parent_id"]: r["split"] for r in short})
            self.assertTrue(all(len(r["views"]) == 1 for r in rows))
            self.assertEqual([len(r["views"][0]["chunks"]) for r in rows[-4:]], [3, 3, 4, 4])
            self.assertIn("U", rows[0]["sequence"])
            with self.assertRaisesRegex(ValueError, "empty directory"):
                prepare_general(args)
            args.output = root / "repeat"
            prepare_general(args)
            self.assertEqual(metadata["dataset_id"], verify_dataset(args.output)["dataset_id"])

    def test_protocol_accepts_only_the_intended_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = prepared(Path(temporary))
            experiment = verify_dataset(args.output)["experiment"]
            rows = read_jsonl(args.output / "samples.jsonl")
            training = next(r for r in rows if r["split"] == "train")
            validation = next(r for r in rows if r["split"] == "validation")
            changed = deepcopy(rows)
            long = changed[-1]
            long.update(species_key_taxon_id=training["species_key_taxon_id"], group_id=training["group_id"])
            validate_length_entries(changed, experiment)  # Explicitly allowed.
            for mutate, message in (
                (lambda r: r.update(split="train") if r["provided_split"] == "long_test" else None, "stay in test"),
                (lambda r: r.update(group_id="fake") if r["parent_id"] == training["parent_id"] else None, "original species"),
                (lambda r: r.update(species_key_taxon_id=training["species_key_taxon_id"], group_id=training["group_id"])
                 if r["parent_id"] == validation["parent_id"] else None, "leakage"),
                (lambda r: r.update(sequence_sha256=training["sequence_sha256"])
                 if r["parent_id"] == validation["parent_id"] else None, "Duplicate"),
            ):
                corrupted = deepcopy(rows)
                for row in corrupted:
                    mutate(row)
                with self.assertRaisesRegex(ValueError, message):
                    validate_length_entries(corrupted, experiment)
            changed = deepcopy(rows)
            changed[0].pop("provided_split")
            with self.assertRaisesRegex(ValueError, "provenance"):
                validate_length_entries(changed, experiment)

    def test_tsv_validation_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            short, long = inputs(root)
            with short.open() as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            for field, value, message in (("length_aa", "700", "length mismatch"),
                                          ("sequence_sha256", "x", "sha256"),
                                          ("species_key_taxon_id", "", "species_key"),
                                          ("group", "NRPS", "taxonomy"),
                                          ("sequence", "A-" * 250, "Invalid protein")):
                path = root / "bad.tsv"
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, TSV_FIELDS, delimiter="\t")
                    writer.writeheader()
                    writer.writerow({**rows[0], field: value})
                with self.assertRaisesRegex(ValueError, message):
                    read_general_table(path, "short_pool")
            with short.open("a", newline="") as handle:
                csv.DictWriter(handle, TSV_FIELDS, delimiter="\t").writerow(rows[0])
            args = build_parser().parse_args(["prepare-general", "--short-tsv", str(short),
                                              "--long-tsv", str(long), "--output", str(root / "bad")])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                prepare_general(args)
            self.assertFalse(args.output.exists())


class GeneralRunnerTests(unittest.TestCase):
    def test_all_methods_precede_test_and_budget_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "data/embeddings/general_length1500"
            cache_path.mkdir(parents=True)
            from bgc_aggregation.common import write_json
            cache = {"complete": True, "cache_id": "general", "encoder": {"embedding_dim": 1152},
                     "experiment": {"protocol": runner.GENERAL_PROTOCOL}}
            write_json(cache_path / "cache.json", cache)
            args = runner.build_parser().parse_args(["--no-plots", "--skip-neighbors"])
            with patch.object(runner.shared, "PROJECT_ROOT", root), patch.object(runner.shared, "run_logged") as logged:
                runner.run(args)
                commands = [build_parser().parse_args(c.args[0][4:]) for c in logged.call_args_list]
                self.assertEqual([a.command for a in commands], ["train"] * 9 + ["evaluate"] * 9)
                self.assertEqual({a.architecture for a in commands[:9]}, {"transformer", "mlp", "lstm"})
                self.assertTrue(all(a.scope == "all" for a in commands))
                self.assertTrue(all(a.max_steps == 10000 and a.epochs == 20 for a in commands[:9]))
                self.assertTrue(all(a.skip_neighbors for a in commands[9:]))
                args.seeds = [42, 42]
                with self.assertRaisesRegex(ValueError, "only once"):
                    runner.run(args)

    def test_resume_guard_includes_step_budget_and_rejects_legacy_cache(self):
        from bgc_aggregation.common import write_json
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "run"
            directory.mkdir()
            cache = {"cache_id": "general", "encoder": {"embedding_dim": 1152},
                     "experiment": {"protocol": runner.GENERAL_PROTOCOL}}
            training = {**runner.shared.TRAINING, "scope": "all", "epochs": 20, "patience": 5, "max_steps": 10000}
            metadata = {"cache_id": "general", "experiment": cache["experiment"], "initial_checkpoint": None,
                        "training": {**training, "seed": 42},
                        "model": {"architecture": "lstm", "embedding_dim": 1152, "hidden_dim": 128,
                                  "num_layers": 2, "num_heads": 4, "feedforward_dim": 512,
                                  "dropout": 0.1, "use_positions": False}}
            write_json(directory / "run.json", metadata)
            for name in ("best.pt", "last.pt"):
                (directory / name).touch()
            self.assertTrue(runner.shared.matching_run(directory, cache, "lstm", 42, training))
            with self.assertRaisesRegex(ValueError, "settings differ"):
                runner.shared.matching_run(directory, cache, "lstm", 42, {**training, "max_steps": 20000})
            write_json(root / "cache.json", {**cache, "complete": True,
                       "experiment": {"protocol": "length-extrapolation"}})
            args = runner.build_parser().parse_args(["--cache", str(root), "--no-plots"])
            with patch.object(runner.shared, "run_logged") as logged:
                with self.assertRaisesRegex(ValueError, "general-length"):
                    runner.run(args)
                logged.assert_not_called()


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "Requires the server/test PyTorch environment")
class GeneralRuntimeTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        output = redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def make_cache(self, root):
        import torch
        from bgc_aggregation.cache import extract_cache
        class FakeEncoder:
            identity = {"embedding_dim": 8, "model": "fake"}
            def __init__(self, *args):
                pass
            def pool(self, requests):
                return [torch.tensor([float(n) / 255 for n in hashlib.sha256(seq.encode()).digest()[:8]])
                        for seq, start, end in requests]
        args = prepared(root)
        cache_args = build_parser().parse_args(["cache", "--dataset", str(args.output),
                                                "--output", str(root / "cache"), "--device", "cpu"])
        with patch("bgc_aggregation.encoder.ESMCEncoder", FakeEncoder):
            extract_cache(cache_args)
        return root / "cache"

    def test_general_cache_training_resume_and_reports_all_architectures(self):
        import torch
        from bgc_aggregation.cache import load_cache, save_tensor_file
        from bgc_aggregation.data import teacher_statistics
        from bgc_aggregation.evaluate import evaluate
        from bgc_aggregation.train import train
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = self.make_cache(root)
            cache = load_cache(cache_path)
            self.assertTrue(all("species_key_taxon_id" in e for e in cache["entries"]))
            self.assertEqual(len(cache["entries"]), 64)
            expected_stats = teacher_statistics(cache_path, [e for e in cache["entries"] if e["split"] == "train"])
            for architecture in ("mlp", "lstm", "transformer"):
                with self.subTest(architecture=architecture):
                    def training_args(name):
                        return build_parser().parse_args([
                            "train", "--cache", str(cache_path), "--output", str(root / name),
                            "--scope", "all", "--architecture", architecture, "--device", "cpu",
                            "--hidden-dim", "16", "--feedforward-dim", "32", "--batch-size", "8",
                            "--epochs", "20", "--max-steps", "8"])
                    full_args = training_args(architecture)
                    train(full_args)
                    full = torch.load(full_args.output / "last.pt", weights_only=True)
                    self.assertEqual(full["global_step"], 8)
                    self.assertEqual(full["training_config"]["validation_scope"], "all")
                    self.assertEqual(full["training_inputs"]["max_chunks"], 3)
                    self.assertLess(full["validation_inputs"]["max_length"], 1500)
                    torch.testing.assert_close(full["teacher_statistics"]["mean"], expected_stats["mean"])
                    args = training_args(architecture + "_resume")
                    class Interrupted(Exception):
                        pass
                    def interrupt(path, value):
                        save_tensor_file(path, value)
                        if Path(path).name == "last.pt" and value["epoch"] == 1:
                            raise Interrupted()
                    with patch("bgc_aggregation.train.save_tensor_file", side_effect=interrupt):
                        with self.assertRaises(Interrupted):
                            train(args)
                    args.resume = args.output / "last.pt"
                    args.max_steps = 999  # Resume must keep the saved cap.
                    train(args)
                    resumed = torch.load(args.resume, weights_only=True)
                    self.assertEqual(resumed["global_step"], 8)
                    self.assertEqual(full["history"], resumed["history"])
                    for key in full["model_state"]:
                        torch.testing.assert_close(full["model_state"][key], resumed["model_state"][key], atol=0, rtol=0)
                    evaluation = build_parser().parse_args([
                        "evaluate", "--cache", str(cache_path), "--checkpoint", str(full_args.output / "best.pt"),
                        "--scope", "all", "--device", "cpu", "--bootstrap-replicates", "20",
                        "--output", str(root / (architecture + "_report"))])
                    if importlib.util.find_spec("matplotlib"):
                        evaluation.plots = True
                    evaluate(evaluation)
                    report = read_json(evaluation.output / "metrics.json")
                    detail = report["length_extrapolation"]
                    self.assertEqual(detail["cohorts"]["long"]["overall"]["proteins"], 4)
                    self.assertEqual(detail["cohorts"]["long_beyond_chunk_range"]["overall"]["proteins"], 2)
                    self.assertEqual(detail["stratification_field"], "taxonomy_group")
                    self.assertIn("500-512", detail["length_bins"])
                    self.assertIn("1793-2000", detail["length_bins"])
                    self.assertIn("Bacteria", detail["cohorts"]["long"]["by_taxonomy_group"])
                    self.assertEqual(sum(detail["cohorts"][name]["overall"]["proteins"]
                                         if detail["cohorts"][name]["overall"] else 0 for name in
                                         ("long_species_seen_in_training", "long_species_unseen_in_training")), 4)
                    self.assertIn("taxonomy_group", (evaluation.output / "length_summary.tsv").read_text().splitlines()[0])
                    self.assertIn("species_seen_in_training", (evaluation.output / "per_sample.tsv").read_text().splitlines()[0])
                    evaluation.skip_neighbors = True
                    evaluation.split = "validation"
                    evaluate(evaluation)
                    report = read_json(evaluation.output / "metrics.json")
                    self.assertTrue(report["neighbors_skipped"])
                    self.assertIsNone(report["neighborhood_retention"])
                    self.assertIsNone(report["length_extrapolation"]["cohorts"]["long"]["overall"])
            args = training_args("invalid")
            args.scope = "target"
            with self.assertRaisesRegex(ValueError, "scope all"):
                train(args)

    def test_numpy_bootstrap_is_paired_weighted_and_reproducible(self):
        from bgc_aggregation.length_evaluation import paired_group_bootstrap
        rows = [{"group_id": "big" if i < 9 else "small", "model_raw_mse": 0.0 if i < 9 else 10.0,
                 "baseline_raw_mse": 1.0, "model_cosine_distance": 0.1, "baseline_cosine_distance": 0.2}
                for i in range(10)]
        result = paired_group_bootstrap(rows, 300, 42, engine="numpy")
        self.assertEqual(result, paired_group_bootstrap(rows, 300, 42, engine="numpy"))
        self.assertEqual(result["raw_mse_difference"], 0.0)
        self.assertEqual(result["raw_mse_difference_ci95"], [-1.0, 9.0])
        zeros = [dict(r, model_raw_mse=0.0, baseline_raw_mse=0.0) for r in rows]
        self.assertIsNone(paired_group_bootstrap(zeros, 20, engine="numpy")["relative_mse_reduction_ci95"])


if __name__ == "__main__":
    unittest.main()
