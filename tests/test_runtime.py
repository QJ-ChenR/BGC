"""PyTorch-only numerical and workflow tests; skipped on dependency-free laptops."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from bgc_aggregation.__main__ import build_parser
    from bgc_aggregation.cache import encode_view, extract_cache, load_cache, save_tensor_file
    from bgc_aggregation.chunking import make_chunks, make_views
    from bgc_aggregation.common import file_digest, read_json, write_json, write_jsonl
    from bgc_aggregation.data import ParentDataset, collate, teacher_statistics
    from bgc_aggregation.evaluate import evaluate, load_checkpoint, neighborhood_retention
    from bgc_aggregation.model import ChunkAggregator, ModelConfig, reconstruction_loss, weighted_mean
    from bgc_aggregation.train import train


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed; run these tests on the server")
class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)

    def features(self, count, dimension=8):
        return {"embeddings": torch.randn(count, dimension), "positions": torch.rand(count, 5),
                "weights": torch.ones(count) / count, "core_size": 512, "offset": 0}

    def model(self, architecture="transformer"):
        return ChunkAggregator(ModelConfig(embedding_dim=8, hidden_dim=16, num_heads=4,
                                            feedforward_dim=32, dropout=0.0, architecture=architecture))

    def test_initial_output_equals_baseline_and_gradients_flow(self):
        batch = collate([self.features(2), self.features(5)])
        for architecture in ("transformer", "mlp", "lstm"):
            model = self.model(architecture)
            prediction = model(**batch)
            baseline = weighted_mean(batch["embeddings"], batch["weights"])
            torch.testing.assert_close(prediction, baseline)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            for _ in range(2):
                optimizer.zero_grad()
                loss = reconstruction_loss(model(**batch), baseline + 0.2, 1.0).mean()
                loss.backward()
                optimizer.step()
            self.assertGreater(model.input_projection.weight.grad.abs().sum().item(), 0)
            if architecture == "lstm":
                for name, parameter in model.lstm.named_parameters():
                    with self.subTest(parameter=name):
                        self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_padding_cannot_change_predictions_after_learning(self):
        model = self.model().eval()
        torch.nn.init.normal_(model.output_projection.weight, std=0.1)
        short, long = self.features(2), self.features(37)
        alone, padded = collate([short]), collate([short, long])
        expected = model(**alone)[0]
        torch.testing.assert_close(model(**padded)[0], expected, atol=1e-6, rtol=1e-5)
        padded["embeddings"][0, 2:] = 1e6
        padded["positions"][0, 2:] = -1e6
        padded["weights"][0, 2:] = 100
        torch.testing.assert_close(model(**padded)[0], expected, atol=1e-6, rtol=1e-5)

    def test_lstm_padding_and_unsorted_lengths_match_individual_predictions(self):
        model = self.model("lstm").eval()
        torch.nn.init.normal_(model.output_projection.weight, std=0.1)
        # One chunk, unseen counts, and unsorted lengths exercise both packing
        # and restoration of the original batch order. Nonzero head exposes bugs.
        items = [self.features(n) for n in (2, 37, 1, 4)]
        items[0]["weights"] = torch.tensor([0.7, 0.3])
        expected = torch.stack([model(**collate([item]))[0] for item in items])
        batch = collate(items)
        torch.testing.assert_close(model(**batch), expected, atol=1e-6, rtol=1e-5)
        mask = batch["padding_mask"]
        batch["embeddings"][mask] = 1e6
        batch["positions"][mask] = -1e6
        batch["weights"][mask] = 100
        torch.testing.assert_close(model(**batch), expected, atol=1e-6, rtol=1e-5)
        batch["positions"] = torch.randn_like(batch["positions"]) * 100
        torch.testing.assert_close(model(**batch), expected, atol=1e-6, rtol=1e-5)

    def test_lstm_responds_to_chunk_order_with_the_same_weighted_mean(self):
        model = self.model("lstm").eval()
        torch.nn.init.normal_(model.output_projection.weight, std=0.1)
        features = self.features(5)
        order = torch.tensor([2, 0, 4, 1, 3])
        permuted = {**features, **{key: features[key][order]
                                  for key in ("embeddings", "positions", "weights")}}
        original, shuffled = collate([features]), collate([permuted])
        torch.testing.assert_close(weighted_mean(original["embeddings"], original["weights"]),
                                   weighted_mean(shuffled["embeddings"], shuffled["weights"]))
        self.assertGreater((model(**original) - model(**shuffled)).abs().max().item(), 1e-5)

    def test_lstm_configuration_and_invalid_sequence_masks(self):
        with self.assertRaisesRegex(ValueError, "even"):
            self_config = ModelConfig(architecture="lstm", hidden_dim=7, num_heads=1)
            ChunkAggregator(self_config)
        # LSTM has no attention-head divisibility requirement. Single-layer
        # LSTMs must not receive PyTorch's inter-layer dropout argument.
        model = ChunkAggregator(ModelConfig(architecture="lstm", embedding_dim=8,
                                            hidden_dim=6, num_layers=1, dropout=0.1))
        self.assertEqual(model.lstm.hidden_size, 3)
        self.assertEqual(model.lstm.dropout, 0.0)
        self.assertFalse(model.specification()["use_positions"])
        for mask in ([False, True, False], [True, True, True]):
            batch = collate([self.features(3)])
            batch["padding_mask"][0] = torch.tensor(mask)
            with self.assertRaisesRegex(ValueError, "nonempty sequences with right padding"):
                model(**batch)

    def test_context_pooling_has_no_overlap_double_count(self):
        class FakeEncoder:
            def pool(self, requests):
                return [torch.tensor([sum(ord(c) for c in sequence[start:end]) / (end - start)])
                        for sequence, start, end in requests]

        sequence = "ACDEFGHIKLMNPQRSTVWY" * 90
        view = {"core_size": 512, "offset": 123, "chunks": make_chunks(len(sequence), offset=123)}
        features = encode_view(FakeEncoder(), sequence, view)
        result = (features["embeddings"][:, 0] * features["weights"]).sum()
        self.assertAlmostEqual(result.item(), sum(map(ord, sequence)) / len(sequence), places=4)

    def test_adapter_excludes_bos_eos_padding_and_preserves_request_order(self):
        from bgc_aggregation.encoder import ESMCEncoder

        class Tokenizer:
            pad_token_id = 999

            def __call__(self, sequence, add_special_tokens=True):
                return {"input_ids": [100] + ["ACDE".index(residue) + 1 for residue in sequence] + [200]}

        class FakeModel:
            tokenizer = Tokenizer()

            def __call__(self, sequence_tokens):
                return SimpleNamespace(embeddings=sequence_tokens.float().unsqueeze(-1))

        encoder = ESMCEncoder.__new__(ESMCEncoder)
        encoder.model, encoder.device = FakeModel(), torch.device("cpu")
        encoder.batch_size, encoder.token_budget = 2, 4096
        vectors = encoder.pool([("ACDE", 0, 4), ("AC", 0, 2), ("ACDE", 1, 3)])
        self.assertEqual([vector.item() for vector in vectors], [2.5, 1.5, 2.5])
        with self.assertRaisesRegex(ValueError, "length"):
            encoder.pool([("A" * 2047, 0, 2047)])

    def make_cache(self, directory):
        directory.mkdir()
        entries = []
        for index in range(18):
            split = "train" if index < 10 else "validation" if index < 14 else "test"
            views = [self.features(2 + index % 4), self.features(3)]
            target = (views[0]["embeddings"] * views[0]["weights"].unsqueeze(-1)).sum(0) + 0.15
            path = directory / f"sample{index}.pt"
            save_tensor_file(path, {"cache_id": "synthetic", "sample_id": str(index), "teacher": target, "views": views})
            entries.append({"sample_id": str(index), "parent_id": f"protein{index}", "group_id": f"group{index}",
                            "bgc_id": f"bgc{index}", "family": "NRPS", "source": "native", "split": split,
                            "length": 1000, "file": path.name, "sha256": file_digest(path), "views": 2})
        metadata = {"schema_version": 1, "cache_id": "synthetic", "dataset_id": "synthetic-data",
                    "encoder": {"embedding_dim": 8}, "halo": 64, "deployment_core_size": 512,
                    "complete": True, "homology_clustered": True, "entries": entries}
        write_json(directory / "cache.json", metadata)
        return metadata

    def training_args(self, cache, output, architecture="transformer"):
        return build_parser().parse_args(["train", "--cache", str(cache), "--output", str(output), "--device", "cpu",
                                         "--architecture", architecture, "--epochs", "3", "--batch-size", "5", "--hidden-dim", "16", "--heads", "4",
                                         "--feedforward-dim", "32", "--dropout", "0.1", "--learning-rate", "0.002"])

    def test_parent_balance_and_training_only_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache"
            metadata = self.make_cache(path)
            entries = metadata["entries"][:10]
            duplicated = entries + [dict(entries[0], sample_id="duplicate")]
            dataset = ParentDataset(path, duplicated)
            self.assertEqual(len(dataset), 10)
            normal = teacher_statistics(path, entries)
            repeated = teacher_statistics(path, duplicated)
            torch.testing.assert_close(normal["mean"], repeated["mean"])
            self.assertAlmostEqual(normal["variance"], repeated["variance"])

    def test_feature_extraction_resume_and_incomplete_cache_rejection(self):
        class FakeEncoder:
            identity = {"embedding_dim": 2, "model": "fake"}

            def __init__(self, *args):
                pass

            def pool(self, requests):
                return [torch.tensor([float(end - start), 1.0]) for _, start, end in requests]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            samples = [{"sample_id": str(i), "parent_id": str(i), "bgc_id": str(i), "family": "NRPS",
                        "group_id": str(i), "split": "train", "source": "native", "length": 800,
                        "sequence": "A" * 800, "views": make_views(800, str(i))} for i in range(2)]
            write_jsonl(dataset / "samples.jsonl", samples)
            write_jsonl(dataset / "parents.jsonl", [])
            write_json(dataset / "dataset.json", {"schema_version": 1, "dataset_id": "synthetic",
                       "files": {name: file_digest(dataset / name) for name in ("samples.jsonl", "parents.jsonl")},
                       "halo": 64, "deployment_core_size": 512, "summary": {}, "homology_clustered": False})
            args = build_parser().parse_args(["cache", "--dataset", str(dataset), "--output", str(root / "cache"),
                                             "--limit", "1", "--device", "cpu"])
            with patch("bgc_aggregation.encoder.ESMCEncoder", FakeEncoder):
                extract_cache(args)
                checksum = file_digest(root / "cache/0.pt")
                with self.assertRaisesRegex(ValueError, "complete cache"):
                    load_cache(root / "cache")
                args.limit = None
                extract_cache(args)
                self.assertEqual(file_digest(root / "cache/0.pt"), checksum)
                self.assertEqual(len(load_cache(root / "cache")["entries"]), 2)

    def test_training_resume_and_evaluation_without_esm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            self.make_cache(cache)
            full_args = self.training_args(cache, root / "full")
            train(full_args)
            partial_args = self.training_args(cache, root / "resumed")

            class Interrupted(Exception):
                pass

            def interrupt_after_epoch(path, value):
                save_tensor_file(path, value)
                if Path(path).name == "last.pt" and value["epoch"] == 1:
                    raise Interrupted()

            with patch("bgc_aggregation.train.save_tensor_file", side_effect=interrupt_after_epoch):
                with self.assertRaises(Interrupted):
                    train(partial_args)
            partial_args.resume = root / "resumed/last.pt"
            train(partial_args)
            full = torch.load(root / "full/last.pt", weights_only=True)
            resumed = torch.load(root / "resumed/last.pt", weights_only=True)
            for key in full["model_state"]:
                torch.testing.assert_close(full["model_state"][key], resumed["model_state"][key], atol=0, rtol=0)
            args = build_parser().parse_args(["evaluate", "--cache", str(cache), "--checkpoint", str(root / "full/best.pt"),
                                             "--output", str(root / "evaluation"), "--device", "cpu"])
            evaluate(args)
            report = read_json(root / "evaluation/metrics.json")
            self.assertEqual(report["overall"]["parents"], 4)
            changed = load_cache(cache)
            changed["cache_id"] = "different"
            with self.assertRaisesRegex(ValueError, "differ"):
                load_checkpoint(root / "full/best.pt", changed)

    def test_length_protocol_survives_feature_caching(self):
        class FakeEncoder:
            identity = {"embedding_dim": 2, "model": "fake"}

            def __init__(self, *args):
                pass

            def pool(self, requests):
                return [torch.tensor([float(end - start), 1.0]) for _, start, end in requests]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "prepared"
            dataset.mkdir()
            experiment = {"protocol": "length-extrapolation", "length_cutoff": 1500,
                          "chunk_views": "fixed", "crops_per_parent": 0}
            samples = []
            for i, (split, length) in enumerate([("train", 1499), ("validation", 1499),
                                               ("test", 1499), ("test", 1537)]):
                samples.append({"sample_id": str(i), "parent_id": str(i), "group_id": str(i),
                                "bgc_id": str(i), "family": "NRPS", "source": "native", "split": split,
                                "length": length, "sequence": "A" * length,
                                "views": make_views(length, str(i), mode="fixed")})
            write_jsonl(dataset / "samples.jsonl", samples)
            write_jsonl(dataset / "parents.jsonl", [])
            write_json(dataset / "dataset.json", {"schema_version": 1, "dataset_id": "length-test",
                       "files": {name: file_digest(dataset / name) for name in ("samples.jsonl", "parents.jsonl")},
                       "halo": 64, "deployment_core_size": 512, "summary": {}, "homology_clustered": False,
                       "experiment": experiment})
            args = build_parser().parse_args(["cache", "--dataset", str(dataset),
                                             "--output", str(root / "cache"), "--device", "cpu"])
            with patch("bgc_aggregation.encoder.ESMCEncoder", FakeEncoder):
                extract_cache(args)
            cached = load_cache(root / "cache")
            self.assertEqual(cached["experiment"], experiment)
            self.assertEqual([e["split"] for e in cached["entries"]], [s["split"] for s in samples])
            self.assertTrue(all(e["views"] == 1 for e in cached["entries"]))
            held_out = torch.load(root / "cache/3.pt", weights_only=True)
            self.assertEqual(len(held_out["views"][0]["weights"]), 4)

    def make_length_cache(self, directory):
        metadata = self.make_cache(directory)
        metadata["experiment"] = {"protocol": "length-extrapolation", "length_cutoff": 1500,
                                  "chunk_views": "multiscale", "crops_per_parent": 0}
        for index, entry in enumerate(metadata["entries"]):
            length = (800, 1024, 1499)[index % 3] if index < 14 else (1499, 1500, 1537, 2046)[index - 14]
            entry["length"] = length
            views = [self.features((length + 511) // 512), self.features(12)]
            views[1].update(core_size=128, offset=20)
            target = (views[0]["embeddings"] * views[0]["weights"].unsqueeze(-1)).sum(0) + 0.15
            path = directory / entry["file"]
            save_tensor_file(path, {"cache_id": metadata["cache_id"], "sample_id": entry["sample_id"],
                                    "teacher": target, "views": views})
            entry["sha256"] = file_digest(path)
        write_json(directory / "cache.json", metadata)
        return metadata

    def test_length_training_fixed_views_statistics_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            metadata = self.make_length_cache(cache)
            entries = [e for e in metadata["entries"] if e["split"] == "train"]
            dataset = ParentDataset(cache, entries, views="fixed")
            for epoch in range(10):
                dataset.epoch = epoch
                self.assertTrue(all(len(dataset[i]["weights"]) <= 3 for i in range(len(dataset))))
            args = self.training_args(cache, root / "run")
            train(args)
            checkpoint = torch.load(root / "run/best.pt", weights_only=True)
            self.assertEqual(checkpoint["training_config"]["views"], "fixed")
            self.assertEqual(checkpoint["training_config"]["sources"], "native")
            self.assertEqual(checkpoint["training_inputs"]["max_chunks"], 3)
            self.assertLess(checkpoint["validation_inputs"]["max_length"], 1500)
            expected = teacher_statistics(cache, entries)
            torch.testing.assert_close(checkpoint["teacher_statistics"]["mean"], expected["mean"])
            flags = ["--plots"] if importlib.util.find_spec("matplotlib") else []
            evaluation = build_parser().parse_args([
                "evaluate", "--cache", str(cache), "--checkpoint", str(root / "run/best.pt"),
                "--output", str(root / "report"), "--bootstrap-replicates", "50", "--device", "cpu", *flags])
            evaluate(evaluation)
            report = read_json(root / "report/metrics.json")["length_extrapolation"]
            self.assertEqual(report["cohorts"]["short"]["overall"]["proteins"], 1)
            self.assertEqual(report["cohorts"]["long"]["overall"]["proteins"], 3)
            self.assertEqual(report["cohorts"]["long_beyond_chunk_range"]["overall"]["proteins"], 2)
            self.assertEqual(report["cohorts"]["long"]["neighborhood_retention"]["eligible_queries"], 3)
            self.assertEqual(report["cohorts"]["long"]["neighborhood_retention"]["gallery_parents"], 4)
            self.assertTrue((root / "report/length_summary.tsv").is_file())
            if flags:
                self.assertTrue((root / "report/length_error.svg").is_file())
                self.assertTrue((root / "report/length_gain.svg").is_file())
            evaluation.all_views = True
            evaluation.output = root / "all_views"
            evaluate(evaluation)
            augmented_report = read_json(root / "all_views/metrics.json")["length_extrapolation"]
            self.assertEqual(report, augmented_report)
            evaluation.split, evaluation.output = "validation", root / "validation_report"
            evaluate(evaluation)
            validation = read_json(root / "validation_report/metrics.json")["length_extrapolation"]
            self.assertIsNone(validation["cohorts"]["long"]["overall"])
            augmented_args = self.training_args(cache, root / "multiscale")
            augmented_args.views = "multiscale"
            train(augmented_args)
            augmented_checkpoint = torch.load(root / "multiscale/last.pt", weights_only=True)
            self.assertEqual(augmented_checkpoint["training_inputs"]["max_chunks"], 12)

    def test_lstm_length_training_resume_checkpoint_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            self.make_length_cache(cache)
            train(self.training_args(cache, root / "full", "lstm"))
            args = self.training_args(cache, root / "resumed", "lstm")

            class Interrupted(Exception):
                pass

            def interrupt(path, value):
                save_tensor_file(path, value)
                if Path(path).name == "last.pt" and value["epoch"] == 1:
                    raise Interrupted()

            with patch("bgc_aggregation.train.save_tensor_file", side_effect=interrupt):
                with self.assertRaises(Interrupted):
                    train(args)
            args.resume = root / "resumed/last.pt"
            args.architecture = "transformer"  # Saved model/config must win.
            args.views = "multiscale"
            train(args)
            full = torch.load(root / "full/last.pt", weights_only=True)
            resumed = torch.load(args.resume, weights_only=True)
            self.assertEqual(resumed["model_config"]["architecture"], "lstm")
            self.assertFalse(resumed["model_config"]["use_positions"])
            self.assertEqual(resumed["training_config"]["views"], "fixed")
            self.assertEqual(resumed["training_config"]["sources"], "native")
            self.assertEqual(resumed["training_inputs"]["max_chunks"], 3)
            self.assertLess(resumed["validation_inputs"]["max_length"], 1500)
            for key in full["model_state"]:
                torch.testing.assert_close(full["model_state"][key], resumed["model_state"][key], atol=0, rtol=0)
            self.assertEqual(full["history"], resumed["history"])
            run = read_json(root / "resumed/run.json")
            self.assertEqual(run["model"], resumed["model_config"])
            model, saved = load_checkpoint(args.resume, load_cache(cache))
            self.assertEqual(model.specification(), saved["model_config"])
            evaluation = build_parser().parse_args([
                "evaluate", "--cache", str(cache), "--checkpoint", str(root / "resumed/best.pt"),
                "--output", str(root / "report"), "--bootstrap-replicates", "50", "--device", "cpu"])
            evaluate(evaluation)
            report = read_json(root / "report/metrics.json")["length_extrapolation"]
            self.assertEqual(report["cohorts"]["long"]["overall"]["proteins"], 3)
            self.assertEqual(report["cohorts"]["long_beyond_chunk_range"]["overall"]["proteins"], 2)
            self.assertTrue((root / "report/length_summary.tsv").is_file())
            self.assertTrue((root / "report/per_sample.tsv").is_file())

    def test_length_training_rejects_leakage_pretraining_and_missing_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            metadata = self.make_length_cache(cache)
            args = self.training_args(cache, root / "run")
            args.init_from = root / "old.pt"
            with self.assertRaisesRegex(ValueError, "init-from"):
                train(args)
            args.init_from = None
            args.sources = "crop"
            with self.assertRaisesRegex(ValueError, "native"):
                train(args)
            args.sources = "native"
            metadata["entries"][0]["length"] = 1500
            write_json(cache / "cache.json", metadata)
            with self.assertRaisesRegex(ValueError, "shorter"):
                train(args)
            metadata["entries"][0]["length"] = 800
            metadata["experiment"]["chunk_views"] = "fixed"
            write_json(cache / "cache.json", metadata)
            args.views = "multiscale"
            with self.assertRaisesRegex(ValueError, "chunk-views multiscale"):
                train(args)
            self.assertFalse((root / "run").exists())

    def test_length_resume_restores_fixed_sampling_and_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            self.make_length_cache(cache)
            train(self.training_args(cache, root / "full"))
            args = self.training_args(cache, root / "resumed")

            class Interrupted(Exception):
                pass

            def interrupt(path, value):
                save_tensor_file(path, value)
                if Path(path).name == "last.pt" and value["epoch"] == 1:
                    raise Interrupted()

            with patch("bgc_aggregation.train.save_tensor_file", side_effect=interrupt):
                with self.assertRaises(Interrupted):
                    train(args)
            args.resume = root / "resumed/last.pt"
            args.views = "multiscale"  # Resume must preserve the original fixed-view protocol.
            train(args)
            expected = torch.load(root / "full/last.pt", weights_only=True)
            actual = torch.load(args.resume, weights_only=True)
            self.assertEqual(actual["training_config"]["views"], "fixed")
            for key in expected["model_state"]:
                torch.testing.assert_close(expected["model_state"][key], actual["model_state"][key], atol=0, rtol=0)
            actual["experiment"]["length_cutoff"] = 1600
            save_tensor_file(args.resume, actual)
            with self.assertRaisesRegex(ValueError, "protocol differs"):
                train(args)

    def test_legacy_checkpoint_without_step_metadata_still_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            self.make_length_cache(cache)
            train(self.training_args(cache, root / "full"))
            args = self.training_args(cache, root / "legacy")
            class Interrupted(Exception):
                pass

            def save_legacy(path, value):
                value = dict(value)
                value["training_config"] = dict(value["training_config"])
                value["training_config"].pop("max_steps")
                value["training_config"].pop("validation_scope")
                value.pop("global_step")
                value.pop("planned_optimizer_steps")
                save_tensor_file(path, value)
                if Path(path).name == "last.pt" and value["epoch"] == 1:
                    raise Interrupted()

            with patch("bgc_aggregation.train.save_tensor_file", side_effect=save_legacy):
                with self.assertRaises(Interrupted):
                    train(args)
            args.resume = args.output / "last.pt"
            train(args)
            full = torch.load(root / "full/last.pt", weights_only=True)
            resumed = torch.load(args.resume, weights_only=True)
            self.assertEqual(resumed["global_step"], 6)
            self.assertIsNone(resumed["training_config"]["max_steps"])
            for key in full["model_state"]:
                torch.testing.assert_close(full["model_state"][key], resumed["model_state"][key], atol=0, rtol=0)

    def test_neighbor_reuse_matches_independent_query_search(self):
        from torch.nn import functional as F
        from bgc_aggregation.evaluate import neighborhood_scores, summarize_neighborhood

        teacher = torch.randn(131, 8)
        tensors = {"teacher": teacher, "model": teacher + torch.randn_like(teacher) * 0.6,
                   "baseline": teacher + torch.randn_like(teacher),
                   "entries": [{"parent_id": str(i), "group_id": str(i // 2)} for i in range(130)]
                              + [{"parent_id": "0", "group_id": "0"}]}
        for center in (None, teacher.mean(0)):
            expected = {"model": [], "baseline": []}
            vectors = {key: value[:130] - center if center is not None else value[:130]
                       for key, value in tensors.items() if key != "entries"}
            for index in (0, 129):
                candidates = [i for i in range(130) if i // 2 != index // 2]
                def neighbors(key):
                    scores = F.cosine_similarity(vectors[key][index].unsqueeze(0),
                                                 vectors["teacher"][candidates], dim=-1)
                    return {candidates[j] for j in scores.topk(5).indices.tolist()}
                target = neighbors("teacher")
                for key in expected:
                    expected[key].append(len(target & neighbors(key)) / 5)
            scores = neighborhood_scores(tensors, 5, center)
            report = summarize_neighborhood(scores, {"0", "129"})
            self.assertEqual(report["gallery_parents"], 130)
            self.assertEqual(report["eligible_queries"], 2)
            for key in expected:
                self.assertAlmostEqual(report[key], sum(expected[key]) / 2)
            self.assertEqual(summarize_neighborhood(scores, set())["eligible_queries"], 0)

    def test_neighbors_exclude_same_group_and_duplicate_parents(self):
        vectors = torch.eye(4)
        tensors = {"model": vectors, "teacher": vectors, "baseline": vectors,
                   "entries": [{"parent_id": str(i), "group_id": "same" if i < 2 else str(i)} for i in range(4)]}
        report = neighborhood_retention(tensors, k=2)
        self.assertEqual(report["model"], 1.0)
        self.assertEqual(report["eligible_queries"], 4)
        restricted = neighborhood_retention(tensors, k=2, query_parents={"2"})
        self.assertEqual(restricted["eligible_queries"], 1)
        self.assertEqual(restricted["gallery_parents"], 4)


if __name__ == "__main__":
    unittest.main()
