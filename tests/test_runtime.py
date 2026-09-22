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
        for architecture in ("transformer", "mlp"):
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

    def training_args(self, cache, output):
        return build_parser().parse_args(["train", "--cache", str(cache), "--output", str(output), "--device", "cpu",
                                         "--epochs", "3", "--batch-size", "5", "--hidden-dim", "16", "--heads", "4",
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

    def test_neighbors_exclude_same_group_and_duplicate_parents(self):
        vectors = torch.eye(4)
        tensors = {"model": vectors, "teacher": vectors, "baseline": vectors,
                   "entries": [{"parent_id": str(i), "group_id": "same" if i < 2 else str(i)} for i in range(4)]}
        report = neighborhood_retention(tensors, k=2)
        self.assertEqual(report["model"], 1.0)
        self.assertEqual(report["eligible_queries"], 4)


if __name__ == "__main__":
    unittest.main()
