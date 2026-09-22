"""Parent-balanced sampling from small cached tensors; no ESM dependency."""

import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .common import POSITION_DIM, TARGET_FAMILIES, stable_seed


def select_entries(metadata, split, scope="target", sources="both"):
    entries = [e for e in metadata["entries"] if e["split"] == split
               and (scope == "all" or e["family"] in TARGET_FAMILIES)
               and (sources == "both" or e["source"] == sources)]
    if not entries:
        raise ValueError(f"No samples for split={split}, scope={scope}, sources={sources}")
    return entries


def load_record(directory, entry):
    return torch.load(Path(directory) / entry["file"], map_location="cpu", weights_only=True)


class ParentDataset(Dataset):
    """One example per parent per epoch, then sample a crop and a chunk view."""

    def __init__(self, directory, entries, seed=42):
        self.directory, self.seed, self.epoch = directory, seed, 0
        groups = defaultdict(list)
        for entry in entries:
            groups[entry["parent_id"]].append(entry)
        self.parents = [groups[key] for key in sorted(groups)]

    def __len__(self):
        return len(self.parents)

    def __getitem__(self, index):
        entries = self.parents[index]
        rng = random.Random(stable_seed(self.seed, [self.epoch, entries[0]["parent_id"]]))
        entry = rng.choice(entries)
        record = load_record(self.directory, entry)
        # Half of the views use the exact deployment partition.
        view = record["views"][0] if rng.random() < 0.5 else rng.choice(record["views"])
        return {**view, "teacher": record["teacher"], "entry": entry}


def collate(items):
    count = len(items)
    maximum = max(len(item["weights"]) for item in items)
    dimension = items[0]["embeddings"].shape[-1]
    embeddings = torch.zeros(count, maximum, dimension)
    positions = torch.zeros(count, maximum, POSITION_DIM)
    weights = torch.zeros(count, maximum)
    padding_mask = torch.ones(count, maximum, dtype=torch.bool)
    for row, item in enumerate(items):
        length = len(item["weights"])
        embeddings[row, :length] = item["embeddings"]
        positions[row, :length] = item["positions"]
        weights[row, :length] = item["weights"]
        padding_mask[row, :length] = False
    result = {"embeddings": embeddings, "positions": positions,
              "weights": weights, "padding_mask": padding_mask}
    if "teacher" in items[0]:
        result["teacher"] = torch.stack([item["teacher"] for item in items])
    return result


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def predict_batch(model, batch):
    return model(batch["embeddings"], batch["positions"], batch["weights"], batch["padding_mask"])


def teacher_statistics(directory, entries):
    """Each parent contributes equal weight, regardless of its crop count."""
    groups = defaultdict(list)
    for entry in entries:
        groups[entry["parent_id"]].append(entry)
    first, second = [], []
    for group in groups.values():
        vectors = torch.stack([load_record(directory, entry)["teacher"].double() for entry in group])
        first.append(vectors.mean(dim=0))
        second.append(vectors.square().mean(dim=0))
    mean = torch.stack(first).mean(dim=0)
    variance = (torch.stack(second).mean(dim=0) - mean.square()).mean().item()
    if variance <= 1e-8:
        raise ValueError("Training teacher variance is too small for meaningful reconstruction")
    return {"mean": mean.float(), "variance": variance, "parents": len(groups)}
