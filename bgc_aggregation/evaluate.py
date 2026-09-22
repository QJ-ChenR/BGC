"""Reconstruction and neighborhood preservation, with explicit crop provenance."""

import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .cache import feature_contract, load_cache
from .common import write_json
from .data import collate, load_record, predict_batch, select_entries, to_device
from .model import ChunkAggregator, ModelConfig, weighted_mean


METRICS = ("model_mse", "baseline_mse", "mean_mse", "model_raw_mse", "baseline_raw_mse", "mean_raw_mse", "model_cosine_distance",
           "baseline_cosine_distance", "mean_cosine_distance", "model_loss", "baseline_loss")


def parent_average(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["parent_id"]].append(row)
    return {"parents": len(groups), "observations": len(rows), **{
        key: sum(sum(r[key] for r in group) / len(group) for group in groups.values()) / len(groups)
        for key in METRICS}}


@torch.inference_mode()
def reconstruct(model, directory, entries, statistics, device, batch_size=64,
                cosine_weight=0.5, all_views=False):
    model.eval()
    rows, predictions, teachers, baselines, item_metadata = [], [], [], [], []
    pending = []

    def consume(items):
        batch = to_device(collate(items), device)
        prediction = predict_batch(model, batch).float().cpu()
        target = batch["teacher"].float().cpu()
        baseline = weighted_mean(batch["embeddings"], batch["weights"]).cpu()
        mean = statistics["mean"].expand_as(target)
        errors = {}
        for name, values in (("model", prediction), ("baseline", baseline), ("mean", mean)):
            errors[name + "_raw_mse"] = (values - target).square().mean(dim=-1)
            errors[name + "_mse"] = errors[name + "_raw_mse"] / statistics["variance"]
            errors[name + "_cosine_distance"] = 1 - F.cosine_similarity(values, target, dim=-1)
        for index, item in enumerate(items):
            entry = item["entry"]
            row = {key: entry[key] for key in ("sample_id", "parent_id", "group_id", "family", "source", "length")}
            row.update(view=item["view_index"], chunks=len(item["weights"]), core_size=item["core_size"])
            row.update({key: float(value[index]) for key, value in errors.items()})
            row["model_loss"] = row["model_mse"] + cosine_weight * row["model_cosine_distance"]
            row["baseline_loss"] = row["baseline_mse"] + cosine_weight * row["baseline_cosine_distance"]
            rows.append(row)
            if item["view_index"] == 0:
                predictions.append(prediction[index])
                teachers.append(target[index])
                baselines.append(baseline[index])
                item_metadata.append(entry)

    for entry in entries:
        record = load_record(directory, entry)
        views = record["views"] if all_views else record["views"][:1]
        for index, view in enumerate(views):
            pending.append({**view, "teacher": record["teacher"], "entry": entry, "view_index": index})
            if len(pending) == batch_size:
                consume(pending)
                pending = []
    if pending:
        consume(pending)
    tensors = {"model": torch.stack(predictions), "teacher": torch.stack(teachers),
               "baseline": torch.stack(baselines), "entries": item_metadata}
    return rows, tensors


def summarize(rows):
    report = {"overall": parent_average(rows)}
    for field in ("family", "source", "core_size"):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        report["by_" + field] = {key: parent_average(value) for key, value in sorted(groups.items())}
    for field, boundaries in (("length", (512, 1024, 1536, 2046)), ("chunks", (1, 2, 4, 8, 16, 32))):
        groups = defaultdict(list)
        for row in rows:
            upper = next((limit for limit in boundaries if row[field] <= limit), None)
            lower = max([0] + [limit for limit in boundaries if limit < row[field]])
            label = f"{lower + 1}-{upper}" if upper is not None else f">{boundaries[-1]}"
            groups[label].append(row)
        report["by_" + field] = {key: parent_average(value) for key, value in groups.items()}
    return report


def neighborhood_retention(tensors, k=10, center=None):
    """Compare predicted and teacher neighbor sets in the same teacher gallery.

    Exclude every candidate in the query's split group, including related crops.
    Use one deterministic sample per parent so crop-rich parents do not dominate.
    """
    indices, seen = [], set()
    for index, entry in enumerate(tensors["entries"]):
        if entry["parent_id"] not in seen:
            indices.append(index)
            seen.add(entry["parent_id"])
    entries = [tensors["entries"][index] for index in indices]
    vectors = {key: value[indices].float() for key, value in tensors.items() if key != "entries"}
    if center is not None:
        vectors = {key: value - center for key, value in vectors.items()}
    vectors = {key: F.normalize(value, dim=-1) for key, value in vectors.items()}
    totals = {"model": [], "baseline": []}
    gallery = vectors["teacher"]
    for start in range(0, len(entries), 128):
        end = min(len(entries), start + 128)
        blocked = torch.tensor([[query["group_id"] == candidate["group_id"] for candidate in entries]
                                for query in entries[start:end]], dtype=torch.bool)
        scores = {key: (value[start:end] @ gallery.T).masked_fill(blocked, -float("inf"))
                  for key, value in vectors.items()}
        for row in range(end - start):
            actual_k = min(k, int((~blocked[row]).sum()))
            if actual_k == 0:
                continue
            expected = set(scores["teacher"][row].topk(actual_k).indices.tolist())
            for key in totals:
                observed = set(scores[key][row].topk(actual_k).indices.tolist())
                totals[key].append(len(expected & observed) / actual_k)
    return {"requested_k": k, "eligible_queries": len(totals["model"]),
            **{key: sum(values) / len(values) if values else None for key, values in totals.items()}}


def load_checkpoint(path, cache=None, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    if cache is not None and (checkpoint["cache_id"] != cache["cache_id"]
                              or checkpoint["feature_contract"] != feature_contract(cache)):
        raise ValueError("Checkpoint and cache differ; use the original prepared dataset and feature cache")
    model = ChunkAggregator(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device), checkpoint


def evaluate(args):
    from .encoder import select_device

    cache = load_cache(args.cache)
    device = select_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, cache, device)
    entries = select_entries(cache, args.split, args.scope, args.sources)
    rows, tensors = reconstruct(model, args.cache, entries, checkpoint["teacher_statistics"],
                               device, args.batch_size, checkpoint["training_config"]["cosine_weight"],
                               args.all_views)
    report = summarize(rows)
    report.update(split=args.split, scope=args.scope, sources=args.sources,
                  cache_id=cache["cache_id"], checkpoint_epoch=checkpoint["epoch"],
                  homology_clustered=cache["homology_clustered"],
                  neighborhood_retention=neighborhood_retention(tensors, args.neighbors),
                  centered_neighborhood_retention=neighborhood_retention(
                      tensors, args.neighbors, checkpoint["teacher_statistics"]["mean"]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", report)
    with (output / "per_sample.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Model loss: {report['overall']['model_loss']:.6f}")
    print(f"Weighted-mean loss: {report['overall']['baseline_loss']:.6f}")
    print(f"Evaluation written to {output}")
