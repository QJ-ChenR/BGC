"""Reconstruction and neighborhood preservation, with explicit crop provenance."""

import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .cache import feature_contract, load_cache
from .common import write_json
from .data import collate, load_record, predict_batch, select_entries, to_device
from .length_evaluation import length_report, plot_length_report, write_length_table
from .model import ChunkAggregator, ModelConfig, weighted_mean
from .protocol import GENERAL_FIELDS, is_general_experiment, is_length_experiment


METRICS = ("model_mse", "baseline_mse", "mean_mse", "model_raw_mse", "baseline_raw_mse", "mean_raw_mse", "model_cosine_distance",
           "baseline_cosine_distance", "mean_cosine_distance", "model_loss", "baseline_loss")


def parent_average(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["parent_id"]].append(row)
    # Average a parent's crop/view errors first so extra observations do not
    # give that protein greater weight in aggregate reconstruction metrics.
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
        # Compare the aggregator with both a chunk-weighted mean and a constant
        # training-teacher mean; all three are scored against the same teacher.
        for name, values in (("model", prediction), ("baseline", baseline), ("mean", mean)):
            errors[name + "_raw_mse"] = (values - target).square().mean(dim=-1)
            errors[name + "_mse"] = errors[name + "_raw_mse"] / statistics["variance"]
            errors[name + "_cosine_distance"] = 1 - F.cosine_similarity(values, target, dim=-1)
        for index, item in enumerate(items):
            entry = item["entry"]
            row = {key: entry[key] for key in ("sample_id", "parent_id", "group_id", "family", "source", "length")}
            row.update({key: entry[key] for key in GENERAL_FIELDS if key in entry})
            row.update(view=item["view_index"], chunks=len(item["weights"]), core_size=item["core_size"])
            row.update({key: float(value[index]) for key, value in errors.items()})
            row["model_loss"] = row["model_mse"] + cosine_weight * row["model_cosine_distance"]
            row["baseline_loss"] = row["baseline_mse"] + cosine_weight * row["baseline_cosine_distance"]
            rows.append(row)
            # Retain only deployment-view vectors for the shared retrieval gallery.
            if item["view_index"] == 0:
                predictions.append(prediction[index])
                teachers.append(target[index])
                baselines.append(baseline[index])
                item_metadata.append(entry)

    # Keep deployment batches identical with and without --all-views. Mixing
    # augmented chunks into those batches changes padding and floating-point results.
    for alternate in ((False, True) if all_views else (False,)):
        for entry in entries:
            record = load_record(directory, entry)
            views = record["views"][1:] if alternate else record["views"][:1]
            for index, view in enumerate(views, start=1 if alternate else 0):
                pending.append({**view, "teacher": record["teacher"], "entry": entry, "view_index": index})
                if len(pending) == batch_size:
                    consume(pending)
                    pending = []
        if pending:
            consume(pending)
            pending = []
    tensors = {"model": torch.stack(predictions), "teacher": torch.stack(teachers),
               "baseline": torch.stack(baselines), "entries": item_metadata}
    return rows, tensors


def summarize(rows):
    report = {"overall": parent_average(rows)}
    fields = ("family", "source", "core_size", "taxonomy_group", "other_subgroup")
    for field in (f for f in fields if f in rows[0]):
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


def neighborhood_scores(tensors, k=10, center=None):
    """Compute each parent's retention once; cohorts reuse the same gallery."""
    indices, seen = [], set()
    for index, entry in enumerate(tensors["entries"]):
        if entry["parent_id"] not in seen:
            indices.append(index)
            seen.add(entry["parent_id"])
    entries = [tensors["entries"][index] for index in indices]
    vectors = {key: value[indices].float() for key, value in tensors.items() if key != "entries"}
    # Optional training-mean centering removes the common embedding offset
    # before cosine comparison, without fitting anything to the held-out split.
    if center is not None:
        vectors = {key: value - center for key, value in vectors.items()}
    vectors = {key: F.normalize(value, dim=-1) for key, value in vectors.items()}
    group_ids = {group: i for i, group in enumerate(sorted({e["group_id"] for e in entries}))}
    groups = torch.tensor([group_ids[e["group_id"]] for e in entries])
    # Keep the gallery fixed to teachers. Only query vectors change between methods.
    gallery = vectors["teacher"]
    by_parent = {}
    for start in range(0, len(entries), 128):
        end = min(len(entries), start + 128)
        # Exclude the query and all same-group candidates (BGC components or species).
        blocked = groups[start:end, None] == groups[None, :]
        scores = {key: (value[start:end] @ gallery.T).masked_fill(blocked, -float("inf"))
                  for key, value in vectors.items()}
        for row in range(end - start):
            actual_k = min(k, int((~blocked[row]).sum()))
            if actual_k == 0:
                continue
            expected = set(scores["teacher"][row].topk(actual_k).indices.tolist())
            by_parent[entries[start + row]["parent_id"]] = {
                key: len(expected & set(scores[key][row].topk(actual_k).indices.tolist())) / actual_k
                for key in ("model", "baseline")}
    return {"requested_k": k, "gallery_parents": len(entries), "by_parent": by_parent}


def summarize_neighborhood(scores, query_parents=None):
    selected = [values for parent, values in scores["by_parent"].items()
                if query_parents is None or parent in query_parents]
    return {"requested_k": scores["requested_k"], "gallery_parents": scores["gallery_parents"],
            "eligible_queries": len(selected),
            **{key: sum(row[key] for row in selected) / len(selected) if selected else None
               for key in ("model", "baseline")}}


def neighborhood_retention(tensors, k=10, center=None, query_parents=None):
    """Exclude same-group candidates and count each parent once."""
    return summarize_neighborhood(neighborhood_scores(tensors, k, center), query_parents)


def load_checkpoint(path, cache=None, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    # Matching model dimensions alone is insufficient: dataset and encoder provenance
    # must match the checkpoint before reconstruction scores are comparable.
    if cache is not None and (checkpoint["cache_id"] != cache["cache_id"]
                              or checkpoint["feature_contract"] != feature_contract(cache)):
        raise ValueError("Checkpoint and cache differ; use the original prepared dataset and feature cache")
    model = ChunkAggregator(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device), checkpoint


def evaluate(args):
    from .encoder import select_device

    cache = load_cache(args.cache)
    length_experiment = is_length_experiment(cache)
    if args.plots:
        if not length_experiment:
            raise ValueError("--plots requires a length-extrapolation cache")
        import matplotlib  # Fail before evaluation if the requested plot dependency is missing.
    device = select_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, cache, device)
    if length_experiment and (checkpoint.get("experiment") != cache["experiment"]
                              or not checkpoint.get("training_inputs")):
        raise ValueError("Checkpoint does not record this length-extrapolation protocol and training inputs")
    if is_general_experiment(cache) and args.scope != "all":
        raise ValueError("General protein evaluation requires --scope all")
    entries = select_entries(cache, args.split, args.scope, args.sources)
    rows, tensors = reconstruct(model, args.cache, entries, checkpoint["teacher_statistics"],
                               device, args.batch_size, checkpoint["training_config"]["cosine_weight"],
                               args.all_views)
    if is_general_experiment(cache):
        # This flag measures species overlap with the actual training split,
        # not whether the protein has a homolog in training.
        training_species = {e["species_key_taxon_id"] for e in cache["entries"] if e["split"] == "train"}
        for row in rows:
            row["species_seen_in_training"] = row["species_key_taxon_id"] in training_species
    report = summarize(rows)
    neighbor_scores = None
    if not args.skip_neighbors:
        print("Computing raw and centered neighbor retention once for the held-out gallery.", flush=True)
        neighbor_scores = (neighborhood_scores(tensors, args.neighbors),
                           neighborhood_scores(tensors, args.neighbors, checkpoint["teacher_statistics"]["mean"]))
    report.update(split=args.split, scope=args.scope, sources=args.sources,
                  cache_id=cache["cache_id"], checkpoint_epoch=checkpoint["epoch"],
                  homology_clustered=cache["homology_clustered"],
                  neighbors_skipped=args.skip_neighbors,
                  neighborhood_retention=summarize_neighborhood(neighbor_scores[0]) if neighbor_scores else None,
                  centered_neighborhood_retention=summarize_neighborhood(neighbor_scores[1]) if neighbor_scores else None)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if length_experiment:
        detail, cohorts = length_report(rows, cache["experiment"], checkpoint["training_inputs"], args.split,
                                        args.bootstrap_replicates, args.bootstrap_seed)
        # Reuse per-query neighbor scores rather than repeating a quadratic search
        # for each length cohort; all cohorts therefore use the same teacher gallery.
        for name, selected in cohorts.items():
            queries = {r["parent_id"] for r in selected}
            detail["cohorts"][name]["neighborhood_retention"] = (
                summarize_neighborhood(neighbor_scores[0], queries) if neighbor_scores else None)
            detail["cohorts"][name]["centered_neighborhood_retention"] = (
                summarize_neighborhood(neighbor_scores[1], queries) if neighbor_scores else None)
        report["length_extrapolation"] = detail
        write_length_table(detail, output)
        if args.plots:
            plot_length_report(detail, output)
        long_result = detail["cohorts"]["long"]["overall"]
        if long_result:
            print(f"Long-cohort raw-MSE evidence vs weighted mean: {long_result['evidence']} "
                  f"({long_result['groups']} groups)")
    write_json(output / "metrics.json", report)
    with (output / "per_sample.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Model loss: {report['overall']['model_loss']:.6f}")
    print(f"Weighted-mean loss: {report['overall']['baseline_loss']:.6f}")
    print(f"Evaluation written to {output}")
