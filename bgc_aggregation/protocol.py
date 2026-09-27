"""Length-extrapolation contracts shared by preparation, training, and evaluation."""

from collections import defaultdict

from .common import MAX_RESIDUES, TARGET_FAMILIES


LENGTH_PROTOCOL = "length-extrapolation"


def experiment_config(args):
    length_experiment = args.protocol == LENGTH_PROTOCOL
    if length_experiment and not 2 <= args.length_cutoff <= MAX_RESIDUES:
        raise ValueError(f"Length cutoff must be between 2 and {MAX_RESIDUES}")
    crops = args.crops_per_parent
    crops = (0 if length_experiment else 4) if crops is None else crops
    if length_experiment and crops:
        raise ValueError("Length-extrapolation requires --crops-per-parent 0 (natural proteins only)")
    return {"protocol": args.protocol,
            "length_cutoff": args.length_cutoff if length_experiment else None,
            "chunk_views": args.chunk_views or ("fixed" if length_experiment else "multiscale"),
            "crops_per_parent": crops}


def is_length_experiment(metadata):
    return metadata.get("experiment", {}).get("protocol") == LENGTH_PROTOCOL


def experiment_role(parent, scope, cutoff):
    if parent["family"] not in TARGET_FAMILIES and scope != "all":
        return "excluded_family"
    if parent["length"] > MAX_RESIDUES:
        return "excluded_above_teacher_limit"
    if parent["split"] == "test":
        return "test_short" if parent["length"] < cutoff else "test_long"
    if parent["length"] >= cutoff:
        return "excluded_long_" + parent["split"]
    return parent["split"]


def cohort_counts(entries, key):
    buckets = defaultdict(list)
    for entry in entries:
        buckets[key(entry)].append(entry)
    return {name: {"parents": len({e["parent_id"] for e in group}),
                   "groups": len({e["group_id"] for e in group}),
                   "families": {family: len({e["parent_id"] for e in group if e["family"] == family})
                                for family in sorted({e["family"] for e in group})}}
            for name, group in sorted(buckets.items())}


def validate_length_entries(entries, experiment, require_test=True):
    """Reject protocol violations before feature extraction or parameter updates."""
    cutoff = experiment["length_cutoff"]
    if not isinstance(cutoff, int) or not 2 <= cutoff <= MAX_RESIDUES:
        raise ValueError("Invalid length-extrapolation cutoff")
    seen = {field: {} for field in ("parent_id", "bgc_id", "group_id")}
    parents = set()
    for entry in entries:
        if entry["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown length-extrapolation split")
        if entry["source"] != "native" or not 1 <= entry["length"] <= MAX_RESIDUES:
            raise ValueError("Length-extrapolation accepts only complete natural proteins within the teacher limit")
        if entry["split"] != "test" and entry["length"] >= cutoff:
            raise ValueError("Length-extrapolation training and validation must be shorter than the cutoff")
        if entry["parent_id"] in parents:
            raise ValueError("Length-extrapolation requires one natural sample per parent")
        parents.add(entry["parent_id"])
        for field, mapping in seen.items():
            previous = mapping.setdefault(entry[field], entry["split"])
            if previous != entry["split"]:
                raise ValueError(f"Length-extrapolation leakage across splits: {field}={entry[field]}")
    def role(e):
        return ("test_short" if e["length"] < cutoff else "test_long") if e["split"] == "test" else e["split"]
    counts = cohort_counts([e for e in entries if e["family"] in TARGET_FAMILIES], role)
    required = {"train", "validation", "test_short", "test_long"} if require_test else {"train", "validation"}
    missing = required - counts.keys()
    if missing:
        raise ValueError(f"No target samples for length-extrapolation cohorts {sorted(missing)}; "
                         f"inspect group membership or change the split seed before training. Counts: {counts}")
    return counts
