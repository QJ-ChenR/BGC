"""Length-extrapolation contracts shared by preparation, training, and evaluation."""

from collections import defaultdict

from .common import MAX_RESIDUES, TARGET_FAMILIES


LENGTH_PROTOCOL = "length-extrapolation"
GENERAL_LENGTH_PROTOCOL = "general-length-extrapolation"
GENERAL_FIELDS = ("dataset_source", "provided_split", "taxonomy_group", "other_subgroup",
                  "organism_taxon_id", "species_key_taxon_id", "genus_taxon_id", "sequence_sha256")


def experiment_config(args):
    length_experiment = args.protocol == LENGTH_PROTOCOL
    if length_experiment and not 2 <= args.length_cutoff <= MAX_RESIDUES:
        raise ValueError(f"Length cutoff must be between 2 and {MAX_RESIDUES}")
    # Crops suit the standard reconstruction task, but would confound a length
    # experiment intended to train only on complete short proteins.
    crops = args.crops_per_parent
    crops = (0 if length_experiment else 4) if crops is None else crops
    if length_experiment and crops:
        raise ValueError("Length-extrapolation requires --crops-per-parent 0 (natural proteins only)")
    return {"protocol": args.protocol,
            "length_cutoff": args.length_cutoff if length_experiment else None,
            "chunk_views": args.chunk_views or ("fixed" if length_experiment else "multiscale"),
            "crops_per_parent": crops}


def is_length_experiment(metadata):
    return metadata.get("experiment", {}).get("protocol") in {LENGTH_PROTOCOL, GENERAL_LENGTH_PROTOCOL}


def experiment_role(parent, scope, cutoff):
    if parent["family"] not in TARGET_FAMILIES and scope != "all":
        return "excluded_family"
    if parent["length"] > MAX_RESIDUES:
        return "excluded_above_teacher_limit"
    # Assign roles without moving a parent out of its previously allocated group.
    if parent["split"] == "test":
        return "test_short" if parent["length"] < cutoff else "test_long"
    if parent["length"] >= cutoff:
        return "excluded_long_" + parent["split"]
    return parent["split"]


def cohort_counts(entries, key):
    # Count unique parents rather than views/crops to describe biological samples.
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
    # Dispatch explicitly: general data allow cross-length species overlap,
    # while the MIBiG protocol isolates BGCs and groups across every split.
    if experiment.get("protocol") == GENERAL_LENGTH_PROTOCOL:
        return validate_general_entries(entries, experiment, require_test)
    cutoff = experiment["length_cutoff"]
    if not isinstance(cutoff, int) or not 2 <= cutoff <= MAX_RESIDUES:
        raise ValueError("Invalid length-extrapolation cutoff")
    # Each identifier records its first split; any later different split is leakage.
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
    # Require usable target cohorts before spending time on ESMC extraction.
    counts = cohort_counts([e for e in entries if e["family"] in TARGET_FAMILIES], role)
    required = {"train", "validation", "test_short", "test_long"} if require_test else {"train", "validation"}
    missing = required - counts.keys()
    if missing:
        raise ValueError(f"No target samples for length-extrapolation cohorts {sorted(missing)}; "
                         f"inspect group membership or change the split seed before training. Counts: {counts}")
    return counts


def is_general_experiment(metadata):
    return metadata.get("experiment", {}).get("protocol") == GENERAL_LENGTH_PROTOCOL


def validate_general_entries(entries, experiment, require_test=True):
    """Species isolation applies to the short pool; the supplied long test is fixed."""
    # Reject an altered contract rather than silently relaxing isolation rules.
    if (experiment.get("length_cutoff") != 1500 or experiment.get("chunk_views") != "fixed"
            or experiment.get("crops_per_parent") != 0
            or experiment.get("grouping") != "species_key_taxon_id"
            or experiment.get("group_isolation_scope") != "short_pool"
            or experiment.get("long_test_policy") != "fixed_allow_species_overlap"):
        raise ValueError("Invalid general length-extrapolation contract")
    seen_parents, seen_samples, seen_hashes, short_species = set(), set(), set(), {}
    for entry in entries:
        if any(field not in entry for field in GENERAL_FIELDS):
            raise ValueError("Missing general protein provenance fields")
        if (entry["source"] != "native" or entry["family"] != "GENERAL"
                or entry["dataset_source"] != "general" or entry.get("bgc_id") is not None):
            raise ValueError("General training requires natural general proteins, without BGC labels")
        species = entry["species_key_taxon_id"]
        if not isinstance(species, str) or not species.isdigit() or int(species) < 1:
            raise ValueError("Invalid species_key_taxon_id")
        # Prefixing the species with its split would hide overlap, so retain one
        # canonical group ID for that species across both input files.
        if entry["group_id"] != "species:" + species:
            raise ValueError("General group_id must retain the original species key")
        if entry["taxonomy_group"] not in {"Bacteria", "Fungi", "Other"}:
            raise ValueError("Invalid taxonomy_group")
        # Protein identity and exact-sequence uniqueness apply to the entire dataset.
        for field, seen in (("parent_id", seen_parents), ("sample_id", seen_samples),
                            ("sequence_sha256", seen_hashes)):
            value = entry[field]
            if not value or value in seen:
                raise ValueError(f"Duplicate or empty general protein {field}")
            seen.add(value)
        if entry["provided_split"] == "short_pool":
            if entry["split"] not in {"train", "validation", "test"} or not 500 <= entry["length"] < 1500:
                raise ValueError("Short-pool proteins must be 500-1499 aa with a valid split")
            # Species isolation applies only among the short-pool splits.
            previous = short_species.setdefault(species, entry["split"])
            if previous != entry["split"]:
                raise ValueError(f"Short-pool species leakage across splits: {species}")
        # Long proteins may share species with training, but must remain held out.
        elif entry["provided_split"] == "long_test":
            if entry["split"] != "test" or not 1500 <= entry["length"] <= 2000:
                raise ValueError("Supplied long proteins must stay in test at 1500-2000 aa")
        else:
            raise ValueError("Unknown provided_split")
    counts = cohort_counts(entries, lambda e: (
        "test_short" if e["length"] < 1500 else "test_long") if e["split"] == "test" else e["split"])
    required = {"train", "validation", "test_short", "test_long"} if require_test else {"train", "validation"}
    if required - counts.keys():
        raise ValueError(f"Missing general cohorts: {sorted(required - counts.keys())}")
    return counts
