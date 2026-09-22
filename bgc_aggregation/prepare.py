"""Select synthases and split parent proteins before generating training crops."""

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .chunking import make_views
from .common import (MAX_RESIDUES, SCHEMA_VERSION, TARGET_FAMILIES, digest,
                     file_digest, read_fasta, read_table, stable_seed, write_json,
                     write_jsonl)


NRPS_RULES = {"NRPS", "NRPS-like", "NRP-metallophore"}
PKS_RULES = {"T1PKS", "transAT-PKS", "transAT-PKS-like", "PKS-like"}
OTHER_PKS_RULES = {"T2PKS", "T3PKS", "HR-T2PKS"}


def classify(annotation):
    """Use this CDS's rule hits, never a whole-BGC label or product-name guess."""
    rules = sorted(set(re.findall(r"(?:^|\|)\s*biosynthetic \(rule-based-clusters\) ([^:|\s]+):",
                                 annotation.get("gene_functions", ""))))
    nrps, pks = bool(NRPS_RULES.intersection(rules)), bool(PKS_RULES.intersection(rules))
    family = "HYBRID" if nrps and pks else "NRPS" if nrps else "PKS" if pks else "OTHER"
    if family == "OTHER" and OTHER_PKS_RULES.intersection(rules):
        family = "OTHER_PKS"
    return family, rules


class UnionFind:
    def __init__(self, identifiers):
        self.parent = {key: key for key in identifiers}

    def find(self, key):
        while key != self.parent[key]:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def split_parents(parents, cluster_path=None, seed=42, fractions=(0.8, 0.1, 0.1)):
    """Join BGC, exact-sequence, and optional homology links before splitting."""
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("Three positive split fractions must sum to one")
    by_id = {p["parent_id"]: p for p in parents}
    union = UnionFind(by_id)
    previous = {}
    for parent in parents:
        identifier = parent["parent_id"]
        for key in [("bgc", parent["bgc_id"]), ("sequence", digest(parent["sequence"]))]:
            if key in previous:
                union.union(identifier, previous[key])
            previous[key] = identifier
    if cluster_path:
        members = {}
        with Path(cluster_path).open(newline="") as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) != 2:
                    raise ValueError("Homology TSV must contain two headerless columns: representative, member")
                representative, member = row
                if representative not in by_id or member not in by_id:
                    raise ValueError(f"Homology TSV contains an unknown identifier: {row}")
                if member in members and members[member] != representative:
                    raise ValueError(f"Multiple homology representatives for {member}")
                members[member] = representative
                union.union(representative, member)
        if set(members) != set(by_id):
            raise ValueError("Homology TSV must cover every input protein, including singleton clusters")
    components = defaultdict(list)
    for identifier in sorted(by_id):
        components[union.find(identifier)].append(identifier)
    if len(components) < 3:
        raise ValueError("At least three independent groups are required for train/validation/test splits")
    groups = list(components.values())
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    names = ("train", "validation", "test")
    targets = [len(parents) * f for f in fractions]
    counts = [0, 0, 0]
    for index, group in enumerate(groups):
        empty = [i for i, count in enumerate(counts) if not count]
        candidates = empty if len(groups) - index == len(empty) else range(3)
        # Minimize the increase in normalized squared allocation error.
        chosen = min(candidates, key=lambda i: ((counts[i] + len(group) - targets[i]) ** 2
                                               - (counts[i] - targets[i]) ** 2) / targets[i])
        counts[chosen] += len(group)
        group_id = digest(sorted(group))[:20]
        for identifier in group:
            by_id[identifier].update(split=names[chosen], group_id=group_id)
    return parents


def crop_intervals(length, count, seed):
    if count < 0:
        raise ValueError("Crops per parent must be nonnegative")
    rng, intervals = random.Random(seed), []
    for index in range(count):
        size = (1024, 1536, MAX_RESIDUES, MAX_RESIDUES)[index % 4]
        size = min(size, length)
        start = 0 if index == 0 else length - size if index == 1 else rng.randint(0, length - size)
        interval = (start, start + size)
        if interval not in intervals:
            intervals.append(interval)
    return intervals


def prepare(args):
    sequences, annotations = read_fasta(args.fasta), read_table(args.annotations)
    if set(sequences) - set(annotations):
        raise ValueError("Every FASTA protein must have an annotation row")
    labels = read_table(args.labels) if args.labels else {}
    if set(labels) - set(sequences):
        raise ValueError("Label overrides contain unknown protein identifiers")
    parents = []
    for identifier, sequence in sorted(sequences.items()):
        annotation = annotations[identifier]
        family, rules = classify(annotation)
        if identifier in labels:
            family = labels[identifier].get("family", "").upper()
            if family not in TARGET_FAMILIES | {"OTHER", "OTHER_PKS"}:
                raise ValueError(f"Invalid family override for {identifier}: {family}")
        bgc = annotation.get("bgc_id", "")
        if not bgc:
            raise ValueError(f"Missing BGC identifier for {identifier}")
        parents.append({"parent_id": identifier, "bgc_id": bgc, "family": family,
                        "rules": rules, "label_source": "override" if identifier in labels else "rule_hits",
                        "sequence": sequence, "length": len(sequence)})
    split_parents(parents, args.clusters, args.seed)
    samples = []
    for parent in parents:
        is_target = parent["family"] in TARGET_FAMILIES
        if parent["length"] <= MAX_RESIDUES and (is_target or args.scope == "all"):
            intervals = [(0, parent["length"])]
            source = "native"
        elif parent["length"] > MAX_RESIDUES and is_target:
            intervals = crop_intervals(parent["length"], args.crops_per_parent,
                                       stable_seed(args.seed, parent["parent_id"]))
            source = "crop"
        else:
            continue
        for start, end in intervals:
            sample_id = digest([parent["parent_id"], start, end, source])[:24]
            samples.append({"sample_id": sample_id, "parent_id": parent["parent_id"],
                            "bgc_id": parent["bgc_id"], "family": parent["family"],
                            "group_id": parent["group_id"], "split": parent["split"],
                            "source": source, "crop_start": start, "crop_end": end,
                            "sequence": parent["sequence"][start:end], "length": end - start,
                            "views": make_views(end - start, sample_id, args.seed, halo=args.halo)})
    for split in ("train", "validation", "test"):
        if not any(s["split"] == split and s["family"] in TARGET_FAMILIES for s in samples):
            raise ValueError(f"No target teacher samples in {split}; inspect groups or change the split seed")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Preparation requires a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "parents.jsonl", parents)
    write_jsonl(output / "samples.jsonl", samples)
    with (output / "selection.tsv").open("w", newline="") as handle:
        fields = ["sequence_id", "bgc_id", "family", "label_source", "rules", "length", "group_id", "split", "target_family"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for parent in parents:
            writer.writerow({**parent, "sequence_id": parent["parent_id"], "rules": ";".join(parent["rules"]),
                             "target_family": parent["family"] in TARGET_FAMILIES})
    summary = {"parents": len(parents), "teacher_samples": len(samples),
               "parent_counts": dict(sorted(Counter(f'{p["split"]}/{p["family"]}/'
                   f'{"long" if p["length"] > MAX_RESIDUES else "short"}' for p in parents).items())),
               "sample_counts": dict(sorted(Counter(f'{s["split"]}/{s["family"]}/{s["source"]}'
                                                    for s in samples).items())),
               "groups": len({p["group_id"] for p in parents}),
               "largest_group_parents": max(Counter(p["group_id"] for p in parents).values()),
               "unselected_long_ids": [p["parent_id"] for p in parents
                                       if p["length"] > MAX_RESIDUES and p["family"] not in TARGET_FAMILIES]}
    metadata = {"schema_version": SCHEMA_VERSION, "seed": args.seed, "scope": args.scope,
                "halo": args.halo, "deployment_core_size": 512, "max_residues": MAX_RESIDUES,
                "crops_per_parent": args.crops_per_parent,
                "homology_clustered": bool(args.clusters),
                "input_hashes": {"fasta": file_digest(args.fasta), "annotations": file_digest(args.annotations),
                                 "clusters": file_digest(args.clusters) if args.clusters else None,
                                 "labels": file_digest(args.labels) if args.labels else None},
                "files": {name: file_digest(output / name) for name in ("parents.jsonl", "samples.jsonl")},
                "summary": summary}
    metadata["dataset_id"] = digest(metadata)
    write_json(output / "dataset.json", metadata)
    if not args.clusters:
        print("NOTE: Splits isolate BGCs and exact duplicates; non-identical homologs are not controlled.")
    print(f"Prepared {len(parents)} parent proteins, {len(samples)} teacher samples, {summary['groups']} groups.")
    for key, count in summary["sample_counts"].items():
        print(f"  {key}: {count}")
    print(f"Selection audit: {output / 'selection.tsv'}")
    if summary["unselected_long_ids"]:
        print(f"Long proteins outside the target rule filter: {len(summary['unselected_long_ids'])}. "
              "Review dataset.json/selection.tsv and use --labels for reviewed overrides.")
    print("Crop targets represent isolated fragments, not full-length long-protein embeddings.")
