"""Import the supplied UniProt TSV pools without sequence-homology clustering."""

import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from .chunking import make_views
from .common import SCHEMA_VERSION, MAX_RESIDUES, digest, file_digest, write_json, write_jsonl
from .protocol import GENERAL_LENGTH_PROTOCOL, validate_length_entries


TSV_FIELDS = ("accession", "group", "other_subgroup", "organism", "scientific_name",
              "organism_taxon_id", "species_key_taxon_id", "genus_taxon_id", "length_aa",
              "reviewed", "protein_existence", "protein_name", "sequence_sha256", "uniprot_url", "sequence")


def read_general_table(path, provided_split):
    # The input file determines its allowed length range; do not trim sequences
    # to make an invalid row fit the training or test cohort.
    limits = (500, 1499) if provided_split == "short_pool" else (1500, 2000)
    parents = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        # utf-8-sig also accepts spreadsheet exports with a UTF-8 byte-order mark.
        reader = csv.DictReader(handle, delimiter="\t")
        if (not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or set(TSV_FIELDS) - set(reader.fieldnames)):
            raise ValueError(f"Missing or duplicate TSV columns in {path}; expected {TSV_FIELDS}")
        for number, row in enumerate(reader, 2):
            location = f"{path}:{number}"
            if None in row or any(row.get(field) is None for field in TSV_FIELDS):
                raise ValueError(f"Malformed TSV row at {location}")
            # Check residues before hashing so the supplied checksum is verified
            # against the original sequence, without silent cleanup or substitution.
            accession, sequence = row["accession"], row["sequence"]
            if not accession or any(c.isspace() for c in accession):
                raise ValueError(f"Invalid accession at {location}")
            if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWYXBZUO"):
                raise ValueError(f"Invalid protein sequence at {location}; supply uppercase ungapped residues")
            if not row["length_aa"].isdigit() or int(row["length_aa"]) != len(sequence):
                raise ValueError(f"Sequence length mismatch at {location}")
            if not limits[0] <= len(sequence) <= limits[1]:
                raise ValueError(f"Protein outside {limits[0]}-{limits[1]} aa at {location}")
            checksum = hashlib.sha256(sequence.encode()).hexdigest()
            if row["sequence_sha256"] != checksum:
                raise ValueError(f"sequence_sha256 mismatch at {location}")
            # Use stable numeric taxon IDs for grouping; organism names can vary.
            for field in ("organism_taxon_id", "species_key_taxon_id", "genus_taxon_id"):
                if not row[field].isdigit() or int(row[field]) < 1 or str(int(row[field])) != row[field]:
                    raise ValueError(f"Missing or invalid {field} at {location}")
            if row["group"] not in {"Bacteria", "Fungi", "Other"}:
                raise ValueError(f"Invalid taxonomy group at {location}")
            # Taxonomy is metadata, not a biosynthetic family or a model feature.
            # Keeping the real species key makes cross-length overlap auditable.
            parents.append({"parent_id": accession, "sequence": sequence, "length": len(sequence),
                            "bgc_id": None, "family": "GENERAL", "dataset_source": "general",
                            "provided_split": provided_split, "taxonomy_group": row["group"],
                            "group_id": "species:" + row["species_key_taxon_id"],
                            **{field: row[field] for field in TSV_FIELDS if field not in
                               {"accession", "sequence", "length_aa", "group"}}})
    if not parents:
        raise ValueError(f"Empty protein TSV: {path}")
    return parents


def split_short_pool(parents, seed=42):
    """Greedy species allocation balancing protein counts and taxonomy/length strata."""
    # Allocate whole species together so related rows from that species cannot
    # cross train/validation/short-test boundaries. Cross-species homology can remain.
    groups = defaultdict(list)
    for parent in sorted(parents, key=lambda p: p["parent_id"]):
        groups[parent["group_id"]].append(parent)
    if len(groups) < 3:
        raise ValueError("At least three short-pool species groups are required")

    # Balance taxonomy jointly with length, including the small one-chunk cohort.
    def stratum(parent):
        upper = next(b for b in (512, 1023, 1499) if parent["length"] <= b)
        return parent["taxonomy_group"], upper

    total_strata = Counter(stratum(p) for p in parents)
    fractions, names = (0.8, 0.1, 0.1), ("train", "validation", "test")
    counts, strata = [0, 0, 0], [Counter(), Counter(), Counter()]
    ordered = [groups[key] for key in sorted(groups)]
    random.Random(seed).shuffle(ordered)
    # Place large groups first; the preceding shuffle breaks size ties reproducibly.
    ordered.sort(key=len, reverse=True)
    for index, group in enumerate(ordered):
        additions = Counter(stratum(p) for p in group)

        # Choose the split with the smallest increase in normalized squared error,
        # balancing both its total count and its taxonomy/length counts.
        def cost(i):
            target = len(parents) * fractions[i]
            delta = ((counts[i] + len(group) - target) ** 2 - (counts[i] - target) ** 2) / target
            for key, number in additions.items():
                target = total_strata[key] * fractions[i]
                delta += ((strata[i][key] + number - target) ** 2 - (strata[i][key] - target) ** 2) / target
            return delta

        # Reserve the final groups for any empty splits; never split a species
        # merely to reach an exact 80/10/10 protein ratio.
        empty = [i for i in range(3) if not counts[i]]
        candidates = empty if len(ordered) - index == len(empty) else range(3)
        chosen = min(candidates, key=cost)
        counts[chosen] += len(group)
        strata[chosen].update(additions)
        for parent in group:
            parent["split"] = names[chosen]


def prepare_general(args):
    # Validate inputs and split membership before writing a new dataset directory.
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Preparation requires a new or empty directory: {output}")
    short = read_general_table(args.short_tsv, "short_pool")
    long = read_general_table(args.long_tsv, "long_test")
    parents = short + long
    # Reject repeated accessions or exact sequences across both input pools.
    # This check is not a sequence-similarity or homology search.
    for field in ("parent_id", "sequence_sha256"):
        duplicates = [key for key, count in Counter(p[field] for p in parents).items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate {field} across input rows: {duplicates[:5]}")
    # Only the short pool participates in allocation. Every supplied long protein
    # stays in test, even when its species also occurs in short training data.
    split_short_pool(short, args.seed)
    for parent in long:
        parent["split"] = "test"
    # Carry these isolation rules into caches and checkpoints so later stages
    # cannot accidentally apply the stricter MIBiG protocol to this dataset.
    experiment = {"protocol": GENERAL_LENGTH_PROTOCOL, "length_cutoff": 1500,
                  "chunk_views": "fixed", "crops_per_parent": 0, "grouping": "species_key_taxon_id",
                  "group_isolation_scope": "short_pool", "long_test_policy": "fixed_allow_species_overlap"}
    # One natural protein supplies one full-sequence teacher and one chunk view.
    # A sample ID binds the accession to its sequence for reproducible cache names.
    samples = []
    for parent in parents:
        role = ("test_short" if parent["length"] < 1500 else "test_long") if parent["split"] == "test" else parent["split"]
        parent["experiment_role"] = role
        sample_id = digest(["general", parent["parent_id"], parent["sequence_sha256"]])[:24]
        samples.append({**parent, "sample_id": sample_id, "source": "native", "crop_start": 0,
                        "crop_end": parent["length"], "views": make_views(
                            parent["length"], sample_id, args.seed, halo=args.halo, mode="fixed")})
    cohorts = validate_length_entries(samples, experiment)
    # Report overlap with the actual training split separately from overlap with
    # the entire short pool; only the former defines the evaluation seen/unseen flag.
    training_species = {p["group_id"] for p in short if p["split"] == "train"}
    short_species = {p["group_id"] for p in short}
    summary = {"parents": len(parents), "teacher_samples": len(samples), "short_pool": len(short),
               "long_test": len(long), "length_cohorts": cohorts,
               "long_test_species_seen_in_short_pool": sum(p["group_id"] in short_species for p in long),
               "long_test_species_seen_in_training": sum(p["group_id"] in training_species for p in long),
               "cohort_taxonomy_counts": dict(sorted(Counter(
                   f'{p["experiment_role"]}/{p["taxonomy_group"]}' for p in parents).items())),
               "cohort_chunk_counts": dict(sorted(Counter(
                   f'{p["experiment_role"]}/{(p["length"] + 511) // 512}' for p in parents).items()))}
    output.mkdir(parents=True, exist_ok=True)
    # Keep full provenance in JSONL and a compact, readable split audit in TSV.
    write_jsonl(output / "parents.jsonl", parents)
    write_jsonl(output / "samples.jsonl", samples)
    with (output / "selection.tsv").open("w", newline="") as handle:
        fields = ("parent_id", "provided_split", "split", "experiment_role", "length", "group_id",
                  "taxonomy_group", "other_subgroup", "species_key_taxon_id", "sequence_sha256")
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(parents)
    # Input and output checksums tie this split and its settings to later caches.
    metadata = {"schema_version": SCHEMA_VERSION, "experiment": experiment, "seed": args.seed,
                "scope": "all", "validation_scope": "all", "halo": args.halo, "deployment_core_size": 512,
                "max_residues": MAX_RESIDUES, "crops_per_parent": 0, "homology_clustered": False,
                "split_method": "species-grouped greedy taxonomy/length balance", "split_fractions": [0.8, 0.1, 0.1],
                "homology_note": "No homology screen; species grouping does not establish sequence independence.",
                "input_hashes": {"short_tsv": file_digest(args.short_tsv), "long_tsv": file_digest(args.long_tsv)},
                "files": {name: file_digest(output / name) for name in ("parents.jsonl", "samples.jsonl", "selection.tsv")},
                "summary": summary}
    metadata["dataset_id"] = digest(metadata)
    write_json(output / "dataset.json", metadata)
    print(f"Prepared {len(short)} short-pool proteins and {len(long)} fixed long-test proteins.")
    for role, counts in cohorts.items():
        print(f"  {role}: {counts}")
    print("Short splits isolate species; the fixed long test may share species and homologs with training.")
    print(f"Selection audit: {output / 'selection.tsv'}")
