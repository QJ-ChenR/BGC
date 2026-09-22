"""Dependency-free file formats, sequence validation, and reproducibility helpers."""

import csv
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = 1
MAX_RESIDUES = 2046
POSITION_DIM = 5
LENGTH_SCALE = 512.0
TARGET_FAMILIES = {"NRPS", "PKS", "HYBRID"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def stable_seed(seed, key):
    return int(digest([seed, key])[:16], 16)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_table(path, key="sequence_id"):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if key not in (reader.fieldnames or []):
            raise ValueError(f"Missing column {key!r} in {path}")
        rows = {}
        for row in reader:
            if not row[key] or row[key] in rows:
                raise ValueError(f"Empty or duplicate {key} in {path}: {row[key]!r}")
            rows[row[key]] = row
    return rows


def read_fasta(path):
    """Uppercase residues and remove one terminal stop; reject gaps/internal stops."""
    records = {}
    identifier, parts = None, []

    def save():
        if identifier is None:
            return
        sequence = "".join(parts).upper()
        if sequence.endswith("*"):
            sequence = sequence[:-1]
        invalid = set(sequence) - set("ACDEFGHIKLMNPQRSTVWYXBZUO")
        if not sequence or invalid:
            raise ValueError(f"Invalid protein {identifier}: empty sequence or symbols {sorted(invalid)}")
        if identifier in records:
            raise ValueError(f"Duplicate FASTA identifier: {identifier}")
        records[identifier] = sequence

    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                save()
                words = line[1:].split()
                if not words:
                    raise ValueError("Empty FASTA header")
                identifier, parts = words[0], []
            elif identifier is None:
                raise ValueError("Sequence found before the first FASTA header")
            else:
                parts.append("".join(line.split()))
    save()
    if not records:
        raise ValueError(f"No proteins found in {path}")
    return records


def verify_dataset(directory):
    directory = Path(directory)
    metadata = read_json(directory / "dataset.json")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported dataset schema")
    for name, expected in metadata["files"].items():
        if file_digest(directory / name) != expected:
            raise ValueError(f"Dataset file changed after preparation: {name}")
    return metadata
