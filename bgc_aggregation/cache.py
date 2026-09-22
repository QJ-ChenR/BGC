"""Resume-safe feature extraction and validated, ESM-free training caches."""

from pathlib import Path

import torch

from .chunking import position_features
from .common import (SCHEMA_VERSION, digest, file_digest, read_json, read_jsonl,
                     verify_dataset, write_json)


def save_tensor_file(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def encode_view(encoder, sequence, view, pooled=None):
    chunks = view["chunks"]
    requests = [(sequence[c["context_start"]:c["context_end"]],
                 c["start"] - c["context_start"], c["end"] - c["context_start"])
                for c in chunks]
    if pooled is None:
        pooled = {}
    missing = list(dict.fromkeys(request for request in requests if request not in pooled))
    pooled.update(zip(missing, encoder.pool(missing)))
    return {"embeddings": torch.stack([pooled[r] for r in requests]),
            "positions": torch.tensor(position_features(chunks, len(sequence)), dtype=torch.float32),
            "weights": torch.tensor([(c["end"] - c["start"]) / len(sequence) for c in chunks],
                                    dtype=torch.float32),
            "core_size": view["core_size"], "offset": view["offset"]}


def extract_cache(args):
    from .encoder import ESMCEncoder

    dataset = verify_dataset(args.dataset)
    samples = read_jsonl(Path(args.dataset) / "samples.jsonl")
    encoder = ESMCEncoder(args.model, args.device, args.weights, args.esm_batch_size, args.token_budget)
    contract = {"schema_version": SCHEMA_VERSION, "dataset_id": dataset["dataset_id"],
                "encoder": encoder.identity, "halo": dataset["halo"],
                "deployment_core_size": dataset["deployment_core_size"]}
    cache_id = digest(contract)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "cache.json"
    if metadata_path.exists() and read_json(metadata_path)["cache_id"] != cache_id:
        raise ValueError("Cache settings or input data changed; use a different output directory")
    if not metadata_path.exists() and any(output.iterdir()):
        raise ValueError("Cache output contains unrelated files; use an empty directory")
    write_json(metadata_path, {**contract, "cache_id": cache_id, "complete": False})
    entries = []
    for number, sample in enumerate(samples, 1):
        path = output / f'{sample["sample_id"]}.pt'
        if args.limit is not None and number > args.limit:
            break
        if path.exists():
            cached = torch.load(path, map_location="cpu", weights_only=True)
            if cached["cache_id"] != cache_id or cached["sample_id"] != sample["sample_id"]:
                raise ValueError(f"Incompatible cached sample: {path}")
        else:
            sequence = sample["sequence"]
            pooled = {}
            teacher = encoder.pool([(sequence, 0, len(sequence))])[0]
            views = [encode_view(encoder, sequence, view, pooled) for view in sample["views"]]
            cached = {"cache_id": cache_id, "sample_id": sample["sample_id"],
                      "teacher": teacher, "views": views}
            save_tensor_file(path, cached)
        entries.append({key: sample[key] for key in
                        ("sample_id", "parent_id", "bgc_id", "family", "group_id", "split", "source", "length")}
                       | {"file": path.name, "sha256": file_digest(path), "views": len(cached["views"])})
        if number == 1 or number % 25 == 0 or number == len(samples):
            print(f"Cached {number}/{len(samples)} teacher samples", flush=True)
    complete = len(entries) == len(samples)
    write_json(metadata_path, {**contract, "cache_id": cache_id, "complete": complete,
                              "entries": entries, "dataset_summary": dataset["summary"],
                              "homology_clustered": dataset["homology_clustered"]})
    print(f"Cache {'complete' if complete else 'incomplete; rerun without --limit'}: {output}")


def load_cache(directory):
    directory = Path(directory)
    metadata = read_json(directory / "cache.json")
    if metadata["schema_version"] != SCHEMA_VERSION or not metadata.get("complete"):
        raise ValueError("Training/evaluation requires a complete cache with a supported schema")
    for entry in metadata["entries"]:
        path = directory / entry["file"]
        if not path.is_file() or file_digest(path) != entry["sha256"]:
            raise ValueError(f"Missing or modified cache file: {path}")
    return metadata


def feature_contract(cache):
    return {key: cache[key] for key in ("schema_version", "encoder", "halo", "deployment_core_size")}
