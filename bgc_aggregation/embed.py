"""Embed complete proteins, using direct ESMC for short inputs and aggregation for long ones."""

from pathlib import Path

import torch
from torch.nn import functional as F

from .cache import encode_view, save_tensor_file
from .chunking import make_chunks
from .common import MAX_RESIDUES, digest, file_digest, read_fasta, read_json, write_json
from .data import collate, predict_batch, to_device
from .encoder import ESMCEncoder
from .evaluate import load_checkpoint


@torch.inference_mode()
def embed(args):
    sequences = read_fasta(args.fasta)
    model, checkpoint = load_checkpoint(args.checkpoint)
    contract = checkpoint["feature_contract"]
    encoder = ESMCEncoder(contract["encoder"]["model"], args.device, args.weights,
                          args.esm_batch_size, args.token_budget)
    if encoder.identity != contract["encoder"]:
        raise ValueError("Inference encoder weights, version, or precision differ from training features")
    model = model.to(encoder.device).eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"checkpoint_sha256": file_digest(args.checkpoint),
                "fasta_sha256": file_digest(args.fasta), "feature_contract": contract}
    run_id = digest(identity)
    manifest = output / "embedding_run.json"
    if manifest.exists() and read_json(manifest)["run_id"] != run_id:
        raise ValueError("Inference settings changed; use a new output directory")
    if not manifest.exists() and any(output.iterdir()):
        raise ValueError("Inference output must be empty or belong to this run")
    write_json(manifest, {**identity, "run_id": run_id, "complete": False})
    records, vectors, baselines = [], [], []
    for index, (identifier, sequence) in enumerate(sequences.items(), 1):
        shard = output / f"{digest(identifier)[:24]}.pt"
        if shard.exists():
            result = torch.load(shard, map_location="cpu", weights_only=True)
            if result["run_id"] != run_id or result["sequence_id"] != identifier:
                raise ValueError(f"Incompatible inference shard: {shard}")
        else:
            if len(sequence) <= MAX_RESIDUES:
                vector = encoder.pool([(sequence, 0, len(sequence))])[0]
                baseline, count, method = vector, 1, "direct_esmc"
            else:
                chunks = make_chunks(len(sequence), contract["deployment_core_size"], contract["halo"])
                view = {"core_size": contract["deployment_core_size"], "offset": 0, "chunks": chunks}
                features = encode_view(encoder, sequence, view)
                vector = predict_batch(model, to_device(collate([features]), encoder.device))[0].cpu()
                baseline = (features["embeddings"] * features["weights"].unsqueeze(-1)).sum(dim=0)
                count, method = len(chunks), "aggregated_chunks"
            if not torch.isfinite(vector).all():
                raise ValueError(f"Non-finite embedding for {identifier}")
            result = {"run_id": run_id, "sequence_id": identifier, "length": len(sequence),
                      "chunks": count, "method": method, "embedding": vector, "baseline": baseline}
            save_tensor_file(shard, result)
        vectors.append(result["embedding"])
        baselines.append(result["baseline"])
        records.append({key: result[key] for key in ("sequence_id", "length", "chunks", "method")})
        if index == 1 or index % 25 == 0 or index == len(sequences):
            print(f"Embedded {index}/{len(sequences)} proteins", flush=True)
    matrix = torch.stack(vectors)
    save_tensor_file(output / "embeddings.pt", {"ids": list(sequences), "embeddings": matrix,
                     "normalized_embeddings": F.normalize(matrix, dim=-1),
                     "weighted_mean_embeddings": torch.stack(baselines), "metadata": records,
                     "run_id": run_id})
    write_json(output / "proteins.json", records)
    write_json(manifest, {**identity, "run_id": run_id, "complete": True, "proteins": len(records)})
    print(f"Protein embeddings: {output / 'embeddings.pt'}")
