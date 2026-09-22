"""Command-line entry point; preparation and help need only the standard library."""

import argparse
import importlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Value must be positive")
    return value


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("Value must be nonnegative")
    return value


def add_encoder_options(parser):
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu, or auto")
    parser.add_argument("--weights", type=Path, help="Optional local official ESMC .pth weights")
    parser.add_argument("--esm-batch-size", type=positive, default=2)
    parser.add_argument("--token-budget", type=positive, default=4096,
                        help="Maximum padded tokens per ESMC batch; must be at least 2048")


def build_parser():
    parser = argparse.ArgumentParser(description="Distill full-sequence ESMC embeddings from protein chunks.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Select proteins, group splits, and generate teacher crops")
    prepare.add_argument("--fasta", type=Path, default=Path("data/processed/core_genes/core_genes.faa"))
    prepare.add_argument("--annotations", type=Path, default=Path("data/processed/core_genes/core_genes.tsv"))
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--labels", type=Path, help="Optional TSV overrides with sequence_id and family columns")
    prepare.add_argument("--clusters", type=Path, help="Complete headerless MMseqs2 representative/member TSV")
    prepare.add_argument("--scope", choices=("target", "all"), default="target")
    prepare.add_argument("--crops-per-parent", type=nonnegative, default=4)
    prepare.add_argument("--halo", type=nonnegative, default=64)
    prepare.add_argument("--seed", type=nonnegative, default=42)

    cache = commands.add_parser("cache", help="Extract frozen ESMC teachers and multi-view chunk features")
    cache.add_argument("--dataset", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--model", choices=("esmc_300m", "esmc_600m"), default="esmc_600m")
    cache.add_argument("--limit", type=positive, help="Debug extraction only; an incomplete cache cannot be trained")
    add_encoder_options(cache)

    train = commands.add_parser("train", help="Train a residual transformer or mean-vector MLP baseline")
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--architecture", choices=("transformer", "mlp"), default="transformer")
    train.add_argument("--hidden-dim", type=positive, default=128)
    train.add_argument("--layers", type=positive, default=2)
    train.add_argument("--heads", type=positive, default=4)
    train.add_argument("--feedforward-dim", type=positive, default=512)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--no-positions", action="store_true")
    train.add_argument("--epochs", type=positive, default=100)
    train.add_argument("--batch-size", type=positive, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--warmup-fraction", type=float, default=0.05)
    train.add_argument("--patience", type=positive, default=10)
    train.add_argument("--cosine-weight", type=float, default=0.5)
    train.add_argument("--gradient-clip", type=float, default=1.0)
    train.add_argument("--seed", type=nonnegative, default=42)
    train.add_argument("--scope", choices=("target", "all"), default="target")
    train.add_argument("--sources", choices=("both", "native", "crop"), default="both")
    train.add_argument("--resume", type=Path, help="Restore an entire run from last.pt in the same output directory")
    train.add_argument("--init-from", type=Path, help="Initialize weights for a new fine-tuning run on the same cache")

    evaluate = commands.add_parser("evaluate", help="Compare held-out reconstruction with the weighted-mean baseline")
    evaluate.add_argument("--cache", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument("--scope", choices=("target", "all"), default="target")
    evaluate.add_argument("--sources", choices=("both", "native", "crop"), default="both")
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch-size", type=positive, default=64)
    evaluate.add_argument("--neighbors", type=positive, default=10)
    evaluate.add_argument("--all-views", action="store_true", help="Also evaluate alternate chunk partitions")

    embed = commands.add_parser("embed", help="Create one vector per protein, resuming completed proteins")
    embed.add_argument("--fasta", type=Path, required=True)
    embed.add_argument("--checkpoint", type=Path, required=True)
    embed.add_argument("--output", type=Path, required=True)
    add_encoder_options(embed)

    check = commands.add_parser("check-env", help="Inspect server dependencies and optionally run ESMC inference")
    check.add_argument("--check-esmc", action="store_true", help="Load ESMC weights and check batched residue pooling")
    check.add_argument("--model", choices=("esmc_300m", "esmc_600m"), default="esmc_600m")
    add_encoder_options(check)
    return parser


def check_environment(args):
    for name in ("torch", "esm", "transformers"):
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = "not installed"
        print(f"{name}: {installed}")
    if importlib.util.find_spec("torch") is None:
        print("Preparation and dependency-free tests can run here. GPU stages require the server environment.")
        if args.check_esmc:
            raise ValueError("ESMC inference requires the server dependencies")
        return
    import torch

    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    if args.check_esmc:
        from .encoder import ESMCEncoder

        encoder = ESMCEncoder(args.model, args.device, args.weights, args.esm_batch_size, args.token_budget)
        sequence = "M" + "ACDEFGHIKLMNPQRSTVWY" * 12
        requests = [(sequence, 0, len(sequence)), (sequence[:100], 15, 70)]
        batched = encoder.pool(requests)
        single = [encoder.pool([request])[0] for request in requests]
        for left, right in zip(batched, single):
            if left.shape != (encoder.identity["embedding_dim"],):
                raise ValueError("Unexpected ESMC embedding shape")
            torch.testing.assert_close(left, right, rtol=0.02, atol=0.02)
        print("ESMC inference passed: finite residue means and consistent single/batched outputs.")


def main():
    parser = build_parser()
    args = parser.parse_args()
    targets = {"prepare": ("prepare", "prepare"), "cache": ("cache", "extract_cache"),
               "train": ("train", "train"), "evaluate": ("evaluate", "evaluate"), "embed": ("embed", "embed")}
    try:
        if args.command == "check-env":
            check_environment(args)
        else:
            module, function = targets[args.command]
            getattr(importlib.import_module(f"bgc_aggregation.{module}"), function)(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")
    except (ModuleNotFoundError, PackageNotFoundError) as error:
        parser.exit(2, f"Missing server dependency: {error}\nSee docs/aggregation.md for installation instructions.\n")


if __name__ == "__main__":
    main()
