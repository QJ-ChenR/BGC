#!/usr/bin/env python3
"""Train/evaluate general-protein aggregators using the shared length runner.

Run prepare-general and cache first. This script defaults to all three methods,
seeds 42/43/44, and a 10,000-update cap. Checkpoints use short validation only.
"""

import argparse
import subprocess
from pathlib import Path

if __package__:
    from . import run_length_extrapolation as shared
else:
    import run_length_extrapolation as shared


GENERAL_PROTOCOL = "general-length-extrapolation"


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


def build_parser():
    parser = shared.build_parser()
    parser.description = __doc__
    parser.set_defaults(cache=Path("data/embeddings/general_length1500"),
                        prefix="general_length1500", architectures=shared.ARCHITECTURES)
    parser.add_argument("--epochs", type=positive, default=20)
    parser.add_argument("--max-steps", type=positive, default=10000)
    parser.add_argument("--patience", type=positive, default=5)
    parser.add_argument("--seeds", type=nonnegative, nargs="+", default=shared.SEEDS)
    parser.add_argument("--skip-neighbors", action="store_true",
                        help="Omit neighbor search; reconstruction and confidence intervals remain available")
    parser.add_argument("--bootstrap-replicates", type=positive, default=2000)
    parser.add_argument("--bootstrap-seed", type=nonnegative, default=42)
    # The shared parser retains the legacy defaults when used by the BGC runner.
    for action in parser._actions:
        if action.dest == "architectures":
            action.help = "Methods to run (default: transformer mlp lstm)"
    return parser


def run(args):
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Specify each training seed only once")
    training = {**shared.TRAINING, "scope": "all", "epochs": args.epochs,
                "max_steps": args.max_steps, "patience": args.patience}
    shared.run(args, protocol=GENERAL_PROTOCOL, training=training, seeds=args.seeds)


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode if error.returncode > 0 else 1,
                    "Run failed; remaining jobs were not started. Inspect logs/ and rerun after fixing the error.\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Rerun to resume matching checkpoints.\n")


if __name__ == "__main__":
    main()
