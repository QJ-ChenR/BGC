#!/usr/bin/env python3
"""Run steps 5 and 6 of the fixed-view length-extrapolation experiment.

Usage: python script/run_length_extrapolation.py
Uses the active Python environment. Relative paths are resolved from the project
root. All six training runs must succeed before any test evaluation starts.
Matching existing runs resume from last.pt; logs append across restarts.
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("transformer", "mlp")
SEEDS = (42, 43, 44)
TRAINING = {"scope": "target", "sources": "native", "views": "fixed",
            "epochs": 100, "batch_size": 64, "learning_rate": 1e-4, "patience": 10,
            "weight_decay": 0.01, "warmup_fraction": 0.05, "cosine_weight": 0.5,
            "gradient_clip": 1.0}
MODEL_OPTIONS = {"hidden_dim": 128, "layers": 2, "heads": 4,
                 "feedforward_dim": 512, "dropout": 0.1}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=Path("data/embeddings/length1500_no_mmseqs"))
    parser.add_argument("--prefix", default="length1500_no_mmseqs",
                        help="Filename prefix for runs/, results/, and logs/")
    parser.add_argument("--device", default="cuda", help="For example cuda, cuda:0, or cpu")
    parser.add_argument("--stage", choices=("all", "train", "evaluate"), default="all",
                        help="Use train then evaluate separately to inspect short validation results first")
    parser.add_argument("--no-plots", action="store_true", help="Skip SVG plots and the matplotlib dependency")
    return parser


def read_json(path):
    return json.loads(Path(path).read_text())


def matching_run(directory, cache, architecture, seed):
    """Do not resume a different experiment under a familiar directory name."""
    if not directory.exists() or not any(directory.iterdir()):
        return False
    required = ("run.json", "best.pt", "last.pt")
    if not all((directory / name).is_file() for name in required):
        raise ValueError(f"Incomplete run directory: {directory}. Preserve it and use a new --prefix, "
                         "or restore run.json, best.pt, and last.pt before resuming.")
    metadata = read_json(directory / "run.json")
    expected = {**TRAINING, "seed": seed}
    expected_model = {"architecture": architecture, "embedding_dim": cache["encoder"]["embedding_dim"],
                      "hidden_dim": MODEL_OPTIONS["hidden_dim"], "num_layers": MODEL_OPTIONS["layers"],
                      "num_heads": MODEL_OPTIONS["heads"], "feedforward_dim": MODEL_OPTIONS["feedforward_dim"],
                      "dropout": MODEL_OPTIONS["dropout"], "use_positions": True}
    if (metadata.get("cache_id") != cache["cache_id"]
            or metadata.get("experiment") != cache["experiment"]
            or metadata.get("model") != expected_model
            or any(metadata.get("training", {}).get(key) != value for key, value in expected.items())
            or metadata.get("initial_checkpoint") is not None):
        raise ValueError(f"Run settings differ from this experiment: {directory}. Use a new --prefix.")
    return True


def run_logged(command, log_path):
    """Mirror combined output to the terminal and an append-only log; propagate errors."""
    print(f"\n$ {shlex.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] {shlex.join(command)}\n")
        log.flush()
        with subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as child:
            try:
                for line in child.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
                status = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        if status:
            raise subprocess.CalledProcessError(status, command)


def run(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.prefix):
        raise ValueError("--prefix must be a filename prefix using letters, numbers, '.', '_', or '-'")
    cache_path = args.cache if args.cache.is_absolute() else PROJECT_ROOT / args.cache
    cache_path = cache_path.resolve()
    cache = read_json(cache_path / "cache.json")
    if not cache.get("complete"):
        raise ValueError("The feature cache is incomplete. Finish step 4 before running this script.")
    if cache.get("experiment", {}).get("protocol") != "length-extrapolation":
        raise ValueError("Use a cache prepared with --protocol length-extrapolation")
    if args.stage != "train" and not args.no_plots:
        import importlib.util
        if importlib.util.find_spec("matplotlib") is None:
            raise ValueError("Plots require matplotlib. Install requirements-server.txt or use --no-plots.")

    jobs = []
    # Check every destination before starting a potentially long series of runs.
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            name = f"{args.prefix}_{architecture}_s{seed}"
            directory = PROJECT_ROOT / "runs" / name
            resume = matching_run(directory, cache, architecture, seed)
            if args.stage == "evaluate" and not resume:
                raise ValueError(f"No matching trained run: {directory}. Run --stage train first.")
            jobs.append((architecture, seed, name, directory, resume))

    logs = PROJECT_ROOT / "logs"
    logs.mkdir(exist_ok=True)
    base = [sys.executable, "-u", "-m", "bgc_aggregation"]
    common = ["--cache", str(cache_path), "--device", args.device]
    print(f"Python: {sys.executable}\nCache: {cache_path}\nStage: {args.stage}", flush=True)
    if args.stage in {"all", "train"}:
        for architecture, seed, name, directory, resume in jobs:
            command = [*base, "train", *common, "--architecture", architecture,
                       "--seed", str(seed), "--output", str(directory)]
            for key, value in {**TRAINING, **MODEL_OPTIONS}.items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            if resume:
                command.extend(["--resume", str(directory / "last.pt")])
            run_logged(command, logs / f"{name}.log")
    if args.stage in {"all", "evaluate"}:
        for _, _, name, directory, _ in jobs:
            command = [*base, "evaluate", *common, "--checkpoint", str(directory / "best.pt"),
                       "--split", "test", "--scope", "target", "--sources", "native",
                       "--bootstrap-replicates", "2000", "--bootstrap-seed", "42",
                       "--output", str(PROJECT_ROOT / "results" / name)]
            if not args.no_plots:
                command.append("--plots")
            run_logged(command, logs / f"{name}_evaluation.log")
    print(f"Completed {args.stage}. Logs: {logs}", flush=True)


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
