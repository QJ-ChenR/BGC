"""Train only the aggregator, with parent-balanced sampling and resumable state."""

import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .cache import feature_contract, load_cache, save_tensor_file
from .common import stable_seed, write_json
from .data import (ParentDataset, collate, predict_batch, select_entries,
                   teacher_statistics, to_device)
from .encoder import select_device
from .evaluate import load_checkpoint, reconstruct, summarize
from .model import ChunkAggregator, ModelConfig, reconstruction_loss


TRAINING_KEYS = ("epochs", "batch_size", "learning_rate", "weight_decay", "warmup_fraction",
                 "patience", "cosine_weight", "seed", "scope", "sources", "gradient_clip")


def train(args):
    cache = load_cache(args.cache)
    device = select_device(args.device)
    output = Path(args.output)
    if args.resume and args.init_from:
        raise ValueError("Use either --resume or --init-from, not both")
    if args.resume:
        if Path(args.resume).resolve().parent != output.resolve() or not (output / "best.pt").is_file():
            raise ValueError("Resume in the original run directory containing best.pt")
        model, saved = load_checkpoint(args.resume, cache, device)
        config = saved["training_config"]
        print("Resuming saved model, optimizer, scheduler, RNG, and training settings.")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use a new output directory, or --resume to continue an existing run")
        config = {key: getattr(args, key) for key in TRAINING_KEYS}
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"])
        specification = ModelConfig(
            embedding_dim=cache["encoder"]["embedding_dim"], hidden_dim=args.hidden_dim,
            num_layers=args.layers, num_heads=args.heads, feedforward_dim=args.feedforward_dim,
            dropout=args.dropout, architecture=args.architecture, use_positions=not args.no_positions)
        model = ChunkAggregator(specification).to(device)
        if args.init_from:
            initialized, initial = load_checkpoint(args.init_from, cache, device)
            if initial["model_config"] != asdict(specification):
                raise ValueError("Fine-tuning requires the same model architecture as the initial checkpoint")
            model.load_state_dict(initialized.state_dict())
            del initialized
        saved = None
    if (min(config["epochs"], config["batch_size"], config["patience"]) < 1
            or config["learning_rate"] <= 0 or config["weight_decay"] < 0
            or not 0 <= config["warmup_fraction"] < 1
            or config["cosine_weight"] < 0 or config["gradient_clip"] <= 0):
        raise ValueError("Invalid training hyperparameters")
    training_entries = select_entries(cache, "train", config["scope"], config["sources"])
    # Model selection always targets NRPS/PKS, including when pretraining on all families.
    validation_entries = select_entries(cache, "validation", "target", config["sources"])
    dataset = ParentDataset(args.cache, training_entries, config["seed"])
    statistics = saved["teacher_statistics"] if saved else teacher_statistics(args.cache, training_entries)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    steps = math.ceil(len(dataset) / config["batch_size"]) * config["epochs"]
    warmup = int(steps * config["warmup_fraction"])

    def schedule(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start_epoch, best_loss, stale, history = 0, float("inf"), 0, []
    if saved:
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda" and saved["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        start_epoch, best_loss = saved["epoch"], saved["best_loss"]
        stale, history = saved["stale_epochs"], saved["history"]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run.json", {"training": config, "model": model.specification(),
               "cache_id": cache["cache_id"], "feature_contract": feature_contract(cache),
               "training_parents": len(dataset), "torch_version": str(torch.__version__),
               "device": str(device), "cuda_version": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "initial_checkpoint": str(args.init_from) if args.init_from else None})

    def checkpoint(epoch):
        return {"format_version": 1, "model_config": model.specification(),
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "teacher_statistics": statistics,
                "feature_contract": feature_contract(cache), "cache_id": cache["cache_id"],
                "training_config": config, "epoch": epoch, "best_loss": best_loss,
                "stale_epochs": stale, "history": history,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}

    def validate():
        rows, _ = reconstruct(model, args.cache, validation_entries, statistics, device,
                              config["batch_size"], config["cosine_weight"])
        return summarize(rows)

    if not saved:
        initial_metrics = validate()
        best_loss = initial_metrics["overall"]["model_loss"]
        history.append({"epoch": 0, "validation": initial_metrics})
        save_tensor_file(output / "best.pt", checkpoint(0))
        save_tensor_file(output / "last.pt", checkpoint(0))
        write_json(output / "history.json", history)
        print(f"Initial target validation loss: {best_loss:.6f}", flush=True)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}; "
          f"training parents: {len(dataset)}; device: {device}", flush=True)
    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        if stale >= config["patience"]:
            print("Early stopping criterion already reached.")
            break
        dataset.epoch = epoch
        generator = torch.Generator().manual_seed(stable_seed(config["seed"], epoch))
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                            collate_fn=collate, generator=generator, num_workers=0)
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = predict_batch(model, batch)
            loss = reconstruction_loss(prediction, batch["teacher"], statistics["variance"],
                                       config["cosine_weight"]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Training loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            size = len(prediction)
            total += loss.item() * size
            count += size
        metrics = validate()
        validation_loss = metrics["overall"]["model_loss"]
        improved = validation_loss < best_loss
        if improved:
            best_loss, stale = validation_loss, 0
        else:
            stale += 1
        history.append({"epoch": epoch, "training_loss": total / count, "validation": metrics,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if improved:
            save_tensor_file(output / "best.pt", checkpoint(epoch))
        save_tensor_file(output / "last.pt", checkpoint(epoch))
        write_json(output / "history.json", history)
        print(f"Epoch {epoch}: train={total / count:.6f}, validation={validation_loss:.6f}, "
              f"weighted_mean={metrics['overall']['baseline_loss']:.6f}, best={best_loss:.6f}", flush=True)
    print(f"Best checkpoint: {output / 'best.pt'}")
    print("The test split was not used for training or model selection.")
