"""Train only the aggregator, with parent-balanced sampling and resumable state."""

import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .cache import feature_contract, load_cache, save_tensor_file
from .common import stable_seed, write_json
from .data import (ParentDataset, collate, input_summary, predict_batch, select_entries,
                   teacher_statistics, to_device)
from .encoder import select_device
from .evaluate import load_checkpoint, reconstruct, summarize
from .model import ChunkAggregator, ModelConfig, reconstruction_loss
from .protocol import is_general_experiment, is_length_experiment


TRAINING_KEYS = ("epochs", "batch_size", "learning_rate", "weight_decay", "warmup_fraction",
                 "patience", "cosine_weight", "seed", "scope", "sources", "gradient_clip", "views", "max_steps")


def train(args):
    cache = load_cache(args.cache)
    length_experiment = is_length_experiment(cache)
    general_experiment = is_general_experiment(cache)
    # Length experiments isolate extrapolation from pretrained aggregator exposure;
    # only the frozen ESMC encoder starts with pretrained weights.
    if length_experiment and args.init_from:
        raise ValueError("Length-extrapolation starts from random aggregator weights; --init-from is not allowed")
    device = select_device(args.device)
    output = Path(args.output)
    if args.resume and args.init_from:
        raise ValueError("Use either --resume or --init-from, not both")
    if args.resume:
        if Path(args.resume).resolve().parent != output.resolve() or not (output / "best.pt").is_file():
            raise ValueError("Resume in the original run directory containing best.pt")
        model, saved = load_checkpoint(args.resume, cache, device)
        # Resume restores the original experiment budget and sampling settings,
        # even when this invocation supplies different command-line defaults.
        config = dict(saved["training_config"])
        config.setdefault("views", "multiscale")
        config.setdefault("max_steps", None)
        if length_experiment and saved.get("experiment") != cache["experiment"]:
            raise ValueError("Checkpoint length-extrapolation protocol differs from the cache")
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
        # Standard fine-tuning copies model weights but starts a fresh optimizer.
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
    if config.get("max_steps") is not None and config["max_steps"] < 1:
        raise ValueError("max_steps must be positive")
    if general_experiment and config["scope"] != "all":
        raise ValueError("General protein training requires --scope all")
    config["validation_scope"] = "all" if general_experiment else "target"
    config["views"] = config.get("views") or ("fixed" if length_experiment else "multiscale")
    if length_experiment:
        if config["sources"] == "crop":
            raise ValueError("Length-extrapolation accepts only native proteins")
        config["sources"] = "native"
        if config["views"] == "multiscale" and cache["experiment"]["chunk_views"] != "multiscale":
            raise ValueError("Multiscale training requires a cache prepared with --chunk-views multiscale")
    # Separate parameter-update data from checkpoint-selection data before computing
    # statistics or building batches. The test split is never selected here.
    training_entries = select_entries(cache, "train", config["scope"], config["sources"])
    # General data use general short validation; legacy BGC runs still target NRPS/PKS.
    validation_entries = select_entries(cache, "validation", config["validation_scope"], config["sources"])
    dataset = ParentDataset(args.cache, training_entries, config["seed"], config["views"])
    training_summary = input_summary(args.cache, training_entries, config["views"], strict=length_experiment)
    validation_summary = input_summary(args.cache, validation_entries, "fixed", strict=length_experiment)
    # Reuse training-only normalization on resume and in every subsequent evaluation.
    statistics = saved["teacher_statistics"] if saved else teacher_statistics(args.cache, training_entries)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    # An update cap makes the training budget explicit when dataset size changes;
    # warmup and cosine decay must use the same capped number of optimizer steps.
    steps_per_epoch = math.ceil(len(dataset) / config["batch_size"])
    steps = steps_per_epoch * config["epochs"]
    if config["max_steps"] is not None:
        steps = min(steps, config["max_steps"])
    warmup = int(steps * config["warmup_fraction"])

    # Ramp up the learning rate, then decay smoothly to zero at the planned budget.
    def schedule(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start_epoch, best_loss, stale, history = 0, float("inf"), 0, []
    global_step = 0
    if saved:
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        # Restore dropout RNG as well as optimizer/scheduler state for reproducible continuation.
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda" and saved["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        start_epoch, best_loss = saved["epoch"], saved["best_loss"]
        stale, history = saved["stale_epochs"], saved["history"]
        global_step = saved.get("global_step", start_epoch * steps_per_epoch)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run.json", {"training": config, "model": model.specification(),
               "cache_id": cache["cache_id"], "feature_contract": feature_contract(cache),
               "planned_optimizer_steps": steps, "steps_per_epoch": steps_per_epoch,
               "training_parents": len(dataset), "training_inputs": training_summary,
               "validation_inputs": validation_summary, "experiment": cache.get("experiment"),
               "torch_version": str(torch.__version__),
               "device": str(device), "cuda_version": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "initial_checkpoint": str(args.init_from) if args.init_from else None})

    # Save both inference weights and all state needed to continue at an epoch boundary.
    def checkpoint(epoch):
        return {"format_version": 1, "model_config": model.specification(),
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "teacher_statistics": statistics,
                "feature_contract": feature_contract(cache), "cache_id": cache["cache_id"],
                "training_config": config, "experiment": cache.get("experiment"),
                "training_inputs": training_summary, "validation_inputs": validation_summary,
                "epoch": epoch, "global_step": global_step, "planned_optimizer_steps": steps, "best_loss": best_loss,
                "stale_epochs": stale, "history": history,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}

    # Evaluate the fixed deployment view without gradient updates; its reconstruction
    # loss drives early stopping, regardless of augmented training views.
    def validate():
        rows, _ = reconstruct(model, args.cache, validation_entries, statistics, device,
                              config["batch_size"], config["cosine_weight"])
        return summarize(rows)

    # A fresh aggregator starts at the weighted-mean baseline; fine-tuning
    # starts at the imported weights. Keep either initial state eligible
    # as best if subsequent updates only worsen validation.
    if not saved:
        initial_metrics = validate()
        best_loss = initial_metrics["overall"]["model_loss"]
        history.append({"epoch": 0, "global_step": 0, "validation": initial_metrics})
        save_tensor_file(output / "best.pt", checkpoint(0))
        save_tensor_file(output / "last.pt", checkpoint(0))
        write_json(output / "history.json", history)
        print(f"Initial {config['validation_scope']} validation loss: {best_loss:.6f}", flush=True)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}; "
          f"training parents: {len(dataset)}; device: {device}", flush=True)
    print(f"Optimizer budget: {steps} updates; {steps_per_epoch} updates per full epoch.", flush=True)
    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        if global_step >= steps:
            print("Optimizer update budget reached.")
            break
        if stale >= config["patience"]:
            print("Early stopping criterion already reached.")
            break
        dataset.epoch = epoch
        # Shuffling is keyed by epoch; resuming at a saved boundary reconstructs
        # the same order instead of consuming a process-global random stream.
        generator = torch.Generator().manual_seed(stable_seed(config["seed"], epoch))
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                            collate_fn=collate, generator=generator, num_workers=0)
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            batch = to_device(batch, device)
            # Only the small aggregator participates in backpropagation; ESMC features
            # and full-sequence teachers are already frozen tensors in the cache.
            optimizer.zero_grad(set_to_none=True)
            prediction = predict_batch(model, batch)
            loss = reconstruction_loss(prediction, batch["teacher"], statistics["variance"],
                                       config["cosine_weight"]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Training loss became non-finite")
            loss.backward()
            # Bound unusually large updates before stepping the optimizer.
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            global_step += 1
            size = len(prediction)
            total += loss.item() * size
            count += size
            # A cap may stop midway through an epoch; still validate and checkpoint it.
            if global_step >= steps:
                break
        metrics = validate()
        # best.pt follows validation performance; last.pt stores the latest resumable state.
        validation_loss = metrics["overall"]["model_loss"]
        improved = validation_loss < best_loss
        if improved:
            best_loss, stale = validation_loss, 0
        else:
            stale += 1
        history.append({"epoch": epoch, "global_step": global_step, "training_loss": total / count, "validation": metrics,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if improved:
            save_tensor_file(output / "best.pt", checkpoint(epoch))
        save_tensor_file(output / "last.pt", checkpoint(epoch))
        write_json(output / "history.json", history)
        print(f"Epoch {epoch}: train={total / count:.6f}, validation={validation_loss:.6f}, "
              f"weighted_mean={metrics['overall']['baseline_loss']:.6f}, best={best_loss:.6f}, "
              f"step={global_step}/{steps}", flush=True)
    print(f"Best checkpoint: {output / 'best.pt'}")
    print("The test split was not used for training or model selection.")
