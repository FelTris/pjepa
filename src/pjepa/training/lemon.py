"""LEMON SSL training with Cholec80 development checkpoint selection."""

from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import torch
from pjepa.data.lemon_sharded import LemonVideoDataset, collate_lemon_videos
from pjepa.data.surgical_phase_plstitch import (
    SurgicalPhaseVideoDataset,
    collate_surgical_phase_videos,
)
from pjepa.models.mask import make_random_masks_from_valid_idx
from pjepa.utils.runtime import ensure_dir, resolve_device, resolve_path, seed_everything
from pjepa.probes.surgical_phase_linear_probe import run_frozen_phase_probe

try:
    import wandb
except Exception:
    wandb = None
from pjepa.models.factory import build_model
from pjepa.training.lemon_support import (
    _ema_update,
    validate_masked_l1,
    _make_loader,
    _atomic_save_best,
)


def train(config: dict[str, Any]) -> dict[str, Any]:
    runtime_cfg = config.get("runtime", {})
    paths_cfg = config["paths"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    ssl_cfg = config["ssl"]
    train_cfg = config["train"]
    validation_cfg = config["validation"]
    probe_cfg = config["probe"]
    if str(probe_cfg.get("selection_metric", "phase_acc")) != "phase_acc":
        raise ValueError("LEMON checkpoint selection must use Cholec80 validation phase_acc.")
    loader_cfg = config.get("loader", {})
    wandb_cfg = config.get("wandb", {})
    seed = seed_everything(
        int(runtime_cfg.get("seed", 1538574472)), bool(runtime_cfg.get("deterministic_cudnn", True))
    )
    device = resolve_device(runtime_cfg.get("device", "auto"))
    lemon_dataset = LemonVideoDataset(
        resolve_path(paths_cfg["lemon_index"]),
        split=str(data_cfg.get("lemon_split", "pretrain")),
        sample_rate=int(data_cfg.get("sample_rate", 4)),
        max_sequence_len=int(data_cfg.get("max_sequence_len", 4096)),
        seed=seed,
        max_cached_shards=int(loader_cfg.get("max_cached_shards", 1)),
        mmap=bool(loader_cfg.get("mmap", True)),
    )
    expected_dim = int(model_cfg.get("input_dim", 768))
    if lemon_dataset.archive.feature_dim != expected_dim:
        raise ValueError(
            f"LEMON feature dim {lemon_dataset.archive.feature_dim} != model input dim {expected_dim}."
        )
    cholec80_archive = resolve_path(paths_cfg["cholec80_archive"])
    cholec80_protocol = str(data_cfg.get("cholec80_protocol", "development"))
    cholec80_train_dataset = SurgicalPhaseVideoDataset(
        cholec80_archive, dataset="cholec80", protocol=cholec80_protocol, split="train"
    )
    cholec80_val_dataset = SurgicalPhaseVideoDataset(
        cholec80_archive, dataset="cholec80", protocol=cholec80_protocol, split="val"
    )
    if cholec80_train_dataset.archive.feature_dim != expected_dim:
        raise ValueError("Cholec80 and model input feature dimensions do not match.")
    for metadata_key in (
        "checkpoint_sha256",
        "pl_stitch_git_commit",
        "preprocessing",
        "feature_type",
    ):
        lemon_value = lemon_dataset.archive.index.get(metadata_key)
        cholec80_value = cholec80_val_dataset.archive.payload.get(metadata_key)
        if lemon_value != cholec80_value:
            raise ValueError(
                f"LEMON/Cholec80 PL-Stitch metadata mismatch for {metadata_key}: {lemon_value!r} vs {cholec80_value!r}."
            )
    if not math.isclose(
        lemon_dataset.effective_fps,
        cholec80_val_dataset.archive.target_fps,
        rel_tol=0.0,
        abs_tol=1e-06,
    ):
        raise ValueError(
            f"LEMON effective FPS must match Cholec80 target FPS for 1D RoPE transfer: {lemon_dataset.effective_fps} vs {cholec80_val_dataset.archive.target_fps}."
        )
    attention = str(ssl_cfg.get("student_encoder_attention", "block_causal"))
    block_size = int(ssl_cfg.get("student_encoder_block_size", 32))
    if (
        attention == "block_causal"
        and int(data_cfg.get("max_sequence_len", 4096)) % block_size != 0
    ):
        raise ValueError(
            "For stable video-aligned boundaries, max_sequence_len must be divisible by block size."
        )
    if str(ssl_cfg.get("rope_mode", "1d_flat")) != "1d_flat":
        raise ValueError("LEMON training requires ssl.rope_mode='1d_flat'.")
    train_loader = _make_loader(
        lemon_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        loader_cfg=loader_cfg.get("lemon", loader_cfg),
        seed=seed,
        collate_fn=collate_lemon_videos,
        length_bucketed=True,
    )
    cholec80_val_loader = _make_loader(
        cholec80_val_dataset,
        batch_size=int(validation_cfg.get("batch_size", 1)),
        shuffle=False,
        loader_cfg=loader_cfg.get("surgical_phase", loader_cfg),
        seed=seed,
        collate_fn=collate_surgical_phase_videos,
    )
    probe_train_loader = _make_loader(
        cholec80_train_dataset,
        batch_size=int(probe_cfg.get("extraction_batch_size", 1)),
        shuffle=False,
        loader_cfg=loader_cfg.get("surgical_phase", loader_cfg),
        seed=seed,
        collate_fn=collate_surgical_phase_videos,
    )
    probe_val_loader = _make_loader(
        cholec80_val_dataset,
        batch_size=int(probe_cfg.get("extraction_batch_size", 1)),
        shuffle=False,
        loader_cfg=loader_cfg.get("surgical_phase", loader_cfg),
        seed=seed,
        collate_fn=collate_surgical_phase_videos,
    )
    model = build_model(model_cfg, ssl_cfg).to(device)
    teacher = model.build_teacher().to(device)
    teacher.eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 0.0001)),
        weight_decay=float(train_cfg.get("weight_decay", 0.04)),
    )
    milestones = [int(value) for value in train_cfg.get("lr_milestones", [])]
    scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=float(train_cfg.get("lr_gamma", 0.1))
        )
        if milestones
        else None
    )
    run_name = str(config.get("experiment", {}).get("run_name", "lemon_pjepa"))
    checkpoint_dir = Path(ensure_dir(resolve_path(paths_cfg["ckpt_dir"]))) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    best_phase_acc = float("-inf")
    use_bfloat16 = bool(train_cfg.get("use_bfloat16", False))
    autocast_enabled = bool(use_bfloat16 and device.type == "cuda")
    accumulation_steps = int(ssl_cfg.get("accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("ssl.accumulation_steps must be positive.")
    wandb_run = None
    if bool(wandb_cfg.get("use_wandb", True)) and wandb is not None:
        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "pjepa")),
            entity=wandb_cfg.get("entity"),
            name=run_name,
            mode=wandb_cfg.get("mode"),
            config=config,
        )
    print(
        f"LEMON: {len(lemon_dataset.video_indices)} videos, {lemon_dataset.total_effective_tokens} tokens at {lemon_dataset.effective_fps:.3f} fps, {lemon_dataset.tokens_per_epoch} retained tokens/epoch, {lemon_dataset.num_cropped_videos} cropped videos"
    )
    print(
        f"Cholec80 {cholec80_protocol}: {len(cholec80_train_dataset.video_ids)} train / {len(cholec80_val_dataset.video_ids)} val videos"
    )
    print(
        f"Trainable P-JEPA parameters: {sum((p.numel() for p in model.parameters() if p.requires_grad))}"
    )
    raw_probe_result = None
    if bool(probe_cfg.get("run_raw_baseline", True)):
        raw_probe_result = run_frozen_phase_probe(
            None,
            probe_train_loader,
            probe_val_loader,
            device=device,
            config=probe_cfg,
            num_classes=cholec80_train_dataset.archive.num_classes,
            feature_source="raw",
            select_best_on_eval=True,
            log_prefix="Cholec80 raw",
        )
        raw_metrics = raw_probe_result["metrics"]
        print(
            f"[Cholec80 raw baseline] acc={raw_metrics['phase_acc']:.6f} macro_f1={raw_metrics['phase_macro_f1']:.6f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {f"cholec80_raw_probe/{name}": value for (name, value) in raw_metrics.items()},
                step=0,
            )
    latest_l1 = None
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    num_epochs = int(train_cfg["num_epochs"])
    for epoch in range(num_epochs):
        lemon_dataset.set_epoch(epoch)
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        teacher.eval()
        total_loss = 0.0
        loss_batches = 0
        micro_batches = 0
        for batch_index, batch in enumerate(train_loader):
            features = batch["features"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True).bool()
            (context_mask, target_mask) = make_random_masks_from_valid_idx(
                valid_mask.unsqueeze(1), mask_ratio=float(ssl_cfg.get("mask_ratio", 0.6))
            )
            if not bool(target_mask.any().item()):
                continue
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled
            ):
                output = model(
                    {
                        "x": features.unsqueeze(1),
                        "valid_mask": valid_mask.unsqueeze(1),
                        "context_mask": context_mask,
                        "target_mask": target_mask,
                    },
                    teacher=teacher,
                )
                loss = (
                    (output["preds_at_targets"].float() - output["teacher_at_targets"].float())
                    .abs()
                    .mean()
                )
            (loss / accumulation_steps).backward()
            total_loss += float(loss.detach().item())
            loss_batches += 1
            micro_batches += 1
            is_last_batch = batch_index + 1 == len(train_loader)
            if micro_batches >= accumulation_steps or is_last_batch:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_cfg.get("clip_grad_norm", 1.0))
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                _ema_update(teacher, model.student_enc, float(ssl_cfg.get("ema_momentum", 0.999)))
                micro_batches = 0
                global_step += 1
        if micro_batches:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(train_cfg.get("clip_grad_norm", 1.0))
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _ema_update(teacher, model.student_enc, float(ssl_cfg.get("ema_momentum", 0.999)))
            global_step += 1
        if scheduler is not None:
            scheduler.step()
        epoch_metrics = {
            "epoch": epoch + 1,
            "train/l1": total_loss / max(1, loss_batches),
            "train/lr": float(optimizer.param_groups[0]["lr"]),
            "train/global_step": global_step,
        }
        print(
            f"[LEMON epoch {epoch + 1:03d}] l1={epoch_metrics['train/l1']:.6f} lr={epoch_metrics['train/lr']:.6g}"
        )
        validation_every = int(validation_cfg.get("every", 1))
        if validation_every > 0 and (
            (epoch + 1) % validation_every == 0 or epoch + 1 == num_epochs
        ):
            latest_l1 = validate_masked_l1(
                model,
                teacher,
                cholec80_val_loader,
                device=device,
                mask_ratio=float(ssl_cfg.get("mask_ratio", 0.6)),
                seed=int(validation_cfg.get("mask_seed", seed)),
                mask_repeats=int(validation_cfg.get("mask_repeats", 1)),
                use_bfloat16=use_bfloat16,
            )
            epoch_metrics.update(
                {f"cholec80_ssl/{name}": value for (name, value) in latest_l1.items()}
            )
            print(
                f"[Cholec80 masked validation {epoch + 1:03d}] l1={latest_l1['l1']:.6f} masked_tokens={int(latest_l1['masked_tokens'])}"
            )
        probe_every = int(probe_cfg.get("every", 10))
        should_probe = probe_every > 0 and (
            (epoch + 1) % probe_every == 0 or epoch + 1 == num_epochs
        )
        if should_probe:
            probe_result = run_frozen_phase_probe(
                model,
                probe_train_loader,
                probe_val_loader,
                device=device,
                config=probe_cfg,
                num_classes=cholec80_train_dataset.archive.num_classes,
                feature_source="student",
                select_best_on_eval=True,
                log_prefix="Cholec80 student",
            )
            probe_metrics = probe_result["metrics"]
            epoch_metrics.update(
                {f"cholec80_probe/{name}": value for (name, value) in probe_metrics.items()}
            )
            phase_acc = float(probe_metrics["phase_acc"])
            if phase_acc > best_phase_acc:
                best_phase_acc = phase_acc
                _atomic_save_best(
                    best_path,
                    model=model,
                    config=config,
                    epoch=epoch + 1,
                    probe_result=probe_result,
                    cholec80_l1=latest_l1,
                )
                print(
                    f"Saved new single best checkpoint: {best_path} (cholec80_val_phase_acc={best_phase_acc:.6f})"
                )
        if wandb_run is not None:
            wandb.log(epoch_metrics, step=epoch + 1)
    if wandb_run is not None:
        wandb.finish()
    if not best_path.exists():
        raise RuntimeError("Training finished without a probe-selected checkpoint.")
    return {
        "best_checkpoint": str(best_path),
        "best_cholec80_phase_acc": best_phase_acc,
        "last_cholec80_l1": latest_l1,
        "raw_cholec80_probe": None if raw_probe_result is None else raw_probe_result["metrics"],
    }
