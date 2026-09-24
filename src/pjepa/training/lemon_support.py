"""LEMON deterministic masks, loaders, and checkpoint serialization."""

from __future__ import annotations
import copy
import hashlib
import os
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader
from pjepa.data.lemon_sharded import LengthBucketBatchSampler
from pjepa.models.feature_vjepa_rope import FeatureJEPA

try:
    import wandb
except Exception:
    wandb = None


@torch.no_grad()
def _ema_update(teacher: torch.nn.Module, student: torch.nn.Module, momentum: float) -> None:
    teacher_parameters = dict(teacher.named_parameters())
    for name, student_parameter in student.named_parameters():
        teacher_parameters[name].lerp_(student_parameter, 1.0 - float(momentum))
    teacher_buffers = dict(teacher.named_buffers())
    for name, student_buffer in student.named_buffers():
        if name in teacher_buffers:
            teacher_buffers[name].copy_(student_buffer)


def _deterministic_masks(
    valid_mask: torch.Tensor,
    sample_names: list[str],
    *,
    mask_ratio: float,
    base_seed: int,
    repeat: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros_like(valid_mask, dtype=torch.bool)
    for batch_idx, sample_name in enumerate(sample_names):
        valid_indices = torch.nonzero(valid_mask[batch_idx], as_tuple=False).flatten()
        valid_length = int(valid_indices.numel())
        if valid_length <= 1:
            continue
        num_masked = min(valid_length - 1, max(1, int(round(valid_length * float(mask_ratio)))))
        digest = hashlib.sha256(f"{base_seed}:{repeat}:{sample_name}".encode("utf-8")).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], byteorder="little", signed=False))
        selected = valid_indices[torch.randperm(valid_length, generator=generator)[:num_masked]]
        target[batch_idx, selected] = True
    context = valid_mask.bool() & ~target
    return (context, target)


@torch.no_grad()
def validate_masked_l1(
    model: FeatureJEPA,
    teacher: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    mask_ratio: float,
    seed: int,
    mask_repeats: int = 1,
    use_bfloat16: bool = False,
) -> dict[str, float]:
    model_was_training = bool(model.training)
    teacher_was_training = bool(teacher.training)
    model.eval()
    teacher.eval()
    absolute_error = 0.0
    value_count = 0
    masked_tokens = 0
    sequences = 0
    autocast_enabled = bool(use_bfloat16 and device.type == "cuda")
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].bool()
        sample_names = [str(value) for value in batch["sample_name"]]
        for repeat in range(int(mask_repeats)):
            (context_mask, target_mask) = _deterministic_masks(
                valid_mask, sample_names, mask_ratio=mask_ratio, base_seed=seed, repeat=repeat
            )
            if not bool(target_mask.any().item()):
                continue
            target_mask = target_mask.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled
            ):
                output = model(
                    {
                        "x": features.unsqueeze(1),
                        "valid_mask": valid_mask.to(device, non_blocking=True).unsqueeze(1),
                        "context_mask": context_mask.to(device, non_blocking=True).unsqueeze(1),
                        "target_mask": target_mask.unsqueeze(1),
                    },
                    teacher=teacher,
                )
            difference = (
                output["preds_at_targets"].float() - output["teacher_at_targets"].float()
            ).abs()
            absolute_error += float(difference.sum().item())
            value_count += int(difference.numel())
            masked_tokens += int(target_mask.sum().item())
        sequences += int(features.shape[0])
    if model_was_training:
        model.train()
    if teacher_was_training:
        teacher.train()
    return {
        "l1": absolute_error / max(1, value_count),
        "masked_tokens": float(masked_tokens),
        "values": float(value_count),
        "sequences": float(sequences),
        "mask_repeats": float(mask_repeats),
    }


def _make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    loader_cfg: dict[str, Any],
    seed: int,
    collate_fn=None,
    length_bucketed: bool = False,
) -> DataLoader:
    num_workers = int(loader_cfg.get("num_workers", 0))
    kwargs = {
        "dataset": dataset,
        "num_workers": num_workers,
        "pin_memory": bool(loader_cfg.get("pin_memory", torch.cuda.is_available())),
        "collate_fn": collate_fn,
    }
    if length_bucketed:
        kwargs["batch_sampler"] = LengthBucketBatchSampler(
            dataset.sequence_lengths,
            batch_size=int(batch_size),
            seed=int(seed),
            bucket_size_multiplier=int(loader_cfg.get("bucket_size_multiplier", 50)),
            drop_last=False,
        )
    else:
        kwargs.update(
            {
                "batch_size": int(batch_size),
                "shuffle": bool(shuffle),
                "drop_last": False,
                "generator": torch.Generator().manual_seed(int(seed)),
            }
        )
    if num_workers > 0:
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = int(loader_cfg.get("prefetch_factor", 2))
        multiprocessing_context = loader_cfg.get("multiprocessing_context")
        if multiprocessing_context:
            kwargs["multiprocessing_context"] = str(multiprocessing_context)
    return DataLoader(**kwargs)


def _atomic_save_best(
    path: Path,
    *,
    model: FeatureJEPA,
    config: dict[str, Any],
    epoch: int,
    probe_result: dict[str, Any],
    cholec80_l1: dict[str, float] | None,
) -> None:
    payload = {
        "version": "lemon_pjepa_best_v2",
        "model_state_dict": model.state_dict(),
        "config": copy.deepcopy(config),
        "epoch": int(epoch),
        "selection_metric": "cholec80_val_phase_acc",
        "selection_value": float(probe_result["metrics"]["phase_acc"]),
        "probe_metrics": dict(probe_result["metrics"]),
        "cholec80_masked_l1": None if cholec80_l1 is None else dict(cholec80_l1),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
