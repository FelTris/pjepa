from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from pjepa.data.surgical_phase_plstitch import (
    SurgicalPhaseVideoDataset,
    collate_surgical_phase_videos,
)
from pjepa.probes.surgical_phase_linear_probe import SurgicalPhaseLinearHead
from pjepa.models.factory import build_model
from pjepa.utils.runtime import load_json, resolve_device, resolve_path, seed_everything


DEFAULT_CONTEXTS = (0, 1, 2, 4, 8, 16, 32, 64, None)


def parse_contexts(values: list[str] | None) -> list[int | None]:
    if not values:
        return list(DEFAULT_CONTEXTS)
    parsed: list[int | None] = []
    for value in values:
        normalized = str(value).strip().lower()
        context = None if normalized in {"full", "unbounded", "none"} else int(normalized)
        if context is not None and context < 0:
            raise ValueError("Context block budgets must be non-negative.")
        if context not in parsed:
            parsed.append(context)
    if not parsed:
        raise ValueError("At least one context budget is required.")
    return parsed


def context_name(max_past_blocks: int | None) -> str:
    return "full" if max_past_blocks is None else str(int(max_past_blocks))


def block_window_specs(
    total_tokens: int,
    block_size: int,
    max_past_blocks: int,
) -> list[tuple[int, int, int, int]]:
    """Return ``(start, end, target_start, target_end)`` for every query block."""
    total_tokens = int(total_tokens)
    block_size = int(block_size)
    max_past_blocks = int(max_past_blocks)
    if total_tokens <= 0 or block_size <= 0 or max_past_blocks < 0:
        raise ValueError("Window dimensions must be positive and context non-negative.")
    specs: list[tuple[int, int, int, int]] = []
    num_blocks = (total_tokens + block_size - 1) // block_size
    for query_block in range(num_blocks):
        first_block = max(0, query_block - max_past_blocks)
        start = first_block * block_size
        target_start = (query_block - first_block) * block_size
        target_end = min(target_start + block_size, total_tokens - start)
        end = start + target_end
        specs.append((start, end, target_start, target_end))
    return specs


def _window_batch_size(max_window_length: int, max_attention_pairs: int) -> int:
    pairs_per_window = max(1, int(max_window_length) ** 2)
    return max(1, min(32, int(max_attention_pairs) // pairs_per_window))


def encode_with_context_limit(
    model,
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    max_past_blocks: int | None,
    block_size: int,
    max_attention_pairs: int = 8_000_000,
) -> torch.Tensor:
    """Encode one padded video while hard-capping input history per block."""
    if int(features.shape[0]) != 1 or int(valid_mask.shape[0]) != 1:
        raise ValueError("Context-window extraction requires batch_size=1.")
    valid_length = int(valid_mask[0].sum().item())
    if max_past_blocks is None:
        return model.student_enc(
            features.unsqueeze(1),
            valid_mask=valid_mask.unsqueeze(1),
            context_mask=valid_mask.unsqueeze(1),
            N=1,
            L=int(features.shape[1]),
            segment_lengths=None,
        ).float()

    specs = block_window_specs(valid_length, block_size, max_past_blocks)
    max_window_length = min(valid_length, (int(max_past_blocks) + 1) * int(block_size))
    batch_size = _window_batch_size(max_window_length, max_attention_pairs)
    output = None
    for batch_start in range(0, len(specs), batch_size):
        batch_specs = specs[batch_start : batch_start + batch_size]
        padded_length = max(end - start for start, end, _, _ in batch_specs)
        window_features = features.new_zeros(len(batch_specs), padded_length, features.shape[-1])
        window_mask = torch.zeros(
            len(batch_specs), padded_length, dtype=torch.bool, device=features.device
        )
        for index, (start, end, _, _) in enumerate(batch_specs):
            length = end - start
            window_features[index, :length] = features[0, start:end]
            window_mask[index, :length] = True
        encoded = model.student_enc(
            window_features.unsqueeze(1),
            valid_mask=window_mask.unsqueeze(1),
            context_mask=window_mask.unsqueeze(1),
            N=1,
            L=padded_length,
            segment_lengths=None,
            position_offsets=torch.as_tensor(
                [start for start, _, _, _ in batch_specs],
                dtype=torch.long,
                device=features.device,
            ),
        ).float()
        if output is None:
            output = encoded.new_zeros(1, valid_length, encoded.shape[-1])
        for index, (start, _, target_start, target_end) in enumerate(batch_specs):
            query_block = (start + target_start) // block_size
            global_start = query_block * block_size
            global_end = min(global_start + block_size, valid_length)
            output[0, global_start:global_end] = encoded[index, target_start:target_end]
    if output is None:
        raise RuntimeError("No context windows were encoded.")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen full-context surgical-phase linear head while "
            "limiting the P-JEPA student's block-causal lookback."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=["cholec80", "m2cai16"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Frozen P-JEPA checkpoint.")
    parser.add_argument(
        "--linear-checkpoint", required=True, help="Frozen full-context linear head."
    )
    parser.add_argument("--results", required=True, help="Output JSON path.")
    parser.add_argument(
        "--contexts",
        nargs="*",
        default=None,
        help="Past-block budgets, for example: 0 1 2 4 8 16 32 64 full.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-bfloat16", action="store_true")
    return parser.parse_args()


def _load_linear_head(
    checkpoint_path: str,
    *,
    dataset: SurgicalPhaseVideoDataset,
    device: torch.device,
) -> tuple[SurgicalPhaseLinearHead, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("version") != "surgical_phase_linear_probe_v1":
        raise ValueError(f"Unsupported linear checkpoint: {checkpoint_path}")
    if str(checkpoint.get("source_dataset", "")).lower() != dataset.archive.dataset:
        raise ValueError("Linear checkpoint source dataset does not match the evaluation dataset.")
    if checkpoint.get("feature_source") != "student":
        raise ValueError("Context sweep requires a student-feature linear checkpoint.")
    if list(checkpoint.get("phase_class_names", [])) != dataset.archive.phase_class_names:
        raise ValueError("Linear checkpoint phase classes do not match the evaluation archive.")
    input_dim = int(checkpoint["input_dim"])
    head = SurgicalPhaseLinearHead(input_dim, dataset.archive.num_classes).to(device)
    head.load_state_dict(checkpoint["model_state_dict"], strict=True)
    head.eval()
    return head, checkpoint


def _per_phase_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(class_names):
        predicted_class = predictions == class_index
        target_class = targets == class_index
        true_positive = int((predicted_class & target_class).sum().item())
        false_positive = int((predicted_class & ~target_class).sum().item())
        false_negative = int((~predicted_class & target_class).sum().item())
        support = int(target_class.sum().item())
        f1_denominator = 2 * true_positive + false_positive + false_negative
        rows.append(
            {
                "class_index": int(class_index),
                "class_name": str(class_name),
                "support": support,
                "recall": 0.0 if support == 0 else true_positive / support,
                "f1": 0.0 if f1_denominator == 0 else 2.0 * true_positive / f1_denominator,
            }
        )
    return rows


@torch.no_grad()
def evaluate_context(
    model,
    head: SurgicalPhaseLinearHead,
    loader: DataLoader,
    *,
    device: torch.device,
    max_past_blocks: int | None,
    use_bfloat16: bool,
    class_names: list[str],
    block_size: int,
) -> dict[str, Any]:
    model.eval()
    head.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    loss_sum = 0.0
    examples = 0
    sequences = 0
    autocast_enabled = bool(use_bfloat16 and device.type == "cuda")
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True).bool()
        phase = batch["targets"]["phase"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            encoded = encode_with_context_limit(
                model,
                features,
                valid_mask,
                max_past_blocks=max_past_blocks,
                block_size=block_size,
            )
        selected_features = encoded[valid_mask]
        selected_targets = phase[valid_mask]
        logits = head(selected_features)
        loss_sum += float(F.cross_entropy(logits, selected_targets, reduction="sum").item())
        predictions.append(logits.argmax(dim=-1).cpu())
        targets.append(selected_targets.cpu())
        examples += int(selected_targets.numel())
        sequences += int(features.shape[0])
    if not predictions:
        raise RuntimeError("No surgical-phase tokens were evaluated.")
    all_predictions = torch.cat(predictions)
    all_targets = torch.cat(targets)
    per_phase = _per_phase_metrics(all_predictions, all_targets, class_names)
    return {
        "phase_acc": float((all_predictions == all_targets).float().mean().item()),
        "phase_macro_f1": float(sum(row["f1"] for row in per_phase) / len(per_phase)),
        "phase_loss": float(loss_sum / max(1, examples)),
        "tokens": int(examples),
        "sequences": int(sequences),
        "per_phase": per_phase,
    }


def _write_csv(output_path: Path, results: list[dict[str, Any]]) -> None:
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "past_blocks",
                "past_seconds",
                "phase_acc",
                "phase_macro_f1",
                "phase_loss",
                "tokens",
                "sequences",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result["past_blocks"],
                    result["past_seconds"],
                    result["metrics"]["phase_acc"],
                    result["metrics"]["phase_macro_f1"],
                    result["metrics"]["phase_loss"],
                    result["metrics"]["tokens"],
                    result["metrics"]["sequences"],
                ]
            )


def main() -> None:
    args = parse_args()
    config_path = str(Path(args.config).expanduser().resolve())
    config = load_json(config_path)
    runtime_cfg = config.get("runtime", {})
    seed = seed_everything(
        int(runtime_cfg.get("seed", 1538574472)),
        bool(runtime_cfg.get("deterministic_cudnn", True)),
    )
    device = resolve_device(args.device or runtime_cfg.get("device", "auto"))
    checkpoint_path = resolve_path(args.checkpoint)
    linear_checkpoint_path = resolve_path(args.linear_checkpoint)
    paths_cfg = config["paths"]
    if args.dataset == "cholec80":
        archive_path = resolve_path(paths_cfg["cholec80_archive"])
        protocol = "table1"
    else:
        archive_path = resolve_path(paths_cfg["m2cai16_archive"])
        protocol = "official"
    dataset = SurgicalPhaseVideoDataset(
        archive_path,
        dataset=args.dataset,
        protocol=protocol,
        split="test",
    )
    loader_cfg = config.get("loader", {}).get("surgical_phase", {})
    num_workers = int(loader_cfg.get("num_workers", 0))
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": 1,
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_surgical_phase_videos,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = False
    loader = DataLoader(**loader_kwargs)

    pjepa_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    embedded = pjepa_checkpoint.get("config") if isinstance(pjepa_checkpoint, dict) else None
    model_source = embedded or config
    model = build_model(model_source["model"], model_source["ssl"])
    model.load_state_dict(pjepa_checkpoint.get("model_state_dict", pjepa_checkpoint), strict=True)
    if model.student_encoder_attention != "block_causal":
        raise ValueError("Context sweep requires a block-causal P-JEPA checkpoint.")
    model.to(device).eval()
    head, linear_checkpoint = _load_linear_head(
        linear_checkpoint_path,
        dataset=dataset,
        device=device,
    )
    from pjepa.checkpoints import assert_encoder_identity

    assert_encoder_identity(linear_checkpoint, checkpoint_path)

    block_size = int(model.student_encoder_block_size)
    target_fps = float(dataset.archive.target_fps)
    contexts = parse_contexts(args.contexts)
    results: list[dict[str, Any]] = []
    for max_past_blocks in contexts:
        name = context_name(max_past_blocks)
        print(f"[{args.dataset}] evaluating past_blocks={name}", flush=True)
        metrics = evaluate_context(
            model,
            head,
            loader,
            device=device,
            max_past_blocks=max_past_blocks,
            use_bfloat16=bool(args.use_bfloat16),
            class_names=dataset.archive.phase_class_names,
            block_size=block_size,
        )
        result = {
            "past_blocks": name,
            "past_seconds": None
            if max_past_blocks is None
            else float(max_past_blocks * block_size / target_fps),
            "metrics": metrics,
        }
        results.append(result)
        print(
            f"[{args.dataset} past_blocks={name}] "
            f"acc={metrics['phase_acc']:.6f} macro_f1={metrics['phase_macro_f1']:.6f} "
            f"loss={metrics['phase_loss']:.6f}",
            flush=True,
        )

    output_path = Path(resolve_path(args.results)).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "surgical_phase_context_sweep_v1",
        "dataset": args.dataset,
        "protocol": protocol,
        "split": "test",
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "linear_checkpoint": str(Path(linear_checkpoint_path).expanduser().resolve()),
        "config_path": config_path,
        "seed": int(seed),
        "target_fps": target_fps,
        "block_size": block_size,
        "current_block_attention": "bidirectional",
        "linear_head_retrained_per_context": False,
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    _write_csv(output_path, results)
    print(f"Wrote {output_path} and {output_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
