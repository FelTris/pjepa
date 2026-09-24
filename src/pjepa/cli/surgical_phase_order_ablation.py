from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
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
from pjepa.cli.surgical_phase_context_sweep import _per_phase_metrics
from pjepa.utils.runtime import load_json, resolve_device, resolve_path, seed_everything


CONDITIONS = (
    "original",
    "within_segment_reverse",
    "segment_order_reverse",
    "full_reverse",
    "random_segment_order",
)


def contiguous_segments(labels: torch.Tensor) -> list[tuple[int, int]]:
    labels = labels.reshape(-1)
    if not labels.numel():
        raise ValueError("Cannot segment an empty label sequence.")
    boundaries = (labels[1:] != labels[:-1]).nonzero(as_tuple=False).flatten() + 1
    starts = [0, *[int(value) for value in boundaries.tolist()]]
    ends = [*[int(value) for value in boundaries.tolist()], int(labels.numel())]
    return list(zip(starts, ends, strict=True))


def _stable_random_order(num_segments: int, *, seed: int, video_id: str, repeat: int) -> list[int]:
    if num_segments <= 1:
        return list(range(num_segments))
    digest = hashlib.sha256(f"{seed}:{video_id}:{repeat}".encode("utf-8")).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
    order = torch.randperm(num_segments, generator=generator).tolist()
    if order == list(range(num_segments)):
        order = order[1:] + order[:1]
    return [int(value) for value in order]


def transformation_indices(
    labels: torch.Tensor,
    condition: str,
    *,
    seed: int = 0,
    video_id: str = "",
    repeat: int = 0,
) -> tuple[torch.Tensor, list[int]]:
    condition = str(condition)
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition={condition!r}.")
    labels = labels.reshape(-1)
    segments = contiguous_segments(labels)
    num_segments = len(segments)
    if condition in {"original", "within_segment_reverse"}:
        segment_order = list(range(num_segments))
    elif condition in {"segment_order_reverse", "full_reverse"}:
        segment_order = list(reversed(range(num_segments)))
    else:
        segment_order = _stable_random_order(
            num_segments, seed=seed, video_id=video_id, repeat=repeat
        )

    reverse_within = condition in {"within_segment_reverse", "full_reverse"}
    pieces: list[torch.Tensor] = []
    for segment_index in segment_order:
        start, end = segments[segment_index]
        if reverse_within:
            pieces.append(torch.arange(end - 1, start - 1, -1, dtype=torch.long))
        else:
            pieces.append(torch.arange(start, end, dtype=torch.long))
    indices = torch.cat(pieces)
    if int(indices.numel()) != int(labels.numel()) or int(indices.unique().numel()) != int(
        labels.numel()
    ):
        raise RuntimeError("Order transformation did not produce a token permutation.")
    return indices, segment_order


def retained_adjacency_fraction(segment_order: list[int]) -> float:
    if len(segment_order) <= 1:
        return 1.0
    retained = sum(
        int(right == left + 1)
        for left, right in zip(segment_order[:-1], segment_order[1:], strict=True)
    )
    return retained / (len(segment_order) - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen surgical-phase heads after controlled phase-segment "
            "order transformations."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=["cholec80", "m2cai16"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Frozen P-JEPA checkpoint.")
    parser.add_argument("--student-linear-checkpoint", required=True)
    parser.add_argument("--raw-linear-checkpoint", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-bfloat16", action="store_true")
    return parser.parse_args()


def _load_head(
    checkpoint_path: str,
    *,
    dataset: SurgicalPhaseVideoDataset,
    feature_source: str,
    device: torch.device,
) -> tuple[SurgicalPhaseLinearHead, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("version") != "surgical_phase_linear_probe_v1":
        raise ValueError(f"Unsupported linear checkpoint: {checkpoint_path}")
    if str(checkpoint.get("source_dataset", "")).lower() != dataset.archive.dataset:
        raise ValueError("Linear checkpoint source dataset does not match evaluation dataset.")
    if checkpoint.get("feature_source") != feature_source:
        raise ValueError("Linear checkpoint feature source does not match requested source.")
    if list(checkpoint.get("phase_class_names", [])) != dataset.archive.phase_class_names:
        raise ValueError("Linear checkpoint phase classes do not match evaluation archive.")
    head = SurgicalPhaseLinearHead(int(checkpoint["input_dim"]), dataset.archive.num_classes).to(
        device
    )
    head.load_state_dict(checkpoint["model_state_dict"], strict=True)
    head.eval()
    return head, checkpoint


def _metrics(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    *,
    loss_sum: float,
    examples: int,
    sequences: int,
    class_names: list[str],
) -> dict[str, Any]:
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


@torch.no_grad()
def evaluate_condition(
    model,
    student_head: SurgicalPhaseLinearHead,
    raw_head: SurgicalPhaseLinearHead,
    loader: DataLoader,
    *,
    device: torch.device,
    condition: str,
    repeat: int,
    seed: int,
    use_bfloat16: bool,
    class_names: list[str],
) -> dict[str, Any]:
    model.eval()
    student_head.eval()
    raw_head.eval()
    student_predictions: list[torch.Tensor] = []
    raw_predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    student_loss_sum = 0.0
    raw_loss_sum = 0.0
    examples = 0
    sequences = 0
    adjacency_weighted_sum = 0.0
    adjacency_denominator = 0
    segment_count = 0
    autocast_enabled = bool(use_bfloat16 and device.type == "cuda")
    for batch in loader:
        if int(batch["features"].shape[0]) != 1:
            raise ValueError("Order ablation requires batch_size=1.")
        valid_length = int(batch["valid_mask"][0].sum().item())
        raw_features = batch["features"][:, :valid_length].to(device, non_blocking=True)
        phase = batch["targets"]["phase"][0, :valid_length]
        video_id = str(batch["video_id"][0])
        indices, segment_order = transformation_indices(
            phase,
            condition,
            seed=seed,
            video_id=video_id,
            repeat=repeat,
        )
        indices = indices.to(device)
        transformed_features = raw_features[:, indices]
        transformed_targets = phase[indices.cpu()].to(device, non_blocking=True)
        valid_mask = torch.ones(1, valid_length, dtype=torch.bool, device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            student_features = model.student_enc(
                transformed_features.unsqueeze(1),
                valid_mask=valid_mask.unsqueeze(1),
                context_mask=valid_mask.unsqueeze(1),
                N=1,
                L=valid_length,
                segment_lengths=None,
            ).float()
        student_logits = student_head(student_features[0])
        raw_logits = raw_head(transformed_features[0])
        student_loss_sum += float(
            F.cross_entropy(student_logits, transformed_targets, reduction="sum").item()
        )
        raw_loss_sum += float(
            F.cross_entropy(raw_logits, transformed_targets, reduction="sum").item()
        )
        student_predictions.append(student_logits.argmax(dim=-1).cpu())
        raw_predictions.append(raw_logits.argmax(dim=-1).cpu())
        targets.append(transformed_targets.cpu())
        examples += valid_length
        sequences += 1
        segment_count += len(segment_order)
        adjacency_pairs = max(0, len(segment_order) - 1)
        adjacency_weighted_sum += retained_adjacency_fraction(segment_order) * adjacency_pairs
        adjacency_denominator += adjacency_pairs

    common = {
        "examples": examples,
        "sequences": sequences,
        "class_names": class_names,
    }
    return {
        "condition": condition,
        "repeat": int(repeat),
        "segments": int(segment_count),
        "retained_directed_segment_adjacency_fraction": float(
            adjacency_weighted_sum / max(1, adjacency_denominator)
        ),
        "student": _metrics(
            student_predictions,
            targets,
            loss_sum=student_loss_sum,
            class_names=common["class_names"],
            examples=common["examples"],
            sequences=common["sequences"],
        ),
        "raw": _metrics(
            raw_predictions,
            targets,
            loss_sum=raw_loss_sum,
            class_names=common["class_names"],
            examples=common["examples"],
            sequences=common["sequences"],
        ),
    }


def aggregate_repeats(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    if not repeats:
        raise ValueError("Cannot aggregate an empty result list.")
    result: dict[str, Any] = {
        "num_repeats": len(repeats),
        "retained_directed_segment_adjacency_fraction_mean": statistics.fmean(
            row["retained_directed_segment_adjacency_fraction"] for row in repeats
        ),
    }
    for source in ("student", "raw"):
        aggregate: dict[str, Any] = {}
        for metric in ("phase_acc", "phase_macro_f1", "phase_loss"):
            values = [float(row[source][metric]) for row in repeats]
            aggregate[f"{metric}_mean"] = statistics.fmean(values)
            aggregate[f"{metric}_std"] = statistics.pstdev(values)
        aggregate["per_phase"] = []
        for class_index, phase in enumerate(repeats[0][source]["per_phase"]):
            phase_row = {
                "class_index": int(class_index),
                "class_name": phase["class_name"],
                "support": int(phase["support"]),
            }
            for metric in ("recall", "f1"):
                values = [float(row[source]["per_phase"][class_index][metric]) for row in repeats]
                phase_row[f"{metric}_mean"] = statistics.fmean(values)
                phase_row[f"{metric}_std"] = statistics.pstdev(values)
            aggregate["per_phase"].append(phase_row)
        result[source] = aggregate
    return result


def _write_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    with output_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "repeat",
                "source",
                "retained_segment_adjacency_fraction",
                "phase_acc",
                "phase_macro_f1",
                "phase_loss",
            ]
        )
        for row in rows:
            for source in ("student", "raw"):
                writer.writerow(
                    [
                        row["condition"],
                        row["repeat"],
                        source,
                        row["retained_directed_segment_adjacency_fraction"],
                        row[source]["phase_acc"],
                        row[source]["phase_macro_f1"],
                        row[source]["phase_loss"],
                    ]
                )


def main() -> None:
    args = parse_args()
    if args.random_repeats <= 0:
        raise ValueError("--random-repeats must be positive.")
    config_path = str(Path(args.config).expanduser().resolve())
    config = load_json(config_path)
    runtime_cfg = config.get("runtime", {})
    seed = seed_everything(
        int(runtime_cfg.get("seed", 1538574472)),
        bool(runtime_cfg.get("deterministic_cudnn", True)),
    )
    device = resolve_device(args.device or runtime_cfg.get("device", "auto"))
    checkpoint_path = resolve_path(args.checkpoint)
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
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_surgical_phase_videos,
    )

    pjepa_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    embedded = pjepa_checkpoint.get("config") if isinstance(pjepa_checkpoint, dict) else None
    model_source = embedded or config
    model = build_model(model_source["model"], model_source["ssl"])
    model.load_state_dict(pjepa_checkpoint.get("model_state_dict", pjepa_checkpoint), strict=True)
    model.to(device).eval()
    student_checkpoint_path = resolve_path(args.student_linear_checkpoint)
    raw_checkpoint_path = resolve_path(args.raw_linear_checkpoint)
    student_head, _ = _load_head(
        student_checkpoint_path,
        dataset=dataset,
        feature_source="student",
        device=device,
    )
    raw_head, _ = _load_head(
        raw_checkpoint_path,
        dataset=dataset,
        feature_source="raw",
        device=device,
    )

    all_rows: list[dict[str, Any]] = []
    condition_results: dict[str, Any] = {}
    for condition in CONDITIONS:
        num_repeats = args.random_repeats if condition == "random_segment_order" else 1
        repeats: list[dict[str, Any]] = []
        for repeat in range(num_repeats):
            print(f"[{args.dataset}] condition={condition} repeat={repeat}", flush=True)
            row = evaluate_condition(
                model,
                student_head,
                raw_head,
                loader,
                device=device,
                condition=condition,
                repeat=repeat,
                seed=seed,
                use_bfloat16=bool(args.use_bfloat16),
                class_names=dataset.archive.phase_class_names,
            )
            repeats.append(row)
            all_rows.append(row)
            print(
                f"[{condition} {repeat}] student acc={row['student']['phase_acc']:.6f} "
                f"macro_f1={row['student']['phase_macro_f1']:.6f} "
                f"raw acc={row['raw']['phase_acc']:.6f}",
                flush=True,
            )
        condition_results[condition] = {
            "repeats": repeats,
            "aggregate": aggregate_repeats(repeats),
        }

    original_raw = condition_results["original"]["repeats"][0]["raw"]
    for row in all_rows:
        if row["raw"]["phase_acc"] != original_raw["phase_acc"]:
            raise RuntimeError("Raw-head accuracy changed after a token permutation.")
        if row["raw"]["phase_macro_f1"] != original_raw["phase_macro_f1"]:
            raise RuntimeError("Raw-head macro-F1 changed after a token permutation.")

    output_path = Path(resolve_path(args.results)).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "surgical_phase_order_ablation_v1",
        "dataset": args.dataset,
        "protocol": protocol,
        "split": "test",
        "segment_definition": "maximal contiguous ground-truth phase run",
        "labels_supplied_to_model": False,
        "random_repeats": int(args.random_repeats),
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "student_linear_checkpoint": str(Path(student_checkpoint_path).expanduser().resolve()),
        "raw_linear_checkpoint": str(Path(raw_checkpoint_path).expanduser().resolve()),
        "config_path": config_path,
        "seed": int(seed),
        "conditions": condition_results,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    _write_csv(output_path, all_rows)
    print(f"Wrote {output_path} and {output_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
