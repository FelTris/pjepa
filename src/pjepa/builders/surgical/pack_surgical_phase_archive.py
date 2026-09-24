#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


from pjepa.builders.surgical.common import (
    atomic_json_dump,
    atomic_torch_save,
    per_video_feature_path,
    read_manifest,
    torch_load,
    validate_feature_payload,
)

FEATURE_DIM = 768


ARCHIVE_VERSION = "surgical_phase_plstitch_archive_v1"
HECVL_ARCHIVE_VERSION = "surgical_phase_hecvl_archive_v1"
SUPPORTED_ARCHIVE_VERSIONS = {ARCHIVE_VERSION, HECVL_ARCHIVE_VERSION}

PHASE_CLASS_NAMES = {
    "cholec80": [
        "Preparation",
        "CalotTriangleDissection",
        "ClippingCutting",
        "GallbladderDissection",
        "GallbladderPackaging",
        "CleaningCoagulation",
        "GallbladderRetraction",
    ],
    "m2cai16": [
        "TrocarPlacement",
        "Preparation",
        "CalotTriangleDissection",
        "ClippingCutting",
        "GallbladderDissection",
        "GallbladderPackaging",
        "CleaningCoagulation",
        "GallbladderRetraction",
    ],
}


@dataclass(frozen=True)
class VideoSpec:
    row: dict[str, str]
    feature_path: Path
    kept_positions: torch.Tensor
    frame_indices: torch.Tensor
    phase_labels: torch.Tensor
    source_tokens: int
    discarded_tokens: int

    @property
    def num_tokens(self) -> int:
        return int(self.kept_positions.numel())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pack sampled PL-Stitch features and aligned phase labels into one "
            "memory-mappable surgical phase archive."
        )
    )
    parser.add_argument("--dataset", choices=sorted(PHASE_CLASS_NAMES), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per-video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=1.0)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--video-id-substr", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-source-verification", action="store_true")
    return parser.parse_args()


def read_phase_annotations(path: str | Path, *, class_to_index: dict[str, int]) -> torch.Tensor:
    annotation_path = Path(path)
    labels_by_frame: dict[int, int] = {}
    with annotation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty phase annotation: {annotation_path}") from exc
        for line_number, row in enumerate(reader, start=2):
            if not row or not any(value.strip() for value in row):
                continue
            if len(row) < 2:
                raise ValueError(f"Malformed phase row at {annotation_path}:{line_number}: {row}")
            frame_index = int(row[0].strip())
            phase_name = row[1].strip()
            if phase_name not in class_to_index:
                raise ValueError(f"Unknown phase {phase_name!r} at {annotation_path}:{line_number}")
            if frame_index in labels_by_frame:
                raise ValueError(
                    f"Duplicate frame {frame_index} at {annotation_path}:{line_number}"
                )
            labels_by_frame[frame_index] = class_to_index[phase_name]
    if not labels_by_frame:
        raise ValueError(f"No phase labels found in {annotation_path}")
    if min(labels_by_frame) < 0:
        raise ValueError(f"Negative frame index in {annotation_path}")
    labels = torch.full((max(labels_by_frame) + 1,), -1, dtype=torch.int64)
    for frame_index, label in labels_by_frame.items():
        labels[frame_index] = label
    return labels


def common_feature_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    feature_type = str(payload.get("feature_type", "final_cls"))
    backbone = str(payload.get("backbone", "")).strip().lower()
    if not backbone:
        backbone = "hecvl" if feature_type == "hecvl_img_emb" else "pl_stitch"
    return {
        "backbone": backbone,
        "source_feature_version": str(payload.get("version", "")),
        "checkpoint_sha256": str(payload.get("checkpoint_sha256", "")),
        "checkpoint_path": str(payload.get("checkpoint_path", "")),
        "pl_stitch_git_commit": str(payload.get("pl_stitch_git_commit", "")),
        "surgvlp_git_commit": str(payload.get("surgvlp_git_commit", "")),
        "target_fps": float(payload["target_fps"]),
        "feature_dim": int(payload["features"].shape[1]),
        "feature_type": feature_type,
        "storage_dtype": str(payload.get("storage_dtype", "float32")),
        "compute_dtype": str(payload.get("compute_dtype", "float32")),
        "preprocessing": payload.get("preprocessing", {}),
    }


def scan_specs(
    rows: list[dict[str, str]],
    *,
    dataset: str,
    per_video_root: Path,
    target_fps: float,
) -> tuple[list[VideoSpec], dict[str, Any], list[dict[str, Any]]]:
    class_names = PHASE_CLASS_NAMES[dataset]
    class_to_index = {name: index for index, name in enumerate(class_names)}
    specs: list[VideoSpec] = []
    common_metadata: dict[str, Any] | None = None
    discarded: list[dict[str, Any]] = []

    for row in tqdm(rows, desc=f"Scanning {dataset} features", unit="video"):
        if row.get("dataset", "").strip().lower() != dataset:
            raise ValueError(
                f"Manifest row {row.get('video_id')} belongs to "
                f"dataset={row.get('dataset')!r}, expected {dataset!r}."
            )
        video_id = row["video_id"]
        feature_path = per_video_feature_path(per_video_root, video_id)
        if not feature_path.is_file():
            raise FileNotFoundError(f"Missing per-video feature file: {feature_path}")
        payload = torch_load(feature_path, mmap=True)
        features, _, _ = validate_feature_payload(
            payload,
            expected_fps=target_fps,
            expected_dim=FEATURE_DIM,
            require_finite=True,
        )
        if str(payload["video_id"]) != video_id:
            raise ValueError(f"Feature video ID does not match manifest: {feature_path}")
        frame_indices = payload.get("feature_frame_indices")
        if not torch.is_tensor(frame_indices):
            raise ValueError(f"Missing feature_frame_indices: {feature_path}")
        frame_indices = frame_indices.to(dtype=torch.int64)
        if frame_indices.numel() > 1 and bool(
            (frame_indices[1:] <= frame_indices[:-1]).any().item()
        ):
            raise ValueError(
                f"Sampled source-frame indices must be strictly increasing: {feature_path}"
            )

        labels = read_phase_annotations(row["phase_annotation_path"], class_to_index=class_to_index)
        valid = frame_indices < int(labels.numel())
        safe_indices = frame_indices.clamp(max=max(0, int(labels.numel()) - 1))
        valid &= labels[safe_indices] >= 0
        kept_positions = torch.nonzero(valid, as_tuple=False).flatten()
        kept_frame_indices = frame_indices[kept_positions].contiguous()
        kept_phase_labels = labels[kept_frame_indices].contiguous()
        discarded_count = int(features.shape[0]) - int(kept_positions.numel())
        if discarded_count:
            invalid_positions = torch.nonzero(~valid, as_tuple=False).flatten()
            discarded.append(
                {
                    "video_id": video_id,
                    "count": discarded_count,
                    "source_frame_indices": frame_indices[invalid_positions].tolist(),
                }
            )
        if kept_positions.numel() == 0:
            raise ValueError(f"No sampled features have valid phase labels: {video_id}")

        metadata = common_feature_metadata(payload)
        if common_metadata is None:
            common_metadata = metadata
        elif metadata != common_metadata:
            raise ValueError(
                f"Feature metadata mismatch in {feature_path}: {metadata} != {common_metadata}"
            )
        specs.append(
            VideoSpec(
                row=row,
                feature_path=feature_path,
                kept_positions=kept_positions,
                frame_indices=kept_frame_indices,
                phase_labels=kept_phase_labels,
                source_tokens=int(features.shape[0]),
                discarded_tokens=discarded_count,
            )
        )
    if not specs or common_metadata is None:
        raise RuntimeError("No per-video features were available to pack.")
    return specs, common_metadata, discarded


def validate_archive(payload: Any, *, expected_dataset: str | None = None) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Archive must be a dictionary.")
    required = {
        "version",
        "dataset",
        "video_ids",
        "paths",
        "source_splits",
        "tokens",
        "times",
        "nominal_times",
        "frame_indices",
        "phase_labels",
        "offsets",
        "phase_class_names",
        "feature_dim",
        "target_fps",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Archive is missing keys: {missing}")
    if payload["version"] not in SUPPORTED_ARCHIVE_VERSIONS:
        raise ValueError(f"Unsupported archive version: {payload['version']!r}")
    dataset = str(payload["dataset"])
    if expected_dataset is not None and dataset != expected_dataset:
        raise ValueError(f"Expected dataset={expected_dataset}, got {dataset}")
    if dataset not in PHASE_CLASS_NAMES:
        raise ValueError(f"Unsupported archive dataset: {dataset}")
    if list(payload["phase_class_names"]) != PHASE_CLASS_NAMES[dataset]:
        raise ValueError("Archive phase-class mapping is not canonical.")

    video_ids = payload["video_ids"]
    paths = payload["paths"]
    source_splits = payload["source_splits"]
    if not all(isinstance(value, list) for value in (video_ids, paths, source_splits)):
        raise ValueError("Archive video IDs, paths and splits must be lists.")
    if not (len(video_ids) == len(paths) == len(source_splits)):
        raise ValueError("Archive video ID/path/split counts differ.")
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("Archive contains duplicate video IDs.")

    tokens = payload["tokens"]
    if (
        not torch.is_tensor(tokens)
        or tokens.ndim != 2
        or tokens.dtype != torch.float32
        or int(tokens.shape[1]) != int(payload["feature_dim"])
    ):
        raise ValueError("Archive tokens must be FP32 [T, feature_dim].")
    total_tokens = int(tokens.shape[0])
    for name, dtype in (
        ("times", torch.float32),
        ("nominal_times", torch.float32),
        ("frame_indices", torch.int64),
        ("phase_labels", torch.int64),
    ):
        tensor = payload[name]
        if (
            not torch.is_tensor(tensor)
            or tensor.ndim != 1
            or tensor.dtype != dtype
            or int(tensor.numel()) != total_tokens
        ):
            raise ValueError(f"Archive {name} has an invalid shape or dtype.")
    offsets = payload["offsets"]
    if (
        not torch.is_tensor(offsets)
        or offsets.ndim != 1
        or offsets.dtype != torch.int64
        or int(offsets.numel()) != len(video_ids) + 1
        or int(offsets[0].item()) != 0
        or int(offsets[-1].item()) != total_tokens
    ):
        raise ValueError("Archive offsets do not span all videos and tokens.")
    if bool((offsets[1:] <= offsets[:-1]).any().item()):
        raise ValueError("Every archived video must contain at least one token.")
    for video_index in range(len(video_ids)):
        start = int(offsets[video_index].item())
        end = int(offsets[video_index + 1].item())
        indices = payload["frame_indices"][start:end]
        if indices.numel() > 1 and bool((indices[1:] <= indices[:-1]).any().item()):
            raise ValueError(f"Frame indices are not increasing for {video_ids[video_index]}")
    num_classes = len(payload["phase_class_names"])
    phase_labels = payload["phase_labels"]
    if bool(((phase_labels < 0) | (phase_labels >= num_classes)).any().item()):
        raise ValueError("Archive contains an out-of-range phase label.")
    if not bool(torch.isfinite(tokens).all().item()):
        raise ValueError("Archive tokens contain NaN or Inf values.")


def verify_source_features(archive: dict[str, Any], specs: list[VideoSpec]) -> None:
    offsets = archive["offsets"]
    for video_index, spec in enumerate(tqdm(specs, desc="Verifying packed sources", unit="video")):
        payload = torch_load(spec.feature_path, mmap=True)
        start = int(offsets[video_index].item())
        end = int(offsets[video_index + 1].item())
        positions = spec.kept_positions
        comparisons = {
            "features": torch.equal(archive["tokens"][start:end], payload["features"][positions]),
            "times": torch.equal(archive["times"][start:end], payload["times"][positions]),
            "nominal_times": torch.equal(
                archive["nominal_times"][start:end],
                payload["nominal_times"][positions],
            ),
            "frame_indices": torch.equal(archive["frame_indices"][start:end], spec.frame_indices),
            "phase_labels": torch.equal(archive["phase_labels"][start:end], spec.phase_labels),
        }
        failed = [name for name, matches in comparisons.items() if not matches]
        if failed:
            raise ValueError(
                f"Packed tensors differ from sources for {spec.row['video_id']}: {failed}"
            )


def main() -> int:
    args = parse_args()
    if args.target_fps <= 0:
        raise ValueError("--target-fps must be positive.")
    manifest = args.manifest.expanduser().resolve()
    per_video_root = args.per_video_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
    rows = read_manifest(manifest)
    if args.video_id_substr:
        rows = [row for row in rows if args.video_id_substr in row["video_id"]]
    if args.limit_videos > 0:
        rows = rows[: args.limit_videos]
    if not rows:
        raise RuntimeError("No manifest videos selected.")
    video_ids = [row["video_id"] for row in rows]
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("Manifest contains duplicate video IDs.")

    specs, metadata, discarded = scan_specs(
        rows,
        dataset=args.dataset,
        per_video_root=per_video_root,
        target_fps=args.target_fps,
    )
    lengths = [spec.num_tokens for spec in specs]
    offsets = torch.zeros(len(specs) + 1, dtype=torch.int64)
    offsets[1:] = torch.as_tensor(lengths, dtype=torch.int64).cumsum(dim=0)
    total_tokens = int(offsets[-1].item())
    tokens = torch.empty((total_tokens, FEATURE_DIM), dtype=torch.float32)
    times = torch.empty(total_tokens, dtype=torch.float32)
    nominal_times = torch.empty(total_tokens, dtype=torch.float32)
    frame_indices = torch.empty(total_tokens, dtype=torch.int64)
    phase_labels = torch.empty(total_tokens, dtype=torch.int64)

    for video_index, spec in enumerate(tqdm(specs, desc=f"Packing {args.dataset}", unit="video")):
        payload = torch_load(spec.feature_path, mmap=True)
        start = int(offsets[video_index].item())
        end = int(offsets[video_index + 1].item())
        positions = spec.kept_positions
        tokens[start:end].copy_(payload["features"][positions])
        times[start:end].copy_(payload["times"][positions])
        nominal_times[start:end].copy_(payload["nominal_times"][positions])
        frame_indices[start:end].copy_(spec.frame_indices)
        phase_labels[start:end].copy_(spec.phase_labels)

    archive_version = HECVL_ARCHIVE_VERSION if metadata["backbone"] == "hecvl" else ARCHIVE_VERSION
    archive: dict[str, Any] = {
        "version": archive_version,
        "dataset": args.dataset,
        "source_manifest": str(manifest),
        "features_root": str(per_video_root),
        "video_ids": [spec.row["video_id"] for spec in specs],
        "paths": [spec.row["relative_path"] for spec in specs],
        "phase_annotation_paths": [spec.row["phase_annotation_path"] for spec in specs],
        "timestamp_paths": [spec.row["timestamp_path"] for spec in specs],
        "source_splits": [spec.row["split"].strip().lower() for spec in specs],
        "source_fps": torch.as_tensor(
            [float(spec.row["source_fps"]) for spec in specs], dtype=torch.float64
        ),
        "source_num_frames": torch.as_tensor(
            [int(spec.row["num_frames"]) for spec in specs], dtype=torch.int64
        ),
        "source_num_annotation_frames": torch.as_tensor(
            [int(spec.row["num_annotation_frames"]) for spec in specs],
            dtype=torch.int64,
        ),
        "source_num_tokens": torch.as_tensor(
            [spec.source_tokens for spec in specs], dtype=torch.int64
        ),
        "num_tokens": torch.as_tensor(lengths, dtype=torch.int64),
        "discarded_unlabelled_tokens": torch.as_tensor(
            [spec.discarded_tokens for spec in specs], dtype=torch.int64
        ),
        "tokens": tokens,
        "times": times,
        "nominal_times": nominal_times,
        "frame_indices": frame_indices,
        "phase_labels": phase_labels,
        "offsets": offsets,
        "phase_class_names": PHASE_CLASS_NAMES[args.dataset],
        "phase_class_to_index": {
            name: index for index, name in enumerate(PHASE_CLASS_NAMES[args.dataset])
        },
        **metadata,
    }
    validate_archive(archive, expected_dataset=args.dataset)
    atomic_torch_save(archive, output)

    reloaded = torch_load(output, mmap=True)
    validate_archive(reloaded, expected_dataset=args.dataset)
    if not args.skip_source_verification:
        verify_source_features(reloaded, specs)
    summary = {
        "archive": str(output),
        "dataset": args.dataset,
        "videos": len(specs),
        "tokens": total_tokens,
        "target_fps": args.target_fps,
        "feature_dim": FEATURE_DIM,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "discarded_unlabelled_tokens": discarded,
        "bytes": output.stat().st_size,
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    atomic_json_dump(summary, summary_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
