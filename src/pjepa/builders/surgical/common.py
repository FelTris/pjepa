from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch


MANIFEST_FIELDS = [
    "dataset",
    "video_id",
    "relative_path",
    "absolute_path",
    "split",
    "phase_annotation_path",
    "timestamp_path",
    "num_annotation_frames",
    "num_timestamp_rows",
    "duration_sec",
    "source_fps",
    "width",
    "height",
    "num_frames",
    "file_size_bytes",
    "status",
    "error",
]


def torch_load(path: str | Path, *, mmap: bool = False) -> Any:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: Any, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_dump(payload: Any, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def read_manifest(path: str | Path, *, valid_only: bool = True) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if valid_only:
        rows = [row for row in rows if row.get("status", "ok").strip().lower() == "ok"]
    return rows


def write_manifest(rows: Iterable[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def per_video_feature_path(root: str | Path, video_id: str) -> Path:
    return Path(root) / f"{video_id}.pt"


def parse_split_list(value: str) -> set[str]:
    values = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not values:
        raise ValueError("Expected at least one split name.")
    return values


def validate_feature_payload(
    payload: Any,
    *,
    expected_fps: float | None = None,
    expected_dim: int = 768,
    require_finite: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dict payload, got {type(payload)!r}.")
    required = {"video_id", "features", "times", "nominal_times", "target_fps"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Missing feature keys: {missing}")

    features = payload["features"]
    times = payload["times"]
    nominal_times = payload["nominal_times"]
    if not torch.is_tensor(features) or features.ndim != 2:
        raise ValueError("features must be a [T, D] tensor.")
    if features.dtype != torch.float32:
        raise ValueError(f"features must be float32, got {features.dtype}.")
    if int(features.shape[1]) != int(expected_dim):
        raise ValueError(f"Expected feature_dim={expected_dim}, got {features.shape[1]}.")
    if not torch.is_tensor(times) or times.ndim != 1 or times.dtype != torch.float32:
        raise ValueError("times must be a float32 [T] tensor.")
    if (
        not torch.is_tensor(nominal_times)
        or nominal_times.ndim != 1
        or nominal_times.dtype != torch.float32
    ):
        raise ValueError("nominal_times must be a float32 [T] tensor.")
    if not (int(features.shape[0]) == int(times.numel()) == int(nominal_times.numel())):
        raise ValueError("Feature and timestamp lengths differ.")
    if features.shape[0] == 0:
        raise ValueError("Feature sequence is empty.")
    if (
        expected_fps is not None
        and abs(float(payload["target_fps"]) - float(expected_fps)) > 1.0e-9
    ):
        raise ValueError(f"Expected target_fps={expected_fps}, got {payload['target_fps']}.")
    if times.numel() > 1 and bool((times[1:] < times[:-1]).any().item()):
        raise ValueError("Decoded timestamps are not monotonic.")
    if nominal_times.numel() > 1 and bool((nominal_times[1:] <= nominal_times[:-1]).any().item()):
        raise ValueError("Nominal timestamps are not strictly increasing.")
    if require_finite:
        if not bool(torch.isfinite(features).all().item()):
            raise ValueError("Features contain NaN or Inf values.")
        if not bool(torch.isfinite(times).all().item()):
            raise ValueError("Decoded timestamps contain NaN or Inf values.")

    frame_indices = payload.get("feature_frame_indices", payload.get("frame_indices"))
    if frame_indices is not None:
        if (
            not torch.is_tensor(frame_indices)
            or frame_indices.ndim != 1
            or frame_indices.dtype != torch.int64
        ):
            raise ValueError("feature_frame_indices must be an int64 [T] tensor.")
        if int(frame_indices.numel()) != int(features.shape[0]):
            raise ValueError("Feature and source-frame-index lengths differ.")
        if bool((frame_indices < 0).any().item()):
            raise ValueError("Source-frame indices cannot be negative.")
        if frame_indices.numel() > 1 and bool(
            (frame_indices[1:] < frame_indices[:-1]).any().item()
        ):
            raise ValueError("Source-frame indices are not monotonic.")
    return features, times, nominal_times
