"""Build and validate cached surgical P-JEPA features for linear evaluations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from pjepa.data.surgical_phase_plstitch import SurgicalPhasePLStitchArchive


STUDENT_CACHE_VERSION = "surgical_phase_pjepa_student_v1"


def _file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def validate_student_feature_cache(
    cache_path: str | Path,
    *,
    source_archive_path: str | Path,
    checkpoint_path: str | Path,
) -> None:
    payload = torch.load(
        str(Path(cache_path).expanduser().resolve()),
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    metadata = payload.get("student_feature_cache", {})
    if metadata.get("version") != STUDENT_CACHE_VERSION:
        raise ValueError(f"Student cache {cache_path} has an unsupported or missing cache version.")
    if metadata.get("source_archive") != _file_identity(source_archive_path):
        raise ValueError(
            "Student cache was created from a different PL-Stitch archive. "
            "Delete it or run with --rebuild-cache."
        )
    if metadata.get("checkpoint") != _file_identity(checkpoint_path):
        raise ValueError(
            "Student cache was created from a different P-JEPA checkpoint. "
            "Delete it or run with --rebuild-cache."
        )


@torch.no_grad()
def build_student_feature_cache(
    model,
    *,
    source_archive_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    device: torch.device,
    use_bfloat16: bool = False,
) -> str:
    source_archive_path = Path(source_archive_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    source = SurgicalPhasePLStitchArchive(source_archive_path)

    previous_training = bool(model.training)
    model.eval()
    encoded_videos: list[torch.Tensor] = []
    offsets = [0]
    autocast_enabled = bool(use_bfloat16 and device.type == "cuda")

    for video_index, video_id in enumerate(source.video_ids):
        item = source.fetch_video(video_index)
        features = item["features"].unsqueeze(0).to(device, non_blocking=True)
        valid_mask = torch.ones(1, int(features.shape[1]), dtype=torch.bool, device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            encoded = model.student_enc(
                features.unsqueeze(1),
                valid_mask=valid_mask.unsqueeze(1),
                context_mask=valid_mask.unsqueeze(1),
                N=1,
                L=int(features.shape[1]),
                segment_lengths=None,
            )
        encoded = encoded.squeeze(0).float().cpu().contiguous()
        if int(encoded.shape[0]) != int(features.shape[1]):
            raise ValueError(
                f"P-JEPA changed the token count for {video_id}: "
                f"{features.shape[1]} -> {encoded.shape[0]}."
            )
        encoded_videos.append(encoded)
        offsets.append(offsets[-1] + int(encoded.shape[0]))
        print(
            f"[student cache] {video_index + 1:02d}/{len(source.video_ids):02d} "
            f"{video_id}: {encoded.shape[0]} tokens"
        )

    if previous_training:
        model.train()

    tokens = torch.cat(encoded_videos, dim=0)
    payload = {
        "dataset": source.dataset,
        "video_ids": list(source.video_ids),
        "source_splits": list(source.source_splits),
        "tokens": tokens,
        "offsets": torch.as_tensor(offsets, dtype=torch.long),
        "frame_indices": source.frame_indices.clone(),
        "phase_labels": source.phase_labels.clone(),
        "phase_class_names": list(source.phase_class_names),
        "feature_dim": int(tokens.shape[-1]),
        "target_fps": float(source.target_fps),
        "student_feature_cache": {
            "version": STUDENT_CACHE_VERSION,
            "source_archive": _file_identity(source_archive_path),
            "checkpoint": _file_identity(checkpoint_path),
            "use_bfloat16": bool(use_bfloat16),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)
    validate_student_feature_cache(
        output_path,
        source_archive_path=source_archive_path,
        checkpoint_path=checkpoint_path,
    )
    return str(output_path)
