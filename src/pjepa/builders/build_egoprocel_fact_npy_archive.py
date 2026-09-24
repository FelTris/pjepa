#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np
import torch


from pjepa.data.common import resolve_feature_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Pack FACT EgoProceL per-video .npy features into the frame archive "
            "format used by the existing P-JEPA EgoProceL loader."
        )
    )
    parser.add_argument("--manifest", required=True, help="EgoProceL manifest CSV.")
    parser.add_argument(
        "--fact-root",
        default="FACT_actseg/data/egoprocel",
        help="FACT EgoProceL root containing features/ and groundTruth/.",
    )
    parser.add_argument("--output", required=True, help="Output .pt archive path.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite an existing output archive."
    )
    parser.add_argument(
        "--splits", nargs="*", default=None, help="Optional official splits to include."
    )
    parser.add_argument(
        "--limit-videos", type=int, default=None, help="Optional limit for smoke tests."
    )
    parser.add_argument(
        "--feature-fps",
        type=float,
        default=10.0,
        help="FPS of FACT dense .npy features and groundTruth files.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=4.0,
        help="FPS to pack into the archive. Use <=0 to keep the native feature FPS.",
    )
    parser.add_argument(
        "--no-trim-to-ground-truth",
        action="store_true",
        help="Do not trim feature arrays to the matching FACT groundTruth length.",
    )
    return parser.parse_args()


def archive_rel_path(video_path: str) -> str:
    path = Path(str(video_path).strip())
    if path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}:
        path = path.with_suffix(".pt")
    return str(path)


def iter_unique_videos(manifest_path: str, splits=None):
    splits = {split.lower() for split in splits} if splits else None
    seen = set()
    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"video_path", "video_uid"}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Manifest {manifest_path} is missing required columns: {missing}")
        for row in reader:
            if splits is not None:
                split = str(row.get("official_split", "")).strip().lower()
                if split not in splits:
                    continue
            rel_path = archive_rel_path(row["video_path"])
            if rel_path in seen:
                continue
            seen.add(rel_path)
            yield {
                "archive_rel_path": rel_path,
                "video_uid": str(row["video_uid"]).strip(),
                "video_path": str(row["video_path"]).strip(),
            }


def count_ground_truth_lines(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_feature_array(path: Path):
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D FACT feature array at {path}, got shape {arr.shape}")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def resample_indices(num_frames: int, source_fps: float, target_fps: float):
    if num_frames <= 0:
        return np.zeros((0,), dtype=np.int64)
    if target_fps <= 0 or np.isclose(source_fps, target_fps):
        return np.arange(num_frames, dtype=np.int64)
    if source_fps <= 0:
        raise ValueError(f"source_fps must be positive, got {source_fps}")

    duration_sec = num_frames / float(source_fps)
    sample_times = np.arange(0.0, duration_sec, 1.0 / float(target_fps), dtype=np.float64)
    indices = np.floor(sample_times * float(source_fps) + 1e-6).astype(np.int64)
    indices = indices[indices < num_frames]
    if indices.size == 0:
        indices = np.asarray([0], dtype=np.int64)
    return indices


def load_resampled_feature(
    feature_path: Path,
    ground_truth_path: Path,
    *,
    feature_fps: float,
    target_fps: float,
    trim_to_ground_truth: bool,
):
    features = load_feature_array(feature_path)
    original_len = int(features.shape[0])

    gt_len = count_ground_truth_lines(ground_truth_path)
    if trim_to_ground_truth and gt_len is not None:
        keep_len = min(int(features.shape[0]), int(gt_len))
        features = features[:keep_len]

    indices = resample_indices(int(features.shape[0]), feature_fps, target_fps)
    features = np.ascontiguousarray(features[indices])
    times = (indices.astype(np.float32) / float(feature_fps)).astype(np.float32)
    return (
        torch.from_numpy(features).to(dtype=torch.float32),
        torch.from_numpy(times).to(dtype=torch.float32),
        original_len,
        None if gt_len is None else int(gt_len),
    )


def build_archive(
    manifest_path: str,
    fact_root: str,
    output_path: str,
    *,
    splits=None,
    limit_videos=None,
    feature_fps: float = 10.0,
    target_fps: float = 4.0,
    trim_to_ground_truth: bool = True,
):
    specs = list(iter_unique_videos(manifest_path, splits=splits))
    if limit_videos is not None:
        specs = specs[: int(limit_videos)]
    if not specs:
        raise ValueError(f"No videos found in {manifest_path}")

    fact_root_path = Path(fact_root)
    features_dir = fact_root_path / "features"
    ground_truth_dir = fact_root_path / "groundTruth"

    available = []
    lengths = []
    feature_dim = None
    skipped_missing = 0
    missing_ground_truth = 0
    trimmed_by_ground_truth = 0

    for idx, spec in enumerate(specs, start=1):
        video_uid = spec["video_uid"]
        feature_path = features_dir / f"{video_uid}.npy"
        ground_truth_path = ground_truth_dir / f"{video_uid}.txt"
        if not feature_path.exists():
            skipped_missing += 1
            continue
        if not ground_truth_path.exists():
            missing_ground_truth += 1

        features, times, original_len, gt_len = load_resampled_feature(
            feature_path,
            ground_truth_path,
            feature_fps=feature_fps,
            target_fps=target_fps,
            trim_to_ground_truth=trim_to_ground_truth,
        )
        if gt_len is not None and int(gt_len) < int(original_len):
            trimmed_by_ground_truth += 1
        if feature_dim is None:
            feature_dim = int(features.shape[1])
        elif int(features.shape[1]) != int(feature_dim):
            raise ValueError(
                f"Inconsistent feature dim for {video_uid}: expected {feature_dim}, got {int(features.shape[1])}"
            )
        if int(features.shape[0]) != int(times.shape[0]):
            raise ValueError(f"Feature/time length mismatch for {video_uid}")
        lengths.append(int(features.shape[0]))
        available.append(spec)

        if idx % 100 == 0 or idx == len(specs):
            print(f"[scan] visited {idx}/{len(specs)} videos")

    if not available:
        raise ValueError(f"No matching FACT .npy features found under {features_dir}")

    offsets = torch.empty((len(available) + 1,), dtype=torch.long)
    offsets[0] = 0
    for idx, seq_len in enumerate(lengths, start=1):
        offsets[idx] = offsets[idx - 1] + int(seq_len)

    total_tokens = int(offsets[-1].item())
    tokens = torch.empty((total_tokens, int(feature_dim)), dtype=torch.float32)
    times = torch.empty((total_tokens,), dtype=torch.float32)

    for idx, spec in enumerate(available):
        video_uid = spec["video_uid"]
        feat_tensor, time_tensor, _original_len, _gt_len = load_resampled_feature(
            features_dir / f"{video_uid}.npy",
            ground_truth_dir / f"{video_uid}.txt",
            feature_fps=feature_fps,
            target_fps=target_fps,
            trim_to_ground_truth=trim_to_ground_truth,
        )
        start = int(offsets[idx].item())
        end = int(offsets[idx + 1].item())
        tokens[start:end] = feat_tensor
        times[start:end] = time_tensor
        if (idx + 1) % 100 == 0 or (idx + 1) == len(available):
            print(f"[build] packed {idx + 1}/{len(available)} videos into {output_path}")

    payload = {
        "version": "egoprocel_fact_npy_frame_pt_v1",
        "source_manifest": str(manifest_path),
        "features_root": str(features_dir),
        "paths": [spec["archive_rel_path"] for spec in available],
        "tokens": tokens,
        "times": times,
        "offsets": offsets,
        "feature_dim": int(feature_dim),
        "feature_fps": float(feature_fps),
        "target_fps": float(target_fps) if target_fps > 0 else float(feature_fps),
        "trim_to_ground_truth": bool(trim_to_ground_truth),
        "skipped_missing": int(skipped_missing),
        "missing_ground_truth": int(missing_ground_truth),
        "trimmed_by_ground_truth": int(trimmed_by_ground_truth),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(
        f"Wrote {output} with {len(available)} videos, {total_tokens} total tokens, "
        f"feature_dim={feature_dim}, target_fps={payload['target_fps']}, "
        f"skipped_missing={skipped_missing}, missing_ground_truth={missing_ground_truth}"
    )


def main():
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
    build_archive(
        manifest_path=args.manifest,
        fact_root=resolve_feature_path(args.fact_root),
        output_path=str(output_path),
        splits=args.splits,
        limit_videos=args.limit_videos,
        feature_fps=float(args.feature_fps),
        target_fps=float(args.target_fps),
        trim_to_ground_truth=not bool(args.no_trim_to_ground_truth),
    )


if __name__ == "__main__":
    main()
