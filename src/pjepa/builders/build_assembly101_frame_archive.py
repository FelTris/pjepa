#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import torch


from pjepa.data.common import resolve_feature_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pack Assembly101 per-video frame feature tensors into a single archive."
    )
    parser.add_argument("--manifest", required=True, help="Assembly101 manifest CSV.")
    parser.add_argument(
        "--features-root", required=True, help="Root containing per-video .pt feature files."
    )
    parser.add_argument("--output", required=True, help="Output archive path.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite an existing output archive."
    )
    parser.add_argument(
        "--splits", nargs="*", default=None, help="Optional official splits to include."
    )
    parser.add_argument(
        "--limit-videos", type=int, default=None, help="Optional limit for smoke tests."
    )
    return parser.parse_args()


def feature_rel_path(video_path: str) -> str:
    path = str(video_path).strip()
    suffix = Path(path).suffix.lower()
    if suffix in {".mp4", ".avi"}:
        return str(Path(path).with_suffix(".pt"))
    return path


def iter_unique_feature_paths(manifest_path: str, splits=None):
    splits = {split.lower() for split in splits} if splits else None
    seen = set()
    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if splits is not None:
                split = str(row.get("official_split", "")).strip().lower()
                if split not in splits:
                    continue
            rel_path = feature_rel_path(row["video_path"])
            if rel_path in seen:
                continue
            seen.add(rel_path)
            yield rel_path


def load_feature_payload(path: str):
    payload = torch.load(path, map_location="cpu")
    if (
        not isinstance(payload, dict)
        or "features" not in payload
        or "feature_times_sec" not in payload
    ):
        raise ValueError(f"Expected dict payload with 'features' and 'feature_times_sec' in {path}")
    features = payload["features"]
    times = payload["feature_times_sec"]
    if not torch.is_tensor(features) or features.ndim != 2:
        raise ValueError(f"Features at {path} must have shape [T, D], got {tuple(features.shape)}")
    if not torch.is_tensor(times) or times.ndim != 1:
        raise ValueError(f"Feature times at {path} must have shape [T], got {tuple(times.shape)}")
    if int(features.shape[0]) != int(times.shape[0]):
        raise ValueError(
            f"Feature/time length mismatch at {path}: {tuple(features.shape)} vs {tuple(times.shape)}"
        )
    return (
        features.to(dtype=torch.float32, device="cpu").contiguous(),
        times.to(dtype=torch.float32, device="cpu").contiguous(),
    )


def build_archive(
    manifest_path: str, features_root: str, output_path: str, splits=None, limit_videos=None
):
    rel_paths = list(iter_unique_feature_paths(manifest_path, splits=splits))
    if limit_videos is not None:
        rel_paths = rel_paths[: int(limit_videos)]
    if not rel_paths:
        raise ValueError(f"No feature paths found in {manifest_path}")

    available_paths = []
    lengths = []
    feature_dim = None
    skipped_missing = 0

    for idx, rel_path in enumerate(rel_paths, start=1):
        full_path = resolve_feature_path(rel_path, features_root)
        if not Path(full_path).exists():
            skipped_missing += 1
            continue
        features, times = load_feature_payload(full_path)
        if feature_dim is None:
            feature_dim = int(features.shape[1])
        elif int(features.shape[1]) != feature_dim:
            raise ValueError(
                f"Inconsistent feature dim for {rel_path}: expected {feature_dim}, got {int(features.shape[1])}"
            )
        lengths.append(int(features.shape[0]))
        available_paths.append(rel_path)
        if idx % 100 == 0 or idx == len(rel_paths):
            print(f"[scan] visited {idx}/{len(rel_paths)} videos")

    offsets = torch.empty((len(available_paths) + 1,), dtype=torch.long)
    offsets[0] = 0
    for idx, seq_len in enumerate(lengths, start=1):
        offsets[idx] = offsets[idx - 1] + int(seq_len)

    total_tokens = int(offsets[-1].item())
    tokens = torch.empty((total_tokens, feature_dim), dtype=torch.float32)
    times = torch.empty((total_tokens,), dtype=torch.float32)

    for idx, rel_path in enumerate(available_paths):
        full_path = resolve_feature_path(rel_path, features_root)
        feat_tensor, time_tensor = load_feature_payload(full_path)
        start = int(offsets[idx].item())
        end = int(offsets[idx + 1].item())
        tokens[start:end] = feat_tensor
        times[start:end] = time_tensor
        if (idx + 1) % 100 == 0 or (idx + 1) == len(available_paths):
            print(f"[build] packed {idx + 1}/{len(available_paths)} videos into {output_path}")

    payload = {
        "version": "assembly101_frame_pt_v1",
        "source_manifest": str(manifest_path),
        "features_root": str(features_root),
        "paths": available_paths,
        "tokens": tokens,
        "times": times,
        "offsets": offsets,
        "feature_dim": int(feature_dim),
        "skipped_missing": int(skipped_missing),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(
        f"Wrote {output} with {len(available_paths)} videos, {total_tokens} total tokens, "
        f"feature_dim={feature_dim}, skipped_missing={skipped_missing}"
    )


def main():
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
    build_archive(
        manifest_path=args.manifest,
        features_root=args.features_root,
        output_path=str(args.output),
        splits=args.splits,
        limit_videos=args.limit_videos,
    )


if __name__ == "__main__":
    main()
