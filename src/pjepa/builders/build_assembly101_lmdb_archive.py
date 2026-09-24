#!/usr/bin/env python3

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import lmdb
import numpy as np
import torch


@dataclass(frozen=True)
class VideoSpec:
    video_path: str
    archive_rel_path: str
    recording_id: str
    view: str
    start_frame: int
    end_frame: int
    source_fps: float


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pack Assembly101 LTContext-style LMDB frame features into a P-JEPA archive."
    )
    parser.add_argument("--manifest", required=True, help="Assembly101 manifest CSV.")
    parser.add_argument("--features-root", required=True, help="Root containing db_TSM_features/.")
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
        "--lmdb-subdir", default="db_TSM_features", help="LMDB subdirectory under features root."
    )
    parser.add_argument(
        "--frame-stride", type=int, default=8, help="Keep every Nth source feature frame."
    )
    parser.add_argument(
        "--target-fps", type=float, default=None, help="Optional metadata override for output fps."
    )
    parser.add_argument(
        "--source-fps", type=float, default=30.0, help="Fallback source feature fps."
    )
    parser.add_argument(
        "--feature-dim", type=int, default=2048, help="Expected LMDB feature dimension."
    )
    parser.add_argument(
        "--save-dtype",
        choices=("float32", "float16"),
        default="float32",
        help="Archive tensor dtype. The runtime loader casts back to float32.",
    )
    parser.add_argument("--video-path-column", default="video_path")
    parser.add_argument("--split-column", default="official_split")
    parser.add_argument("--start-frame-column", default="start_frame")
    parser.add_argument("--end-frame-column", default="end_frame")
    parser.add_argument("--source-fps-column", default="annotation_fps")
    parser.add_argument(
        "--skip-missing-frames",
        action="store_true",
        help="Skip missing sampled LMDB frames instead of failing.",
    )
    return parser.parse_args()


def archive_rel_path(video_path: str) -> str:
    path = Path(str(video_path).strip())
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".avi"}:
        return str(path.with_suffix(".pt"))
    return str(path)


def video_id_and_view(video_path: str):
    path = Path(str(video_path).strip())
    view = path.stem
    recording_id = str(path.parent)
    if not recording_id or recording_id == ".":
        raise ValueError(f"Assembly101 video_path must include recording/view: {video_path}")
    return recording_id, view


def maybe_float(value, default: float) -> float:
    if value in (None, ""):
        return float(default)
    return float(value)


def collect_video_specs(
    manifest_path: str,
    splits,
    video_path_column: str,
    split_column: str,
    start_frame_column: str,
    end_frame_column: str,
    source_fps_column: str,
    default_source_fps: float,
) -> List[VideoSpec]:
    split_filter = {str(split).lower() for split in splits} if splits else None
    grouped: Dict[str, Dict[str, object]] = {}

    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {video_path_column, start_frame_column, end_frame_column}
        missing = [column for column in required if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Manifest {manifest_path} is missing required columns: {missing}")

        for row in reader:
            if split_filter is not None:
                split = str(row.get(split_column, "")).strip().lower()
                if split not in split_filter:
                    continue

            video_path = str(row[video_path_column]).strip()
            start_frame = int(float(row[start_frame_column]))
            end_frame = int(float(row[end_frame_column]))
            if end_frame <= start_frame:
                continue
            source_fps = maybe_float(row.get(source_fps_column), default_source_fps)
            if source_fps <= 0:
                raise ValueError(f"Invalid source fps {source_fps} for {video_path}")

            item = grouped.setdefault(
                video_path,
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "source_fps": source_fps,
                },
            )
            item["start_frame"] = min(int(item["start_frame"]), start_frame)
            item["end_frame"] = max(int(item["end_frame"]), end_frame)
            if abs(float(item["source_fps"]) - source_fps) > 1e-6:
                raise ValueError(
                    f"Inconsistent source fps for {video_path}: {item['source_fps']} vs {source_fps}"
                )

    specs: List[VideoSpec] = []
    for video_path, item in grouped.items():
        recording_id, view = video_id_and_view(video_path)
        specs.append(
            VideoSpec(
                video_path=video_path,
                archive_rel_path=archive_rel_path(video_path),
                recording_id=recording_id,
                view=view,
                start_frame=int(item["start_frame"]),
                end_frame=int(item["end_frame"]),
                source_fps=float(item["source_fps"]),
            )
        )
    specs.sort(key=lambda spec: spec.archive_rel_path)
    return specs


def sampled_frame_indices(spec: VideoSpec, frame_stride: int) -> np.ndarray:
    if int(frame_stride) <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")
    frame_indices = np.arange(
        int(spec.start_frame),
        int(spec.end_frame),
        int(frame_stride),
        dtype=np.int64,
    )
    if frame_indices.size == 0:
        frame_indices = np.asarray([spec.start_frame], dtype=np.int64)
    return np.unique(frame_indices)


def frame_key(recording_id: str, view: str, frame_idx: int) -> bytes:
    return f"{recording_id}/{view}/{view}_{int(frame_idx):010d}.jpg".encode("utf-8")


def open_envs(features_root: str, lmdb_subdir: str, specs: List[VideoSpec]):
    envs = {}
    for view in sorted({spec.view for spec in specs}):
        env_path = Path(features_root) / lmdb_subdir / view
        if not env_path.exists():
            raise FileNotFoundError(f"Assembly101 LMDB view directory not found: {env_path}")
        envs[view] = lmdb.open(
            str(env_path), readonly=True, readahead=False, meminit=False, lock=False
        )
    return envs


def load_video_features(
    env,
    spec: VideoSpec,
    frame_indices: np.ndarray,
    feature_dim: int,
    skip_missing_frames: bool,
):
    rows = []
    kept_frames = []
    with env.begin(write=False) as txn:
        for frame_idx in frame_indices.tolist():
            data = txn.get(frame_key(spec.recording_id, spec.view, int(frame_idx)))
            if data is None:
                if skip_missing_frames:
                    continue
                raise KeyError(
                    f"No Assembly101 LMDB feature for "
                    f"{spec.recording_id}/{spec.view}/{spec.view}_{int(frame_idx):010d}.jpg"
                )
            arr = np.frombuffer(data, dtype=np.float32)
            if int(arr.shape[0]) != int(feature_dim):
                raise ValueError(
                    f"Expected feature_dim={feature_dim}, got {int(arr.shape[0])} for "
                    f"{spec.recording_id}/{spec.view}/{spec.view}_{int(frame_idx):010d}.jpg"
                )
            rows.append(arr)
            kept_frames.append(int(frame_idx))

    if not rows:
        raise ValueError(f"No sampled frames were available for {spec.video_path}")

    features = torch.from_numpy(np.stack(rows, axis=0).copy()).to(torch.float32)
    times = torch.as_tensor(
        np.asarray(kept_frames, dtype=np.float32) / float(spec.source_fps), dtype=torch.float32
    )
    return features, times


def build_archive(
    manifest_path: str,
    features_root: str,
    output_path: str,
    splits,
    limit_videos: Optional[int],
    lmdb_subdir: str,
    frame_stride: int,
    target_fps: Optional[float],
    source_fps: float,
    feature_dim: int,
    save_dtype: str,
    video_path_column: str,
    split_column: str,
    start_frame_column: str,
    end_frame_column: str,
    source_fps_column: str,
    skip_missing_frames: bool,
):
    specs = collect_video_specs(
        manifest_path=manifest_path,
        splits=splits,
        video_path_column=video_path_column,
        split_column=split_column,
        start_frame_column=start_frame_column,
        end_frame_column=end_frame_column,
        source_fps_column=source_fps_column,
        default_source_fps=source_fps,
    )
    if limit_videos is not None:
        specs = specs[: int(limit_videos)]
    if not specs:
        raise ValueError(f"No videos found in {manifest_path}")

    envs = open_envs(features_root, lmdb_subdir, specs)
    frame_indices_by_video = [sampled_frame_indices(spec, frame_stride) for spec in specs]
    lengths = [int(indices.shape[0]) for indices in frame_indices_by_video]

    offsets = torch.empty((len(specs) + 1,), dtype=torch.long)
    offsets[0] = 0
    for idx, seq_len in enumerate(lengths, start=1):
        offsets[idx] = offsets[idx - 1] + int(seq_len)

    total_tokens = int(offsets[-1].item())
    tensor_dtype = torch.float16 if save_dtype == "float16" else torch.float32
    tokens = torch.empty((total_tokens, int(feature_dim)), dtype=tensor_dtype)
    times = torch.empty((total_tokens,), dtype=torch.float32)

    for idx, (spec, frame_indices) in enumerate(zip(specs, frame_indices_by_video)):
        features, time_tensor = load_video_features(
            env=envs[spec.view],
            spec=spec,
            frame_indices=frame_indices,
            feature_dim=feature_dim,
            skip_missing_frames=skip_missing_frames,
        )
        start = int(offsets[idx].item())
        end = start + int(features.shape[0])
        if end != int(offsets[idx + 1].item()):
            delta = int(offsets[idx + 1].item()) - end
            if delta != 0:
                raise ValueError(
                    "Skipping missing frames changes sequence lengths. "
                    "Re-run without --skip-missing-frames or add a pre-scan path."
                )
        tokens[start:end] = features.to(dtype=tensor_dtype)
        times[start:end] = time_tensor
        if (idx + 1) % 25 == 0 or (idx + 1) == len(specs):
            print(
                f"[build] packed {idx + 1}/{len(specs)} videos, "
                f"tokens={end}/{total_tokens}, output={output_path}"
            )

    payload = {
        "version": "assembly101_tsm_lmdb_archive_v1",
        "source_manifest": str(manifest_path),
        "features_root": str(features_root),
        "lmdb_subdir": str(lmdb_subdir),
        "frame_stride": int(frame_stride),
        "target_fps": None if target_fps is None else float(target_fps),
        "effective_fps_fallback": float(source_fps) / float(frame_stride),
        "source_fps_fallback": float(source_fps),
        "paths": [spec.archive_rel_path for spec in specs],
        "video_paths": [spec.video_path for spec in specs],
        "tokens": tokens,
        "times": times,
        "offsets": offsets,
        "feature_dim": int(feature_dim),
        "save_dtype": str(save_dtype),
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(
        f"Wrote {output} with {len(specs)} videos, {total_tokens} total tokens, "
        f"feature_dim={feature_dim}, frame_stride={frame_stride}, save_dtype={save_dtype}"
    )


def main():
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
    build_archive(
        manifest_path=args.manifest,
        features_root=args.features_root,
        output_path=str(output_path),
        splits=args.splits,
        limit_videos=args.limit_videos,
        lmdb_subdir=args.lmdb_subdir,
        frame_stride=int(args.frame_stride),
        target_fps=None if args.target_fps is None else float(args.target_fps),
        source_fps=float(args.source_fps),
        feature_dim=int(args.feature_dim),
        save_dtype=str(args.save_dtype),
        video_path_column=str(args.video_path_column),
        split_column=str(args.split_column),
        start_frame_column=str(args.start_frame_column),
        end_frame_column=str(args.end_frame_column),
        source_fps_column=str(args.source_fps_column),
        skip_missing_frames=bool(args.skip_missing_frames),
    )


if __name__ == "__main__":
    main()
