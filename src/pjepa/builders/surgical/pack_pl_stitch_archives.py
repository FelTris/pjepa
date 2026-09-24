#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


from pjepa.builders.surgical import (
    FEATURE_VERSION,
    HECVL_FEATURE_VERSION,
    HECVL_INDEX_VERSION,
    INDEX_VERSION,
)
from pjepa.builders.surgical.common import (
    atomic_json_dump,
    atomic_torch_save,
    parse_split_list,
    per_video_feature_path,
    read_manifest,
    torch_load,
    validate_feature_payload,
)


@dataclass(frozen=True)
class VideoSpec:
    video_id: str
    split: str
    relative_path: str
    feature_path: Path
    num_tokens: int
    duration_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack per-video FP32 PL-Stitch or HecVL features into bounded shards."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("lemon_dataset/manifests/lemon_manifest.csv"),
    )
    parser.add_argument(
        "--per-video-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--splits", default="pretrain")
    parser.add_argument("--target-fps", type=float, default=4.0)
    parser.add_argument("--target-tokens-per-shard", type=int, default=500_000)
    parser.add_argument(
        "--backbone",
        choices=("auto", "pl_stitch", "hecvl"),
        default="auto",
        help="Validate the feature backbone; auto infers it from each payload.",
    )
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Pack available files and report missing videos.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def scan_specs(
    args: argparse.Namespace,
) -> tuple[list[VideoSpec], list[str], dict[str, Any]]:
    requested_splits = parse_split_list(args.splits)
    rows = [
        row
        for row in read_manifest(args.manifest)
        if row["split"].strip().lower() in requested_splits
    ]
    if args.max_videos > 0:
        rows = rows[: args.max_videos]
    specs: list[VideoSpec] = []
    missing: list[str] = []
    common_metadata: dict[str, Any] | None = None
    for row in tqdm(rows, desc="Scanning per-video features", unit="video"):
        feature_path = per_video_feature_path(args.per_video_root, row["video_id"])
        if not feature_path.is_file():
            missing.append(row["video_id"])
            continue
        payload = torch_load(feature_path, mmap=True)
        features, _, _ = validate_feature_payload(
            payload,
            expected_fps=args.target_fps,
            require_finite=False,
        )
        feature_type = str(payload.get("feature_type", "final_cls"))
        inferred_backbone = str(payload.get("backbone", "")).strip().lower()
        if not inferred_backbone:
            inferred_backbone = "hecvl" if feature_type == "hecvl_img_emb" else "pl_stitch"
        if args.backbone != "auto" and inferred_backbone != args.backbone:
            raise ValueError(
                f"Expected backbone={args.backbone}, got {inferred_backbone} in {feature_path}"
            )
        source_version = str(
            payload.get(
                "version",
                HECVL_FEATURE_VERSION if inferred_backbone == "hecvl" else FEATURE_VERSION,
            )
        )
        metadata = {
            "backbone": inferred_backbone,
            "source_version": source_version,
            "checkpoint_sha256": str(payload.get("checkpoint_sha256", "")),
            "pl_stitch_git_commit": str(payload.get("pl_stitch_git_commit", "")),
            "surgvlp_git_commit": str(payload.get("surgvlp_git_commit", "")),
            "target_fps": float(payload["target_fps"]),
            "feature_dim": int(features.shape[1]),
            "feature_type": feature_type,
            "storage_dtype": str(payload.get("storage_dtype", "float32")),
            "compute_dtype": str(payload.get("compute_dtype", "float32")),
            "preprocessing": payload.get("preprocessing", {}),
        }
        if common_metadata is None:
            common_metadata = metadata
        elif metadata != common_metadata:
            raise ValueError(
                f"Metadata mismatch in {feature_path}: {metadata} != {common_metadata}"
            )
        specs.append(
            VideoSpec(
                video_id=row["video_id"],
                split=row["split"].strip().lower(),
                relative_path=row["relative_path"],
                feature_path=feature_path,
                num_tokens=int(features.shape[0]),
                duration_sec=float(row["duration_sec"]),
            )
        )
    if missing and not args.allow_missing:
        raise FileNotFoundError(
            f"{len(missing)} per-video feature files are missing; first missing IDs: {missing[:10]}. "
            "Use --allow-missing for a deliberate pilot archive."
        )
    if not specs:
        raise RuntimeError("No per-video features were available to pack.")
    assert common_metadata is not None
    return specs, missing, common_metadata


def group_shards(specs: list[VideoSpec], target_tokens: int) -> dict[str, list[list[VideoSpec]]]:
    grouped: dict[str, list[list[VideoSpec]]] = {}
    for split in sorted({spec.split for spec in specs}):
        split_specs = [spec for spec in specs if spec.split == split]
        shards: list[list[VideoSpec]] = []
        current: list[VideoSpec] = []
        current_tokens = 0
        for spec in split_specs:
            if current and current_tokens + spec.num_tokens > target_tokens:
                shards.append(current)
                current = []
                current_tokens = 0
            current.append(spec)
            current_tokens += spec.num_tokens
        if current:
            shards.append(current)
        grouped[split] = shards
    return grouped


def build_shard_payload(
    specs: list[VideoSpec],
    *,
    common_metadata: dict[str, Any],
    source_manifest: Path,
) -> dict[str, Any]:
    offsets = torch.zeros(len(specs) + 1, dtype=torch.int64)
    for index, spec in enumerate(specs, start=1):
        offsets[index] = offsets[index - 1] + spec.num_tokens
    total_tokens = int(offsets[-1].item())
    feature_dim = int(common_metadata["feature_dim"])
    tokens = torch.empty((total_tokens, feature_dim), dtype=torch.float32)
    times = torch.empty((total_tokens,), dtype=torch.float32)
    nominal_times = torch.empty((total_tokens,), dtype=torch.float32)

    for index, spec in enumerate(tqdm(specs, desc="Filling shard", unit="video", leave=False)):
        payload = torch_load(spec.feature_path, mmap=True)
        features, video_times, video_nominal_times = validate_feature_payload(
            payload,
            expected_fps=float(common_metadata["target_fps"]),
            expected_dim=feature_dim,
        )
        start = int(offsets[index].item())
        end = int(offsets[index + 1].item())
        tokens[start:end].copy_(features)
        times[start:end].copy_(video_times)
        nominal_times[start:end].copy_(video_nominal_times)

    return {
        "version": str(common_metadata["source_version"]),
        "source_manifest": str(source_manifest.resolve()),
        "split": specs[0].split,
        "paths": [spec.relative_path for spec in specs],
        "video_ids": [spec.video_id for spec in specs],
        "tokens": tokens,
        "times": times,
        "nominal_times": nominal_times,
        "offsets": offsets,
        "durations_sec": torch.tensor([spec.duration_sec for spec in specs], dtype=torch.float64),
        **common_metadata,
    }


def main() -> int:
    args = parse_args()
    if args.target_tokens_per_shard <= 0 or args.target_fps <= 0:
        raise ValueError("target-tokens-per-shard and target-fps must be positive.")
    specs, missing, common_metadata = scan_specs(args)
    grouped = group_shards(specs, args.target_tokens_per_shard)

    output_root = args.output_root.expanduser().resolve()
    index_path = output_root / "index.pt"
    report_path = output_root / "packing_summary.json"
    planned_paths = [
        output_root / split / f"{split}_{shard_index:05d}.pt"
        for split, shards in grouped.items()
        for shard_index in range(len(shards))
    ]
    existing = [path for path in [index_path, *planned_paths] if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Outputs already exist; pass --overwrite to replace them: {existing[:5]}"
        )

    shard_paths: list[str] = []
    global_video_ids: list[str] = []
    global_paths: list[str] = []
    global_splits: list[str] = []
    global_shard_ids: list[int] = []
    global_local_indices: list[int] = []
    global_num_tokens: list[int] = []
    global_durations: list[float] = []

    for split, shards in grouped.items():
        for shard_index, shard_specs in enumerate(shards):
            shard_path = output_root / split / f"{split}_{shard_index:05d}.pt"
            payload = build_shard_payload(
                shard_specs,
                common_metadata=common_metadata,
                source_manifest=args.manifest,
            )
            atomic_torch_save(payload, shard_path)
            relative_shard_path = shard_path.relative_to(output_root).as_posix()
            global_shard_id = len(shard_paths)
            shard_paths.append(relative_shard_path)
            for local_index, spec in enumerate(shard_specs):
                global_video_ids.append(spec.video_id)
                global_paths.append(spec.relative_path)
                global_splits.append(spec.split)
                global_shard_ids.append(global_shard_id)
                global_local_indices.append(local_index)
                global_num_tokens.append(spec.num_tokens)
                global_durations.append(spec.duration_sec)
            print(
                f"Saved {shard_path}: videos={len(shard_specs)} "
                f"tokens={int(payload['tokens'].shape[0])} dtype={payload['tokens'].dtype}"
            )

    index_version = HECVL_INDEX_VERSION if common_metadata["backbone"] == "hecvl" else INDEX_VERSION
    index_payload = {
        "version": index_version,
        "source_version": str(common_metadata["source_version"]),
        "source_manifest": str(args.manifest.resolve()),
        "shard_root": str(output_root),
        "shards": shard_paths,
        "video_ids": global_video_ids,
        "paths": global_paths,
        "splits": global_splits,
        "shard_ids": torch.tensor(global_shard_ids, dtype=torch.int64),
        "local_indices": torch.tensor(global_local_indices, dtype=torch.int64),
        "num_tokens": torch.tensor(global_num_tokens, dtype=torch.int64),
        "durations_sec": torch.tensor(global_durations, dtype=torch.float64),
        **common_metadata,
    }
    atomic_torch_save(index_payload, index_path)
    summary = {
        "index": str(index_path),
        "videos": len(global_video_ids),
        "tokens": sum(global_num_tokens),
        "shards": len(shard_paths),
        "missing_videos": missing,
        "storage_dtype": "float32",
        "target_fps": args.target_fps,
    }
    atomic_json_dump(summary, report_path)
    print(f"Saved {index_path}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
