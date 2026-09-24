#!/usr/bin/env python3

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


FIELDNAMES = [
    "sample_uid",
    "video_uid",
    "video_path",
    "annotation_path",
    "fact_feature_path",
    "fact_ground_truth_path",
    "source_dataset",
    "procedure_name",
    "task_name",
    "participant_id",
    "group_id",
    "fact_split",
    "official_split",
    "raw_step_id",
    "raw_step_name",
    "step_name_raw",
    "step_name_norm",
    "task_step_key",
    "task_step_id",
    "fact_step_name",
    "fact_step_id",
    "start_sec",
    "end_sec",
    "duration_sec",
    "video_duration_sec",
    "fps",
    "num_frames",
    "feature_num_frames",
    "is_background",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build an EgoProceL segment manifest directly from FACT features, "
            "groundTruth labels, and split files."
        )
    )
    parser.add_argument(
        "--fact-root",
        default="FACT_actseg/data/egoprocel",
        help="FACT EgoProceL root containing features/, groundTruth/, mapping.txt, and split files.",
    )
    parser.add_argument(
        "--output",
        default="egoprocel_train/data/egoprocel_fact_features_segments.csv",
        help="Output CSV manifest.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Annotation/feature FPS used by FACT groundTruth files.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=("train", "test"),
        help="FACT split names to include. Looks for split1.<name> under fact-root.",
    )
    parser.add_argument(
        "--include-unlisted-features",
        action="store_true",
        help="Also include feature/groundTruth pairs not present in the requested split files.",
    )
    return parser.parse_args()


def load_mapping(path: Path):
    id_to_name = {}
    name_to_id = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split(maxsplit=1)
            label_id = int(idx)
            id_to_name[label_id] = name
            name_to_id[name] = label_id
    return id_to_name, name_to_id


def read_split(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stem = line.strip()
            if not stem:
                continue
            for suffix in (".txt", ".npy"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
            rows.append(stem)
    return rows


def infer_source_dataset(video_uid: str):
    if video_uid.startswith("S") and "_" in video_uid:
        return "CMU_Kitchens"
    if video_uid.startswith(("P", "OP")) and "-R" in video_uid:
        return "EGTEA_Gaze+"
    if ".tent." in video_uid:
        return "EPIC-Tents"
    if video_uid.startswith("Head_") or video_uid.startswith("pc_"):
        return "pc_assembly/disassembly"
    if video_uid.isdigit() and len(video_uid) == 4:
        return "MECCANO"
    return "unknown"


def infer_procedure_name(video_uid: str, source_dataset: str):
    if source_dataset == "CMU_Kitchens":
        parts = video_uid.split("_")
        return parts[1] if len(parts) > 1 else ""
    if source_dataset == "EGTEA_Gaze+":
        return video_uid.split("-")[-1]
    if source_dataset == "EPIC-Tents":
        return "Tents"
    if source_dataset == "MECCANO":
        return "MECCANO"
    if source_dataset == "pc_assembly/disassembly":
        return "pc_disassembly" if video_uid.startswith("Head_") else "pc_assembly"
    return ""


def infer_participant_id(video_uid: str, source_dataset: str):
    if source_dataset == "CMU_Kitchens":
        return video_uid.split("_", 1)[0]
    if source_dataset == "EGTEA_Gaze+":
        return video_uid.split("-", 1)[0]
    if source_dataset == "EPIC-Tents":
        return video_uid.split(".", 1)[0]
    if source_dataset == "MECCANO":
        return video_uid
    if source_dataset == "pc_assembly/disassembly":
        return video_uid
    return ""


def iter_runs(labels):
    if not labels:
        return
    start = 0
    current = labels[0]
    for idx, label in enumerate(labels[1:], start=1):
        if label != current:
            yield start, idx, current
            start = idx
            current = label
    yield start, len(labels), current


def load_labels(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def feature_len(path: Path):
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D FACT feature array at {path}, got shape {arr.shape}")
    return int(arr.shape[0])


def build_manifest(
    fact_root: Path, output: Path, splits, fps: float, include_unlisted_features: bool
):
    features_dir = fact_root / "features"
    ground_truth_dir = fact_root / "groundTruth"
    id_to_name, name_to_id = load_mapping(fact_root / "mapping.txt")

    split_for_video = {}
    ordered_videos = []
    for split in splits:
        split_path = fact_root / f"split1.{split}"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing FACT split file: {split_path}")
        for video_uid in read_split(split_path):
            if video_uid in split_for_video:
                raise ValueError(f"Video {video_uid} appears in multiple requested splits.")
            split_for_video[video_uid] = split
            ordered_videos.append(video_uid)

    if include_unlisted_features:
        listed = set(ordered_videos)
        for feature_path in sorted(features_dir.glob("*.npy")):
            video_uid = feature_path.stem
            if video_uid not in listed and (ground_truth_dir / f"{video_uid}.txt").exists():
                split_for_video[video_uid] = "unsplit"
                ordered_videos.append(video_uid)

    output.parent.mkdir(parents=True, exist_ok=True)
    skipped = Counter()
    rows_written = 0
    videos_written = 0
    split_counts = Counter()
    segment_counts = Counter()

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()

        for video_uid in ordered_videos:
            feature_path = features_dir / f"{video_uid}.npy"
            gt_path = ground_truth_dir / f"{video_uid}.txt"
            if not feature_path.exists():
                skipped["missing_feature"] += 1
                continue
            if not gt_path.exists():
                skipped["missing_ground_truth"] += 1
                continue

            labels = load_labels(gt_path)
            if not labels:
                skipped["empty_ground_truth"] += 1
                continue

            feat_len = feature_len(feature_path)
            source_dataset = infer_source_dataset(video_uid)
            procedure_name = infer_procedure_name(video_uid, source_dataset)
            participant_id = infer_participant_id(video_uid, source_dataset)
            task_name = f"{source_dataset}::{procedure_name}" if procedure_name else source_dataset
            group_id = (
                f"{source_dataset}::{procedure_name}::{participant_id}"
                if participant_id
                else f"{source_dataset}::{procedure_name}"
            )
            split = split_for_video[video_uid]
            duration_sec = len(labels) / float(fps)
            run_index = 0

            for start, end, label_name in iter_runs(labels):
                if label_name not in name_to_id:
                    raise ValueError(
                        f"Label {label_name!r} in {gt_path} is not present in mapping.txt"
                    )
                label_id = int(name_to_id[label_name])
                start_sec = start / float(fps)
                end_sec = end / float(fps)
                is_background = int(label_id == 0 or label_name == "background")
                sample_uid = (
                    f"{video_uid}_{start_sec:.3f}_{end_sec:.3f}_{label_name}_{run_index:03d}"
                )
                writer.writerow(
                    {
                        "sample_uid": sample_uid,
                        "video_uid": video_uid,
                        "video_path": f"{video_uid}.pt",
                        "annotation_path": "",
                        "fact_feature_path": f"features/{video_uid}.npy",
                        "fact_ground_truth_path": f"groundTruth/{video_uid}.txt",
                        "source_dataset": source_dataset,
                        "procedure_name": procedure_name,
                        "task_name": task_name,
                        "participant_id": participant_id,
                        "group_id": group_id,
                        "fact_split": split,
                        "official_split": split,
                        "raw_step_id": label_id,
                        "raw_step_name": id_to_name.get(label_id, label_name),
                        "step_name_raw": id_to_name.get(label_id, label_name),
                        "step_name_norm": label_name,
                        "task_step_key": f"{task_name}::{label_name}",
                        "task_step_id": label_id,
                        "fact_step_name": label_name,
                        "fact_step_id": label_id,
                        "start_sec": f"{start_sec:.6f}",
                        "end_sec": f"{end_sec:.6f}",
                        "duration_sec": f"{end_sec - start_sec:.6f}",
                        "video_duration_sec": f"{duration_sec:.6f}",
                        "fps": f"{float(fps):.6f}",
                        "num_frames": len(labels),
                        "feature_num_frames": feat_len,
                        "is_background": is_background,
                    }
                )
                rows_written += 1
                run_index += 1

            videos_written += 1
            split_counts[split] += 1
            segment_counts[split] += run_index

    print(f"Wrote {output}")
    print(f"Videos: {videos_written}; segments: {rows_written}; skipped: {dict(skipped)}")
    print(f"Videos by split: {dict(sorted(split_counts.items()))}")
    print(f"Segments by split: {dict(sorted(segment_counts.items()))}")


def main():
    args = parse_args()
    build_manifest(
        fact_root=Path(args.fact_root),
        output=Path(args.output),
        splits=tuple(args.splits),
        fps=float(args.fps),
        include_unlisted_features=bool(args.include_unlisted_features),
    )


if __name__ == "__main__":
    main()
