from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


CHOLEC80_DEVELOPMENT_SPLITS = {
    "train": [f"video{index:02d}" for index in range(1, 33)],
    "val": [f"video{index:02d}" for index in range(33, 41)],
    "test": [f"video{index:02d}" for index in range(41, 81)],
}

CHOLEC80_TABLE1_SPLITS = {
    "train": [f"video{index:02d}" for index in range(1, 41)],
    "test": [f"video{index:02d}" for index in range(41, 81)],
}


class SurgicalPhasePLStitchArchive:
    """Memory-mapped PL-Stitch features with inline framewise phase labels."""

    def __init__(self, archive_path: str | Path, *, expected_dataset: str | None = None):
        self.archive_path = Path(archive_path).expanduser().resolve()
        self.payload = torch.load(
            str(self.archive_path),
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        required = {
            "dataset",
            "video_ids",
            "source_splits",
            "tokens",
            "offsets",
            "frame_indices",
            "phase_labels",
            "phase_class_names",
            "feature_dim",
            "target_fps",
        }
        missing = sorted(required.difference(self.payload))
        if missing:
            raise ValueError(f"Surgical phase archive is missing keys: {missing}")

        self.dataset = str(self.payload["dataset"]).lower()
        if expected_dataset is not None and self.dataset != str(expected_dataset).lower():
            raise ValueError(
                f"Expected dataset={expected_dataset!r}, archive contains {self.dataset!r}."
            )
        if self.dataset not in {"cholec80", "m2cai16"}:
            raise ValueError(f"Unsupported surgical phase dataset: {self.dataset!r}")

        self.video_ids = [str(value) for value in self.payload["video_ids"]]
        self.video_to_index = {video_id: index for index, video_id in enumerate(self.video_ids)}
        if len(self.video_to_index) != len(self.video_ids):
            raise ValueError("Surgical phase archive contains duplicate video IDs.")
        self.source_splits = [str(value).lower() for value in self.payload["source_splits"]]
        if len(self.source_splits) != len(self.video_ids):
            raise ValueError("Archive source split/video counts differ.")

        self.offsets = torch.as_tensor(self.payload["offsets"], dtype=torch.long)
        self.tokens = self.payload["tokens"]
        self.frame_indices = torch.as_tensor(self.payload["frame_indices"], dtype=torch.long)
        self.phase_labels = torch.as_tensor(self.payload["phase_labels"], dtype=torch.long)
        self.phase_class_names = [str(value) for value in self.payload["phase_class_names"]]
        self.feature_dim = int(self.payload["feature_dim"])
        self.target_fps = float(self.payload["target_fps"])

        total_tokens = int(self.tokens.shape[0])
        if not (int(self.frame_indices.numel()) == int(self.phase_labels.numel()) == total_tokens):
            raise ValueError("Archive token, frame-index and phase-label lengths differ.")
        if int(self.offsets.numel()) != len(self.video_ids) + 1:
            raise ValueError("Archive offsets must contain one boundary per video.")
        if int(self.offsets[0].item()) != 0 or int(self.offsets[-1].item()) != total_tokens:
            raise ValueError("Archive offsets do not span the token tensor.")
        if bool((self.offsets[1:] <= self.offsets[:-1]).any().item()):
            raise ValueError("Every archive video must contain at least one token.")
        num_classes = len(self.phase_class_names)
        if bool(((self.phase_labels < 0) | (self.phase_labels >= num_classes)).any().item()):
            raise ValueError("Archive contains out-of-range phase labels.")

    @property
    def num_classes(self) -> int:
        return len(self.phase_class_names)

    def video_length(self, video_index: int) -> int:
        return int((self.offsets[int(video_index) + 1] - self.offsets[int(video_index)]).item())

    def video_ids_for_split(self, *, protocol: str, split: str) -> list[str]:
        protocol = str(protocol).lower()
        split = str(split).lower()
        if self.dataset == "cholec80":
            if protocol == "development":
                split_map = CHOLEC80_DEVELOPMENT_SPLITS
            elif protocol == "table1":
                split_map = CHOLEC80_TABLE1_SPLITS
            else:
                raise ValueError("Cholec80 protocol must be 'development' or 'table1'.")
            if split not in split_map:
                raise ValueError(f"Cholec80 protocol={protocol!r} has no split={split!r}.")
            selected = list(split_map[split])
        else:
            if protocol != "official":
                raise ValueError("M2CAI16 protocol must be 'official'.")
            if split not in {"train", "test"}:
                raise ValueError("M2CAI16 split must be 'train' or 'test'.")
            selected = [
                video_id
                for video_id, source_split in zip(self.video_ids, self.source_splits, strict=True)
                if source_split == split
            ]
        missing = [video_id for video_id in selected if video_id not in self.video_to_index]
        if missing:
            raise ValueError(
                f"Archive is missing {len(missing)} {self.dataset}/{protocol}/{split} "
                f"videos: {missing[:10]}"
            )
        if not selected:
            raise ValueError(f"No videos found for {self.dataset}/{protocol}/{split}.")
        return selected

    def fetch_video(self, video_index: int) -> dict[str, Any]:
        video_index = int(video_index)
        start = int(self.offsets[video_index].item())
        end = int(self.offsets[video_index + 1].item())
        return {
            "features": self.tokens[start:end].to(torch.float32).clone(),
            "frame_indices": self.frame_indices[start:end].clone(),
            "targets": {"phase": self.phase_labels[start:end].clone()},
            "video_id": self.video_ids[video_index],
            "window_start": 0,
        }


class SurgicalPhaseVideoDataset(Dataset):
    """One complete video per item; downstream inference never crops sequences."""

    def __init__(
        self,
        archive_path: str | Path,
        *,
        dataset: str,
        protocol: str,
        split: str,
        video_ids: list[str] | None = None,
    ):
        self.archive = SurgicalPhasePLStitchArchive(archive_path, expected_dataset=dataset)
        self.protocol = str(protocol).lower()
        self.split = str(split).lower()
        split_video_ids = self.archive.video_ids_for_split(protocol=self.protocol, split=self.split)
        if video_ids is None:
            self.video_ids = split_video_ids
        else:
            requested = [str(value) for value in video_ids]
            if not requested:
                raise ValueError("Explicit surgical-phase video selection is empty.")
            if len(set(requested)) != len(requested):
                raise ValueError("Explicit surgical-phase video selection has duplicates.")
            allowed = set(split_video_ids)
            invalid = [video_id for video_id in requested if video_id not in allowed]
            if invalid:
                raise ValueError(
                    f"Videos are not part of {self.archive.dataset}/{self.protocol}/"
                    f"{self.split}: {invalid}"
                )
            self.video_ids = requested
        self.video_indices = [self.archive.video_to_index[video_id] for video_id in self.video_ids]
        self.sequence_lengths = [
            self.archive.video_length(video_index) for video_index in self.video_indices
        ]

    def __len__(self) -> int:
        return len(self.video_indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        result = self.archive.fetch_video(self.video_indices[int(item)])
        valid_length = int(result["features"].shape[0])
        result["valid_mask"] = torch.ones(valid_length, dtype=torch.bool)
        result["valid_length"] = valid_length
        result["sample_name"] = str(result["video_id"])
        return result


def collate_surgical_phase_videos(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty surgical phase batch.")
    batch_size = len(batch)
    max_length = max(int(item["valid_length"]) for item in batch)
    feature_dim = int(batch[0]["features"].shape[-1])
    features = torch.zeros(batch_size, max_length, feature_dim, dtype=torch.float32)
    valid_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    frame_indices = torch.full((batch_size, max_length), -1, dtype=torch.long)
    phase = torch.full((batch_size, max_length), -100, dtype=torch.long)
    for batch_index, item in enumerate(batch):
        length = int(item["valid_length"])
        features[batch_index, :length] = item["features"]
        valid_mask[batch_index, :length] = True
        frame_indices[batch_index, :length] = item["frame_indices"]
        phase[batch_index, :length] = item["targets"]["phase"]
    return {
        "features": features,
        "valid_mask": valid_mask,
        "valid_length": torch.as_tensor(
            [int(item["valid_length"]) for item in batch], dtype=torch.long
        ),
        "frame_indices": frame_indices,
        "targets": {"phase": phase},
        "video_id": [str(item["video_id"]) for item in batch],
        "window_start": torch.zeros(batch_size, dtype=torch.long),
        "sample_name": [str(item["sample_name"]) for item in batch],
    }
