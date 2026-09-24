from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import math
import random
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler


def _torch_load(path: str | Path, *, mmap: bool = False):
    return torch.load(
        str(path),
        map_location="cpu",
        mmap=bool(mmap),
        weights_only=False,
    )


class ShardedLemonArchive:
    """Bounded-memory reader for the packed LEMON PL-Stitch archive."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        max_cached_shards: int = 1,
        mmap: bool = True,
    ):
        if int(max_cached_shards) < 1:
            raise ValueError("max_cached_shards must be at least one.")
        self.index_path = Path(index_path).expanduser().resolve()
        self.index = _torch_load(self.index_path, mmap=mmap)
        if not isinstance(self.index, dict):
            raise ValueError(f"LEMON index must be a dictionary: {self.index_path}")

        required = {
            "shards",
            "shard_ids",
            "local_indices",
            "video_ids",
            "paths",
            "splits",
            "num_tokens",
            "feature_dim",
            "target_fps",
        }
        missing = sorted(required.difference(self.index))
        if missing:
            raise ValueError(f"LEMON index is missing keys: {missing}")

        self.shard_root = self.index_path.parent
        # Resolve shards next to the supplied index. Stored absolute roots refer
        # to the machine that built the archive and must not override relocation.
        self.max_cached_shards = int(max_cached_shards)
        self.mmap = bool(mmap)
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.video_ids = [str(value) for value in self.index["video_ids"]]
        self.paths = [str(value) for value in self.index["paths"]]

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        return state

    @property
    def feature_dim(self) -> int:
        return int(self.index["feature_dim"])

    @property
    def target_fps(self) -> float:
        return float(self.index["target_fps"])

    def _load_shard(self, shard_id: int) -> dict[str, Any]:
        shard_id = int(shard_id)
        if shard_id in self._cache:
            payload = self._cache.pop(shard_id)
            self._cache[shard_id] = payload
            return payload

        relative = Path(str(self.index["shards"][shard_id]))
        shard_path = relative if relative.is_absolute() else self.shard_root / relative
        payload = _torch_load(shard_path, mmap=self.mmap)
        if not isinstance(payload, dict):
            raise ValueError(f"LEMON shard must be a dictionary: {shard_path}")
        self._cache[shard_id] = payload
        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)
        return payload

    def get_video_window(
        self,
        global_index: int,
        *,
        sample_rate: int,
        sampled_start: int,
        window_size: int,
    ) -> dict[str, Any]:
        sample_rate = int(sample_rate)
        sampled_start = int(sampled_start)
        window_size = int(window_size)
        if sample_rate < 1 or sampled_start < 0 or window_size < 1:
            raise ValueError(
                "sample_rate/window_size must be positive and sampled_start non-negative."
            )

        shard_id = int(self.index["shard_ids"][global_index].item())
        local_index = int(self.index["local_indices"][global_index].item())
        shard = self._load_shard(shard_id)
        offsets = shard["offsets"]
        raw_start = int(offsets[local_index].item())
        raw_end = int(offsets[local_index + 1].item())
        sampled_length = math.ceil((raw_end - raw_start) / sample_rate)
        if sampled_start >= sampled_length:
            raise IndexError(
                f"sampled_start={sampled_start} exceeds video length {sampled_length}."
            )

        selection_start = raw_start + sampled_start * sample_rate
        selection_end = min(
            raw_end,
            selection_start + (window_size - 1) * sample_rate + 1,
        )
        selection = slice(selection_start, selection_end, sample_rate)
        return {
            "features": shard["tokens"][selection].to(torch.float32).clone(),
            "times": shard["times"][selection].to(torch.float32).clone(),
            "video_id": self.video_ids[global_index],
            "path": self.paths[global_index],
            "start_token": sampled_start,
            "effective_fps": self.target_fps / sample_rate,
        }


class LemonVideoDataset(Dataset):
    """One 1D sample per video, capped only when a sequence is exceptionally long."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        split: str = "pretrain",
        sample_rate: int = 4,
        max_sequence_len: int = 4096,
        seed: int = 1538574472,
        max_cached_shards: int = 1,
        mmap: bool = True,
    ):
        self.archive = ShardedLemonArchive(
            index_path,
            max_cached_shards=max_cached_shards,
            mmap=mmap,
        )
        self.split = str(split).lower()
        self.sample_rate = int(sample_rate)
        self.max_sequence_len = int(max_sequence_len)
        self.seed = int(seed)
        self.epoch = 0
        if self.sample_rate < 1 or self.max_sequence_len < 2:
            raise ValueError(
                "sample_rate must be positive and max_sequence_len must be at least two."
            )

        splits = [str(value).lower() for value in self.archive.index["splits"]]
        raw_lengths = [int(value) for value in self.archive.index["num_tokens"].tolist()]
        self.video_indices: list[int] = []
        self.sampled_lengths: list[int] = []
        total_effective_tokens = 0
        for global_index, (video_split, raw_length) in enumerate(zip(splits, raw_lengths)):
            if video_split != self.split:
                continue
            sampled_length = math.ceil(raw_length / self.sample_rate)
            total_effective_tokens += sampled_length
            self.video_indices.append(global_index)
            self.sampled_lengths.append(sampled_length)
        if not self.video_indices:
            raise ValueError(f"No LEMON videos found for split={self.split!r}.")

        self.total_effective_tokens = int(total_effective_tokens)
        self.tokens_per_epoch = int(
            sum(min(length, self.max_sequence_len) for length in self.sampled_lengths)
        )
        self.num_cropped_videos = int(
            sum(length > self.max_sequence_len for length in self.sampled_lengths)
        )
        self.sequence_lengths = [
            min(length, self.max_sequence_len) for length in self.sampled_lengths
        ]
        self.effective_fps = self.archive.target_fps / self.sample_rate

    def __len__(self) -> int:
        return len(self.video_indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, item: int) -> dict[str, Any]:
        selected_position = int(item)
        sampled_length = self.sampled_lengths[selected_position]
        if sampled_length > self.max_sequence_len:
            rng = random.Random(
                self.seed + self.epoch * len(self.video_indices) + selected_position
            )
            sampled_start = rng.randint(0, sampled_length - self.max_sequence_len)
        else:
            sampled_start = 0
        global_index = self.video_indices[selected_position]
        result = self.archive.get_video_window(
            global_index,
            sample_rate=self.sample_rate,
            sampled_start=sampled_start,
            window_size=self.max_sequence_len,
        )

        valid_length = int(result["features"].shape[0])
        result["valid_mask"] = torch.ones(valid_length, dtype=torch.bool)
        result["valid_length"] = valid_length
        result["sample_name"] = f"{result['video_id']}@{sampled_start}"
        return result


def collate_lemon_videos(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad a video batch only to its longest sequence."""
    if not batch:
        raise ValueError("Cannot collate an empty LEMON batch.")
    batch_size = len(batch)
    max_length = max(int(item["valid_length"]) for item in batch)
    feature_dim = int(batch[0]["features"].shape[-1])
    features = torch.zeros(batch_size, max_length, feature_dim, dtype=torch.float32)
    times = torch.zeros(batch_size, max_length, dtype=torch.float32)
    valid_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    for batch_idx, item in enumerate(batch):
        length = int(item["valid_length"])
        features[batch_idx, :length] = item["features"]
        times[batch_idx, :length] = item["times"]
        if length < max_length and length > 0:
            times[batch_idx, length:] = item["times"][-1]
        valid_mask[batch_idx, :length] = True
    return {
        "features": features,
        "times": times,
        "valid_mask": valid_mask,
        "valid_length": torch.as_tensor(
            [int(item["valid_length"]) for item in batch], dtype=torch.long
        ),
        "video_id": [str(item["video_id"]) for item in batch],
        "path": [str(item["path"]) for item in batch],
        "start_token": torch.as_tensor(
            [int(item["start_token"]) for item in batch], dtype=torch.long
        ),
        "sample_name": [str(item["sample_name"]) for item in batch],
        "effective_fps": torch.as_tensor(
            [float(item["effective_fps"]) for item in batch], dtype=torch.float32
        ),
    }


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Shuffle videos while batching examples with similar temporal lengths."""

    def __init__(
        self,
        sequence_lengths: list[int],
        *,
        batch_size: int,
        seed: int,
        bucket_size_multiplier: int = 50,
        drop_last: bool = False,
    ):
        self.sequence_lengths = [int(value) for value in sequence_lengths]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.bucket_size = max(
            self.batch_size,
            self.batch_size * int(bucket_size_multiplier),
        )
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sequence_lengths) // self.batch_size
        return math.ceil(len(self.sequence_lengths) / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.sequence_lengths)))
        rng.shuffle(indices)
        batches: list[list[int]] = []
        for bucket_start in range(0, len(indices), self.bucket_size):
            bucket = indices[bucket_start : bucket_start + self.bucket_size]
            bucket.sort(key=self.sequence_lengths.__getitem__)
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start : batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        rng.shuffle(batches)
        yield from batches
