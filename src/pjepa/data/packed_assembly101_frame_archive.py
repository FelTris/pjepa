from dataclasses import dataclass
import csv
from typing import Dict, List, Optional

import numpy as np
import torch


from pjepa.data.assembly101_keys import feature_rel_path_from_video_path, take_key_from_row
from pjepa.data.common import shuffle_in_place


@dataclass(frozen=True)
class Assembly101SegmentRecord:
    segment_id: str
    start_sec: float
    end_sec: float
    label: int


class PackedAssembly101FrameArchive(object):
    _RAM_CACHE = {}

    def __init__(self, archive_path: str, cache_in_memory: bool = True):
        self.archive_path = str(archive_path)
        self.cache_in_memory = bool(cache_in_memory)
        self._local_state = None

    def _load_archive(self):
        payload = torch.load(self.archive_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Packed archive at {self.archive_path} must be a dict payload.")

        required_keys = {"paths", "tokens", "times", "offsets"}
        missing = [key for key in required_keys if key not in payload]
        if missing:
            raise ValueError(
                f"Packed archive at {self.archive_path} is missing required keys: {missing}"
            )

        paths = [str(path) for path in payload["paths"]]
        tokens = payload["tokens"]
        times = payload["times"]
        offsets = torch.as_tensor(payload["offsets"], dtype=torch.long, device="cpu").contiguous()

        if not torch.is_tensor(tokens) or tokens.ndim != 2:
            raise ValueError(
                f"Archive tokens must have shape [total_tokens, D], got {type(tokens)!r}"
            )
        if not torch.is_tensor(times) or times.ndim != 1:
            raise ValueError(f"Archive times must have shape [total_tokens], got {type(times)!r}")
        if int(tokens.shape[0]) != int(times.shape[0]):
            raise ValueError(
                f"Archive tokens/times length mismatch: {tuple(tokens.shape)} vs {tuple(times.shape)}"
            )
        if offsets.ndim != 1 or int(offsets.numel()) != len(paths) + 1:
            raise ValueError(
                f"Archive offsets must have shape [N+1], got {tuple(offsets.shape)} for {len(paths)} paths"
            )
        if int(offsets[0].item()) != 0:
            raise ValueError("Archive offsets must start at 0.")
        if bool((offsets[1:] < offsets[:-1]).any().item()):
            raise ValueError("Archive offsets must be non-decreasing.")
        if int(offsets[-1].item()) != int(tokens.shape[0]):
            raise ValueError(
                f"Final archive offset ({int(offsets[-1].item())}) must match total token count ({int(tokens.shape[0])})."
            )

        return {
            "tokens": tokens.to(dtype=torch.float32, device="cpu").contiguous(),
            "times": times.to(dtype=torch.float32, device="cpu").contiguous(),
            "offsets": offsets,
            "paths": paths,
            "path_to_idx": {path: idx for idx, path in enumerate(paths)},
            "feature_dim": int(tokens.shape[1]),
        }

    def preload(self):
        if self.cache_in_memory:
            if self.archive_path not in self._RAM_CACHE:
                self._RAM_CACHE[self.archive_path] = self._load_archive()
            return self._RAM_CACHE[self.archive_path]
        if self._local_state is None:
            self._local_state = self._load_archive()
        return self._local_state

    def feature_dim(self) -> int:
        return int(self.preload()["feature_dim"])

    def has_path(self, rel_path: str) -> bool:
        return rel_path in self.preload()["path_to_idx"]

    def fetch(self, rel_path: str):
        state = self.preload()
        try:
            idx = int(state["path_to_idx"][rel_path])
        except KeyError as exc:
            raise KeyError(f"Archive {self.archive_path} is missing path: {rel_path}") from exc

        start = int(state["offsets"][idx].item())
        end = int(state["offsets"][idx + 1].item())
        return {
            "features": state["tokens"][start:end],
            "times": state["times"][start:end],
        }


class PackedAssembly101FrameTakeBatchGenerator(object):
    """
    Archive-backed Assembly101 frame-feature take loader.
    """

    def __init__(
        self,
        num_classes,
        archive_path: str,
        sample_rate: int = 1,
        max_take_len=None,
        split_name: Optional[str] = None,
        label_column: str = "action_id",
        video_path_column: str = "video_path",
        split_column: str = "official_split",
        sample_id_column: str = "sample_uid",
        action_type_column: str = "action_type",
        take_key_mode: str = "action_type_video_view",
        start_sec_column: str = "start_sec",
        end_sec_column: str = "end_sec",
        cache_in_memory: bool = True,
        preload_in_parent: bool = True,
        max_background_segment_tokens: Optional[int] = None,
        background_label_id: int = 0,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        prefetch_factor: Optional[int] = 2,
        multiprocessing_context: Optional[str] = None,
    ):
        del num_workers, persistent_workers, pin_memory, prefetch_factor, multiprocessing_context
        self.num_classes = None if num_classes in (None, "") else int(num_classes)
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        self.max_take_len = None if max_take_len in (None, "") else int(max_take_len)
        self.split_name = None if split_name in (None, "") else str(split_name)
        self.label_column = str(label_column)
        self.video_path_column = str(video_path_column)
        self.split_column = str(split_column)
        self.sample_id_column = str(sample_id_column)
        self.action_type_column = str(action_type_column)
        self.take_key_mode = str(take_key_mode)
        self.start_sec_column = str(start_sec_column)
        self.end_sec_column = str(end_sec_column)
        self.preload_in_parent = bool(preload_in_parent)
        self.max_background_segment_tokens = (
            None
            if max_background_segment_tokens in (None, "")
            else int(max_background_segment_tokens)
        )
        if (
            self.max_background_segment_tokens is not None
            and self.max_background_segment_tokens <= 0
        ):
            raise ValueError(
                f"max_background_segment_tokens must be positive or None, got {max_background_segment_tokens}"
            )
        self.background_label_id = int(background_label_id)

        self.archive = PackedAssembly101FrameArchive(
            archive_path=archive_path,
            cache_in_memory=cache_in_memory,
        )
        self.list_of_examples: List[str] = []
        self.take_to_segments: Dict[str, List[Assembly101SegmentRecord]] = {}
        self.take_to_feature_path: Dict[str, str] = {}
        self.index = 0
        self.missing_segments = 0

        if self.preload_in_parent:
            self.archive.preload()

    def reset(self):
        self.index = 0
        shuffle_in_place(self.list_of_examples)

    def has_next(self):
        return self.index < len(self.list_of_examples)

    @staticmethod
    def _feature_rel_path(video_path: str) -> str:
        return feature_rel_path_from_video_path(video_path)

    def _segment_id_from_row(
        self, row, feature_rel_path: str, start_sec: float, end_sec: float
    ) -> str:
        sample_id = str(row.get(self.sample_id_column, "")).strip()
        if sample_id:
            return sample_id
        return f"{feature_rel_path}::start{start_sec:.3f}_end{end_sec:.3f}"

    def read_data(self, manifest_path):
        take_to_segments: Dict[str, List[Assembly101SegmentRecord]] = {}
        take_to_feature_path: Dict[str, str] = {}
        missing_segments = 0

        if self.preload_in_parent:
            self.archive.preload()

        with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required_columns = {
                self.video_path_column,
                self.label_column,
                self.start_sec_column,
                self.end_sec_column,
            }
            missing_columns = [
                column for column in required_columns if column not in reader.fieldnames
            ]
            if missing_columns:
                raise ValueError(
                    f"Assembly101 manifest {manifest_path} is missing required columns: {missing_columns}"
                )

            for row in reader:
                if self.split_name is not None:
                    if (
                        str(row.get(self.split_column, "")).strip().lower()
                        != self.split_name.lower()
                    ):
                        continue

                feature_rel_path = self._feature_rel_path(row[self.video_path_column])
                if not self.archive.has_path(feature_rel_path):
                    missing_segments += 1
                    continue
                take_key = take_key_from_row(
                    row,
                    video_path_column=self.video_path_column,
                    action_type_column=self.action_type_column,
                    sample_id_column=self.sample_id_column,
                    take_key_mode=self.take_key_mode,
                )
                previous_feature_path = take_to_feature_path.setdefault(take_key, feature_rel_path)
                if previous_feature_path != feature_rel_path:
                    raise ValueError(
                        f"Assembly101 take key {take_key!r} maps to multiple feature paths: "
                        f"{previous_feature_path!r} and {feature_rel_path!r}"
                    )

                label = int(row[self.label_column])
                if self.num_classes is not None and (label < 0 or label >= self.num_classes):
                    raise ValueError(
                        f"Label {label} out of range [0, {self.num_classes}) for {feature_rel_path}"
                    )

                start_sec = float(row[self.start_sec_column])
                end_sec = float(row[self.end_sec_column])
                segment_id = self._segment_id_from_row(row, feature_rel_path, start_sec, end_sec)
                take_to_segments.setdefault(take_key, []).append(
                    Assembly101SegmentRecord(
                        segment_id=segment_id,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        label=label,
                    )
                )

        for take_key, segments in take_to_segments.items():
            segments.sort(key=lambda item: (item.start_sec, item.end_sec, item.segment_id))
            take_to_segments[take_key] = segments

        self.take_to_segments = take_to_segments
        self.take_to_feature_path = take_to_feature_path
        self.list_of_examples = list(take_to_segments.keys())
        shuffle_in_place(self.list_of_examples)
        self.index = 0
        self.missing_segments = missing_segments

        print(
            f"Packed Assembly101 loader split={self.split_name or 'all'}: "
            f"{len(self.list_of_examples)} takes, skipped {self.missing_segments} rows missing from archive. "
            f"take_key_mode={self.take_key_mode}, "
            f"max_background_segment_tokens={self.max_background_segment_tokens}"
        )

    @staticmethod
    def _nearest_index(times_np: np.ndarray, query_time: float) -> int:
        if times_np.size == 0:
            return -1
        insert_idx = int(np.searchsorted(times_np, query_time, side="left"))
        if insert_idx <= 0:
            return 0
        if insert_idx >= times_np.size:
            return int(times_np.size - 1)
        prev_idx = insert_idx - 1
        if abs(times_np[insert_idx] - query_time) < abs(times_np[prev_idx] - query_time):
            return insert_idx
        return prev_idx

    def _sample_segments(self, take: str):
        segments = self.take_to_segments[take]
        if not self.max_take_len or len(segments) <= self.max_take_len:
            return segments
        max_start = len(segments) - self.max_take_len
        start_idx = np.random.randint(0, max_start + 1)
        return segments[start_idx : start_idx + self.max_take_len]

    def _slice_segment(
        self, features: torch.Tensor, times_np: np.ndarray, segment: Assembly101SegmentRecord
    ) -> torch.Tensor:
        left = int(np.searchsorted(times_np, float(segment.start_sec), side="left"))
        right = int(np.searchsorted(times_np, float(segment.end_sec), side="left"))

        if right <= left:
            center_sec = (
                float(segment.start_sec)
                if float(segment.end_sec) <= float(segment.start_sec)
                else 0.5 * (float(segment.start_sec) + float(segment.end_sec))
            )
            nearest_idx = self._nearest_index(times_np, center_sec)
            segment_feats = features[nearest_idx : nearest_idx + 1]
        else:
            segment_feats = features[left:right]

        if self.sample_rate > 1:
            segment_feats = segment_feats[:: self.sample_rate]

        if int(segment_feats.shape[0]) <= 0:
            nearest_idx = self._nearest_index(times_np, float(segment.start_sec))
            segment_feats = features[nearest_idx : nearest_idx + 1]
        if (
            self.max_background_segment_tokens is not None
            and int(segment.label) == self.background_label_id
            and int(segment_feats.shape[0]) > self.max_background_segment_tokens
        ):
            segment_feats = segment_feats[: self.max_background_segment_tokens]
        return segment_feats

    def _load_take(self, take: str):
        segments = self._sample_segments(take)
        payload = self.archive.fetch(self.take_to_feature_path[take])
        features = payload["features"].to(dtype=torch.float32, device="cpu")
        times_np = payload["times"].to(dtype=torch.float32, device="cpu").numpy()

        take_feats = []
        take_targets = []
        segment_lengths = []
        segment_names = []

        for segment in segments:
            seg_feats = self._slice_segment(features, times_np, segment)
            seg_len = int(seg_feats.shape[0])
            if seg_len <= 0:
                continue
            take_feats.append(seg_feats)
            take_targets.append(torch.full((seg_len,), int(segment.label), dtype=torch.long))
            segment_lengths.append(seg_len)
            segment_names.append(segment.segment_id)

        feature_dim = self.archive.feature_dim()
        if take_feats:
            take_input = torch.cat(take_feats, dim=0)
            take_target = torch.cat(take_targets, dim=0)
            feature_dim = int(take_input.shape[1])
        else:
            take_input = torch.zeros((0, feature_dim), dtype=torch.float32)
            take_target = torch.zeros((0,), dtype=torch.long)

        return (
            take_input,
            take_target,
            segment_names,
            torch.as_tensor(segment_lengths, dtype=torch.long),
            feature_dim,
        )

    def next_batch(self, batch_size):
        batch_takes = self.list_of_examples[self.index : self.index + batch_size]
        self.index += batch_size

        batch_inputs = []
        batch_targets = []
        batch_names = []
        batch_seg_lens = []
        feature_dim = None

        for take in batch_takes:
            take_input, take_target, names, seg_lens, this_feature_dim = self._load_take(take)
            batch_inputs.append(take_input)
            batch_targets.append(take_target)
            batch_names.append(names)
            batch_seg_lens.append(seg_lens)
            if feature_dim is None and this_feature_dim:
                feature_dim = int(this_feature_dim)

        batch_size_actual = len(batch_inputs)
        max_tokens = max((int(arr.shape[0]) for arr in batch_inputs), default=0)
        max_segments = max((int(seg_lens.shape[0]) for seg_lens in batch_seg_lens), default=0)
        feature_dim = int(feature_dim or self.archive.feature_dim())

        batch_input = torch.zeros(batch_size_actual, max_tokens, feature_dim, dtype=torch.float32)
        batch_target = torch.ones(batch_size_actual, max_tokens, dtype=torch.long) * (-100)
        valid_mask = torch.zeros(batch_size_actual, max_tokens, dtype=torch.bool)
        segment_lengths = torch.zeros(batch_size_actual, max_segments, dtype=torch.long)

        for batch_idx, (feat_arr, target_arr, seg_lens) in enumerate(
            zip(batch_inputs, batch_targets, batch_seg_lens)
        ):
            token_count = int(feat_arr.shape[0])
            if token_count > 0:
                batch_input[batch_idx, :token_count] = feat_arr
                batch_target[batch_idx, :token_count] = target_arr
                valid_mask[batch_idx, :token_count] = True
            if int(seg_lens.shape[0]) > 0:
                segment_lengths[batch_idx, : int(seg_lens.shape[0])] = seg_lens

        return batch_input, batch_target, valid_mask, batch_names, segment_lengths
