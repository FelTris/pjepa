from collections import OrderedDict
from dataclasses import dataclass
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


from pjepa.data.common import contiguous_window, resolve_feature_path, shuffle_in_place


@dataclass(frozen=True)
class Assembly101SegmentRecord:
    segment_id: str
    start_sec: float
    end_sec: float
    label: int


class Assembly101FrameTakeBatchGenerator(object):
    """
    Manifest-backed flat-sequence loader for Assembly101 frame-wise feature tensors.

    Each example corresponds to one video-level feature file. The loader slices that
    tensor into annotation-aligned segments using feature_times_sec, then flattens the
    selected segments into one contiguous token stream:
      - inputs:          [B, T, D]
      - targets:         [B, T]
      - valid_mask:      [B, T]
      - segment_lengths: [B, S]
    """

    def __init__(
        self,
        num_classes,
        features_root,
        sample_rate=1,
        max_take_len=None,
        split_name: Optional[str] = None,
        label_column: str = "action_id",
        video_path_column: str = "video_path",
        split_column: str = "official_split",
        sample_id_column: str = "sample_uid",
        start_sec_column: str = "start_sec",
        end_sec_column: str = "end_sec",
        cache_num_videos: int = 8,
    ):
        self.num_classes = None if num_classes in (None, "") else int(num_classes)
        self.features_root = str(features_root)
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        self.max_take_len = None if max_take_len in (None, "") else int(max_take_len)
        self.split_name = None if split_name in (None, "") else str(split_name)
        self.label_column = str(label_column)
        self.video_path_column = str(video_path_column)
        self.split_column = str(split_column)
        self.sample_id_column = str(sample_id_column)
        self.start_sec_column = str(start_sec_column)
        self.end_sec_column = str(end_sec_column)
        self.cache_num_videos = max(1, int(cache_num_videos))

        self.list_of_examples: List[str] = []
        self.take_to_segments: Dict[str, List[Assembly101SegmentRecord]] = {}
        self.index = 0
        self._feature_cache: OrderedDict[str, Dict[str, object]] = OrderedDict()

        self.available_takes = 0
        self.missing_takes = 0
        self.missing_segments = 0

    def reset(self):
        self.index = 0
        shuffle_in_place(self.list_of_examples)

    def has_next(self):
        return self.index < len(self.list_of_examples)

    @staticmethod
    def _feature_rel_path(video_path: str) -> str:
        path = str(video_path).strip()
        suffix = Path(path).suffix.lower()
        if suffix in {".mp4", ".avi"}:
            return str(Path(path).with_suffix(".pt"))
        return path

    def _segment_id_from_row(
        self, row, feature_rel_path: str, start_sec: float, end_sec: float
    ) -> str:
        sample_id = str(row.get(self.sample_id_column, "")).strip()
        if sample_id:
            return sample_id
        return f"{feature_rel_path}::start{start_sec:.3f}_end{end_sec:.3f}"

    def read_data(self, manifest_path):
        take_to_segments: Dict[str, List[Assembly101SegmentRecord]] = {}
        missing_takes = 0
        missing_segments = 0

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
                feature_full_path = resolve_feature_path(feature_rel_path, self.features_root)
                if not os.path.exists(feature_full_path):
                    missing_segments += 1
                    continue

                label = int(row[self.label_column])
                if self.num_classes is not None and (label < 0 or label >= self.num_classes):
                    raise ValueError(
                        f"Label {label} out of range [0, {self.num_classes}) for {feature_rel_path}"
                    )

                start_sec = float(row[self.start_sec_column])
                end_sec = float(row[self.end_sec_column])
                segment_id = self._segment_id_from_row(row, feature_rel_path, start_sec, end_sec)
                take_to_segments.setdefault(feature_rel_path, []).append(
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
        self.list_of_examples = list(take_to_segments.keys())
        shuffle_in_place(self.list_of_examples)
        self.index = 0

        self.available_takes = len(self.list_of_examples)
        self.missing_takes = missing_takes
        self.missing_segments = missing_segments

        print(
            f"Assembly101 loader split={self.split_name or 'all'}: "
            f"{self.available_takes} takes, skipped {self.missing_segments} rows with missing features."
        )

    def _load_feature_payload(self, feature_rel_path: str):
        cached = self._feature_cache.get(feature_rel_path)
        if cached is not None:
            self._feature_cache.move_to_end(feature_rel_path)
            return cached

        feature_path = resolve_feature_path(feature_rel_path, self.features_root)
        payload = torch.load(feature_path, map_location="cpu")
        if isinstance(payload, dict):
            if "features" not in payload or "feature_times_sec" not in payload:
                raise ValueError(
                    f"Expected dict payload with 'features' and 'feature_times_sec' in {feature_path}"
                )
            features = payload["features"]
            times = payload["feature_times_sec"]
        else:
            raise ValueError(
                f"Expected dict payload for Assembly101 frame features at {feature_path}, got {type(payload)!r}"
            )

        if not torch.is_tensor(features) or features.ndim != 2:
            raise ValueError(
                f"Assembly101 features at {feature_path} must have shape [T, D], got {tuple(features.shape)}"
            )
        if not torch.is_tensor(times) or times.ndim != 1:
            raise ValueError(
                f"Assembly101 feature times at {feature_path} must have shape [T], got {tuple(times.shape)}"
            )
        if int(features.shape[0]) != int(times.shape[0]):
            raise ValueError(
                f"Feature/time length mismatch at {feature_path}: {tuple(features.shape)} vs {tuple(times.shape)}"
            )

        state = {
            "features": features.to(dtype=torch.float32, device="cpu").contiguous(),
            "times": times.to(dtype=torch.float32, device="cpu").contiguous(),
            "times_np": times.to(dtype=torch.float32, device="cpu").numpy(),
        }
        self._feature_cache[feature_rel_path] = state
        self._feature_cache.move_to_end(feature_rel_path)
        while len(self._feature_cache) > self.cache_num_videos:
            self._feature_cache.popitem(last=False)
        return state

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

    def _slice_segment(self, feature_state, segment: Assembly101SegmentRecord) -> torch.Tensor:
        features = feature_state["features"]
        times_np = feature_state["times_np"]
        if times_np.size == 0:
            raise ValueError("Encountered empty feature timeline.")

        start_sec = float(segment.start_sec)
        end_sec = float(segment.end_sec)
        left = int(np.searchsorted(times_np, start_sec, side="left"))
        right = int(np.searchsorted(times_np, end_sec, side="left"))

        if right <= left:
            center_sec = start_sec if end_sec <= start_sec else 0.5 * (start_sec + end_sec)
            nearest_idx = self._nearest_index(times_np, center_sec)
            if nearest_idx < 0:
                raise ValueError("Failed to find fallback token for empty segment slice.")
            segment_feats = features[nearest_idx : nearest_idx + 1]
        else:
            segment_feats = features[left:right]

        if self.sample_rate > 1:
            segment_feats = segment_feats[:: self.sample_rate]

        if int(segment_feats.shape[0]) <= 0:
            nearest_idx = self._nearest_index(times_np, start_sec)
            segment_feats = features[nearest_idx : nearest_idx + 1]

        return segment_feats

    def _load_take(self, take_key: str):
        segments = contiguous_window(self.take_to_segments[take_key], self.max_take_len)
        feature_state = self._load_feature_payload(take_key)

        take_feats = []
        take_targets = []
        segment_lengths = []
        segment_names = []

        for segment in segments:
            seg_feats = self._slice_segment(feature_state, segment)
            seg_len = int(seg_feats.shape[0])
            if seg_len <= 0:
                continue

            take_feats.append(seg_feats)
            take_targets.append(torch.full((seg_len,), int(segment.label), dtype=torch.long))
            segment_lengths.append(seg_len)
            segment_names.append(segment.segment_id)

        feature_dim = int(feature_state["features"].shape[1])
        if take_feats:
            take_input = torch.cat(take_feats, dim=0)
            take_target = torch.cat(take_targets, dim=0)
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
            take_input, take_target, segment_names, seg_lens, this_feature_dim = self._load_take(
                take
            )
            batch_inputs.append(take_input)
            batch_targets.append(take_target)
            batch_names.append(segment_names)
            batch_seg_lens.append(seg_lens)
            if feature_dim is None and this_feature_dim:
                feature_dim = int(this_feature_dim)

        batch_size_actual = len(batch_inputs)
        max_tokens = max((int(arr.shape[0]) for arr in batch_inputs), default=0)
        max_segments = max((int(seg_lens.shape[0]) for seg_lens in batch_seg_lens), default=0)
        feature_dim = int(feature_dim or 0)

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


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Assembly101 manifest CSV.")
    ap.add_argument(
        "--features-root", required=True, help="Root containing per-video .pt feature files."
    )
    ap.add_argument("--split", default="train", help="Official split to keep.")
    ap.add_argument(
        "--label-column", default="action_id", help="Manifest column containing class ids."
    )
    ap.add_argument(
        "--num-classes", type=int, default=None, help="Optional total number of classes."
    )
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--sample-rate", type=int, default=1)
    ap.add_argument("--max-take-len", type=int, default=None)
    args = ap.parse_args()

    gen = Assembly101FrameTakeBatchGenerator(
        num_classes=args.num_classes,
        features_root=args.features_root,
        sample_rate=args.sample_rate,
        max_take_len=args.max_take_len,
        split_name=args.split,
        label_column=args.label_column,
    )
    gen.read_data(args.manifest)
    gen.reset()

    if gen.has_next():
        batch_input, batch_target, valid_mask, names, segment_lengths = gen.next_batch(
            args.batch_size
        )
        print(f"inputs:          {tuple(batch_input.shape)} dtype={batch_input.dtype}")
        print(f"targets:         {tuple(batch_target.shape)} dtype={batch_target.dtype}")
        print(f"valid_mask:      {tuple(valid_mask.shape)} dtype={valid_mask.dtype}")
        print(f"segment_lengths: {tuple(segment_lengths.shape)} dtype={segment_lengths.dtype}")
        print(f"names:           {len(names)} takes")
