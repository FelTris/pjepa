from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Dict, List, Optional

import lmdb
import numpy as np
import torch


from pjepa.data.assembly101_keys import take_key_from_row
from pjepa.data.common import contiguous_window, shuffle_in_place


@dataclass(frozen=True)
class Assembly101LmdbSegmentRecord:
    segment_id: str
    start_frame: int
    end_frame: int
    label: int
    sample_rate: int


class Assembly101LmdbTakeBatchGenerator(object):
    """
    LMDB-backed Assembly101 frame-feature loader.

    The Assembly101 TSM LMDB stores one float32 feature vector per frame under:
      {recording_id}/{view}/{view}_{frame_idx:010d}.jpg

    Batches follow the flat Assembly101 contract used by the feature SSL training loop:
      - inputs:          [B, T, D]
      - targets:         [B, T]
      - valid_mask:      [B, T]
      - segment_lengths: [B, S]
    """

    _ENV_CACHE: Dict[str, lmdb.Environment] = {}

    def __init__(
        self,
        num_classes,
        features_root,
        feature_source: str = "tsm_lmdb",
        lmdb_subdir: str = "db_TSM_features",
        feature_dim: int = 2048,
        sample_rate: int = 1,
        target_fps: Optional[float] = None,
        source_fps: float = 30.0,
        max_take_len=None,
        split_name: Optional[str] = None,
        label_column: str = "action_id",
        video_path_column: str = "video_path",
        split_column: str = "official_split",
        sample_id_column: str = "sample_uid",
        action_type_column: str = "action_type",
        take_key_mode: str = "action_type_video_view",
        start_frame_column: str = "start_frame",
        end_frame_column: str = "end_frame",
        source_fps_column: str = "annotation_fps",
        cache_num_videos: int = 4,
        strict_missing_frames: bool = True,
    ):
        del feature_source
        self.num_classes = None if num_classes in (None, "") else int(num_classes)
        self.features_root = str(features_root)
        self.lmdb_subdir = str(lmdb_subdir)
        self.feature_dim = int(feature_dim)
        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        self.target_fps = None if target_fps in (None, "") else float(target_fps)
        if self.target_fps is not None and self.target_fps <= 0:
            raise ValueError(f"target_fps must be positive or None, got {target_fps}")
        self.source_fps = float(source_fps)
        if self.source_fps <= 0:
            raise ValueError(f"source_fps must be positive, got {source_fps}")
        self.max_take_len = None if max_take_len in (None, "") else int(max_take_len)
        self.split_name = None if split_name in (None, "") else str(split_name)
        self.label_column = str(label_column)
        self.video_path_column = str(video_path_column)
        self.split_column = str(split_column)
        self.sample_id_column = str(sample_id_column)
        self.action_type_column = str(action_type_column)
        self.take_key_mode = str(take_key_mode)
        self.start_frame_column = str(start_frame_column)
        self.end_frame_column = str(end_frame_column)
        self.source_fps_column = str(source_fps_column)
        self.cache_num_videos = max(1, int(cache_num_videos))
        self.strict_missing_frames = bool(strict_missing_frames)

        self.list_of_examples: List[str] = []
        self.take_to_segments: Dict[str, List[Assembly101LmdbSegmentRecord]] = {}
        self.take_to_video_path: Dict[str, str] = {}
        self.index = 0
        self._envs: Dict[str, lmdb.Environment] = {}
        self._feature_cache: Dict[str, Dict[str, object]] = {}

        self.available_takes = 0
        self.missing_segments = 0

    def reset(self):
        self.index = 0
        shuffle_in_place(self.list_of_examples)

    def has_next(self):
        return self.index < len(self.list_of_examples)

    @staticmethod
    def _video_view_from_path(video_path: str):
        path = Path(str(video_path).strip())
        view = path.stem
        recording_id = str(path.parent)
        if not recording_id or recording_id == ".":
            raise ValueError(
                f"Assembly101 LMDB video_path must include recording/view: {video_path}"
            )
        return recording_id, view

    @staticmethod
    def _segment_id_from_row(
        row, sample_id_column: str, video_path: str, start_frame: int, end_frame: int
    ) -> str:
        sample_id = str(row.get(sample_id_column, "")).strip()
        if sample_id:
            return sample_id
        return f"{video_path}::frames{start_frame}_{end_frame}"

    def _sample_rate_for_row(self, row) -> int:
        if self.target_fps is None:
            return self.sample_rate
        raw_source_fps = row.get(self.source_fps_column, "")
        source_fps = self.source_fps
        if raw_source_fps not in (None, ""):
            source_fps = float(raw_source_fps)
        return max(1, int(round(source_fps / self.target_fps)))

    def _env_path(self, view: str) -> Path:
        return Path(self.features_root) / self.lmdb_subdir / view

    def _get_env(self, view: str):
        env = self._envs.get(view)
        if env is not None:
            return env
        env_path = self._env_path(view)
        if not env_path.exists():
            raise FileNotFoundError(f"Assembly101 LMDB view directory not found: {env_path}")
        env_key = str(env_path.resolve())
        env = self._ENV_CACHE.get(env_key)
        if env is None:
            env = lmdb.open(env_key, readonly=True, readahead=False, meminit=False, lock=False)
            self._ENV_CACHE[env_key] = env
        self._envs[view] = env
        return env

    @staticmethod
    def _frame_key(recording_id: str, view: str, frame_idx: int) -> bytes:
        return f"{recording_id}/{view}/{view}_{int(frame_idx):010d}.jpg".encode("utf-8")

    def read_data(self, manifest_path):
        take_to_segments: Dict[str, List[Assembly101LmdbSegmentRecord]] = {}
        take_to_video_path: Dict[str, str] = {}
        missing_segments = 0

        with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required_columns = {
                self.video_path_column,
                self.label_column,
                self.start_frame_column,
                self.end_frame_column,
            }
            missing_columns = [
                column for column in required_columns if column not in (reader.fieldnames or [])
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

                video_path = str(row[self.video_path_column]).strip()
                _, view = self._video_view_from_path(video_path)
                if not self._env_path(view).exists():
                    missing_segments += 1
                    continue
                take_key = take_key_from_row(
                    row,
                    video_path_column=self.video_path_column,
                    action_type_column=self.action_type_column,
                    sample_id_column=self.sample_id_column,
                    take_key_mode=self.take_key_mode,
                )
                previous_video_path = take_to_video_path.setdefault(take_key, video_path)
                if previous_video_path != video_path:
                    raise ValueError(
                        f"Assembly101 take key {take_key!r} maps to multiple video paths: "
                        f"{previous_video_path!r} and {video_path!r}"
                    )

                label = int(row[self.label_column])
                if self.num_classes is not None and (label < 0 or label >= self.num_classes):
                    raise ValueError(
                        f"Label {label} out of range [0, {self.num_classes}) for {video_path}"
                    )

                start_frame = int(float(row[self.start_frame_column]))
                end_frame = int(float(row[self.end_frame_column]))
                if end_frame <= start_frame:
                    missing_segments += 1
                    continue

                segment_id = self._segment_id_from_row(
                    row,
                    self.sample_id_column,
                    video_path,
                    start_frame,
                    end_frame,
                )
                take_to_segments.setdefault(take_key, []).append(
                    Assembly101LmdbSegmentRecord(
                        segment_id=segment_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        label=label,
                        sample_rate=self._sample_rate_for_row(row),
                    )
                )

        for take_key, segments in take_to_segments.items():
            segments.sort(key=lambda item: (item.start_frame, item.end_frame, item.segment_id))
            take_to_segments[take_key] = segments

        self.take_to_segments = take_to_segments
        self.take_to_video_path = take_to_video_path
        self.list_of_examples = list(take_to_segments.keys())
        shuffle_in_place(self.list_of_examples)
        self.index = 0
        self.available_takes = len(self.list_of_examples)
        self.missing_segments = missing_segments

        fps_msg = (
            f"target_fps={self.target_fps}"
            if self.target_fps is not None
            else f"sample_rate={self.sample_rate}"
        )
        print(
            f"Assembly101 LMDB loader split={self.split_name or 'all'}: "
            f"{self.available_takes} takes, skipped {self.missing_segments} rows with missing features. "
            f"{fps_msg}, feature_dim={self.feature_dim}, take_key_mode={self.take_key_mode}"
        )

    def _load_segment_features(
        self,
        recording_id: str,
        view: str,
        segment: Assembly101LmdbSegmentRecord,
    ) -> torch.Tensor:
        env = self._get_env(view)
        frame_indices = range(
            int(segment.start_frame),
            int(segment.end_frame),
            max(1, int(segment.sample_rate)),
        )
        features = []

        with env.begin(write=False) as txn:
            for frame_idx in frame_indices:
                frame_data = txn.get(self._frame_key(recording_id, view, frame_idx))
                if frame_data is None:
                    if self.strict_missing_frames:
                        raise KeyError(
                            f"No Assembly101 LMDB feature for "
                            f"{recording_id}/{view}/{view}_{int(frame_idx):010d}.jpg"
                        )
                    continue
                arr = np.frombuffer(frame_data, dtype=np.float32)
                if int(arr.shape[0]) != self.feature_dim:
                    raise ValueError(
                        f"Expected LMDB feature dim {self.feature_dim}, got {int(arr.shape[0])} "
                        f"for {recording_id}/{view}/{view}_{int(frame_idx):010d}.jpg"
                    )
                features.append(arr)

        if not features:
            fallback_idx = int(segment.start_frame)
            with env.begin(write=False) as txn:
                frame_data = txn.get(self._frame_key(recording_id, view, fallback_idx))
            if frame_data is None:
                raise KeyError(
                    f"No Assembly101 LMDB feature for fallback frame "
                    f"{recording_id}/{view}/{view}_{fallback_idx:010d}.jpg"
                )
            features.append(np.frombuffer(frame_data, dtype=np.float32))

        return torch.from_numpy(np.stack(features, axis=0).copy()).to(torch.float32)

    def _load_take(self, take_key: str):
        cached = self._feature_cache.get(take_key) if self.max_take_len is None else None
        if cached is not None:
            return cached

        recording_id, view = self._video_view_from_path(self.take_to_video_path[take_key])
        segments = contiguous_window(self.take_to_segments[take_key], self.max_take_len)

        take_feats = []
        take_targets = []
        segment_lengths = []
        segment_names = []

        for segment in segments:
            seg_feats = self._load_segment_features(recording_id, view, segment)
            seg_len = int(seg_feats.shape[0])
            if seg_len <= 0:
                continue
            take_feats.append(seg_feats)
            take_targets.append(torch.full((seg_len,), int(segment.label), dtype=torch.long))
            segment_lengths.append(seg_len)
            segment_names.append(segment.segment_id)

        if take_feats:
            take_input = torch.cat(take_feats, dim=0)
            take_target = torch.cat(take_targets, dim=0)
        else:
            take_input = torch.zeros((0, self.feature_dim), dtype=torch.float32)
            take_target = torch.zeros((0,), dtype=torch.long)

        state = {
            "input": take_input,
            "target": take_target,
            "names": segment_names,
            "segment_lengths": torch.as_tensor(segment_lengths, dtype=torch.long),
        }
        if self.max_take_len is None:
            self._feature_cache[take_key] = state
            while len(self._feature_cache) > self.cache_num_videos:
                first_key = next(iter(self._feature_cache))
                del self._feature_cache[first_key]
        return state

    def next_batch(self, batch_size):
        batch_takes = self.list_of_examples[self.index : self.index + batch_size]
        self.index += batch_size

        loaded = [self._load_take(take) for take in batch_takes]
        batch_size_actual = len(loaded)
        max_tokens = max((int(item["input"].shape[0]) for item in loaded), default=0)
        max_segments = max((int(item["segment_lengths"].shape[0]) for item in loaded), default=0)

        batch_input = torch.zeros(
            batch_size_actual, max_tokens, self.feature_dim, dtype=torch.float32
        )
        batch_target = torch.ones(batch_size_actual, max_tokens, dtype=torch.long) * (-100)
        valid_mask = torch.zeros(batch_size_actual, max_tokens, dtype=torch.bool)
        segment_lengths = torch.zeros(batch_size_actual, max_segments, dtype=torch.long)
        batch_names = []

        for batch_idx, item in enumerate(loaded):
            feat_arr = item["input"]
            target_arr = item["target"]
            seg_lens = item["segment_lengths"]
            token_count = int(feat_arr.shape[0])
            if token_count > 0:
                batch_input[batch_idx, :token_count] = feat_arr
                batch_target[batch_idx, :token_count] = target_arr
                valid_mask[batch_idx, :token_count] = True
            if int(seg_lens.shape[0]) > 0:
                segment_lengths[batch_idx, : int(seg_lens.shape[0])] = seg_lens
            batch_names.append(item["names"])

        return batch_input, batch_target, valid_mask, batch_names, segment_lengths
