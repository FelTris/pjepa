from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Dict, List, Optional

import lmdb
import numpy as np
import torch

from pjepa.data.common import contiguous_window, shuffle_in_place


@dataclass(frozen=True)
class Assembly101LTContextSegmentRecord:
    segment_id: str
    start_frame: int
    end_frame: int
    label: int
    sample_rate: int


@dataclass(frozen=True)
class Assembly101LTContextTakeRecord:
    sample_id: str
    video_id: str
    view: str
    action_type: str
    video_end_frame: Optional[int]


class Assembly101LTContextLmdbTakeBatchGenerator(object):
    """
    LTContext-compatible Assembly101 TSM LMDB loader.

    Unlike the generic Assembly101 segment-manifest loader, this follows the
    original LTContext repo's sample definition: every row in
    train_C10119_rgb.csv / val_C10119_rgb.csv is a separate sequence, and its
    labels are read from coarse_labels/{action_type}_{video_id}.txt.
    """

    _ENV_CACHE: Dict[str, lmdb.Environment] = {}

    def __init__(
        self,
        num_classes,
        features_root,
        annotations_root,
        lmdb_subdir: str = "db_TSM_features",
        feature_dim: int = 2048,
        sample_rate: int = 1,
        max_take_len=None,
        action_id_column: str = "action_id",
        action_cls_column: str = "action_cls",
        video_id_column: str = "video_id",
        view_column: str = "view",
        action_type_column: str = "action_type",
        video_end_frame_column: str = "video_end_frame",
        cache_num_videos: int = 4,
        strict_missing_frames: bool = True,
    ):
        self.num_classes = None if num_classes in (None, "") else int(num_classes)
        self.features_root = str(features_root)
        self.annotations_root = Path(annotations_root)
        self.lmdb_subdir = str(lmdb_subdir)
        self.feature_dim = int(feature_dim)
        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        self.max_take_len = None if max_take_len in (None, "") else int(max_take_len)
        self.action_id_column = str(action_id_column)
        self.action_cls_column = str(action_cls_column)
        self.video_id_column = str(video_id_column)
        self.view_column = str(view_column)
        self.action_type_column = str(action_type_column)
        self.video_end_frame_column = str(video_end_frame_column)
        self.cache_num_videos = max(1, int(cache_num_videos))
        self.strict_missing_frames = bool(strict_missing_frames)

        self.list_of_examples: List[str] = []
        self.take_records: Dict[str, Assembly101LTContextTakeRecord] = {}
        self.take_to_segments: Dict[str, List[Assembly101LTContextSegmentRecord]] = {}
        self.index = 0
        self._envs: Dict[str, lmdb.Environment] = {}
        self._feature_cache: Dict[str, Dict[str, object]] = {}

    def reset(self):
        self.index = 0
        shuffle_in_place(self.list_of_examples)

    def has_next(self):
        return self.index < len(self.list_of_examples)

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
    def _frame_key(video_id: str, view: str, frame_idx: int) -> bytes:
        return f"{video_id}/{view}/{view}_{int(frame_idx):010d}.jpg".encode("utf-8")

    def _load_action_map(self) -> Dict[str, int]:
        actions_path = self.annotations_root / "actions.csv"
        if not actions_path.exists():
            raise FileNotFoundError(f"Assembly101 LTContext actions file not found: {actions_path}")

        action_to_id: Dict[str, int] = {}
        with actions_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {self.action_cls_column, self.action_id_column}
            missing = [column for column in required if column not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{actions_path} is missing required columns: {missing}")
            for row in reader:
                action_to_id[str(row[self.action_cls_column]).strip()] = int(
                    row[self.action_id_column]
                )
        return action_to_id

    def _load_segments(
        self,
        action_type: str,
        video_id: str,
        video_end_frame: Optional[int],
        action_to_id: Dict[str, int],
    ) -> List[Assembly101LTContextSegmentRecord]:
        label_path = self.annotations_root / "coarse_labels" / f"{action_type}_{video_id}.txt"
        if not label_path.exists():
            raise FileNotFoundError(
                f"Assembly101 LTContext coarse label file not found: {label_path}"
            )

        segments: List[Assembly101LTContextSegmentRecord] = []
        with label_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                start_frame = int(parts[0])
                end_frame = int(parts[1])
                label_name = str(parts[2]).strip()
                if video_end_frame is not None:
                    end_frame = min(end_frame, int(video_end_frame))
                if end_frame <= start_frame:
                    continue
                if label_name not in action_to_id:
                    raise KeyError(
                        f"Unknown Assembly101 action label {label_name!r} in {label_path}"
                    )
                label = int(action_to_id[label_name])
                if self.num_classes is not None and (label < 0 or label >= self.num_classes):
                    raise ValueError(
                        f"Label {label} out of range [0, {self.num_classes}) in {label_path}"
                    )
                segments.append(
                    Assembly101LTContextSegmentRecord(
                        segment_id=f"{action_type}_{video_id}_{idx:04d}",
                        start_frame=start_frame,
                        end_frame=end_frame,
                        label=label,
                        sample_rate=self.sample_rate,
                    )
                )
        return segments

    def read_data(self, manifest_path):
        manifest_path = Path(manifest_path)
        action_to_id = self._load_action_map()
        take_records: Dict[str, Assembly101LTContextTakeRecord] = {}
        take_to_segments: Dict[str, List[Assembly101LTContextSegmentRecord]] = {}
        skipped = 0

        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {self.video_id_column, self.view_column, self.action_type_column}
            missing = [column for column in required if column not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(
                    f"Assembly101 LTContext manifest {manifest_path} is missing columns: {missing}"
                )

            for row in reader:
                video_id = str(row[self.video_id_column]).strip()
                view = str(row[self.view_column]).strip()
                action_type = str(row[self.action_type_column]).strip()
                if not self._env_path(view).exists():
                    skipped += 1
                    continue
                raw_video_end = row.get(self.video_end_frame_column, "")
                video_end_frame = None if raw_video_end in (None, "") else int(float(raw_video_end))
                sample_id = f"{action_type}/{video_id}/{view}"
                segments = self._load_segments(action_type, video_id, video_end_frame, action_to_id)
                if not segments:
                    skipped += 1
                    continue
                take_records[sample_id] = Assembly101LTContextTakeRecord(
                    sample_id=sample_id,
                    video_id=video_id,
                    view=view,
                    action_type=action_type,
                    video_end_frame=video_end_frame,
                )
                take_to_segments[sample_id] = segments

        self.take_records = take_records
        self.take_to_segments = take_to_segments
        self.list_of_examples = list(take_to_segments.keys())
        shuffle_in_place(self.list_of_examples)
        self.index = 0
        print(
            f"Assembly101 LTContext LMDB loader: {len(self.list_of_examples)} samples "
            f"from {manifest_path}, skipped {skipped} rows, sample_rate={self.sample_rate}, "
            f"feature_dim={self.feature_dim}"
        )

    def _load_segment_features(
        self,
        video_id: str,
        view: str,
        segment: Assembly101LTContextSegmentRecord,
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
                frame_data = txn.get(self._frame_key(video_id, view, frame_idx))
                if frame_data is None:
                    if self.strict_missing_frames:
                        raise KeyError(
                            f"No Assembly101 LMDB feature for "
                            f"{video_id}/{view}/{view}_{int(frame_idx):010d}.jpg"
                        )
                    continue
                arr = np.frombuffer(frame_data, dtype=np.float32)
                if int(arr.shape[0]) != self.feature_dim:
                    raise ValueError(
                        f"Expected LMDB feature dim {self.feature_dim}, got {int(arr.shape[0])} "
                        f"for {video_id}/{view}/{view}_{int(frame_idx):010d}.jpg"
                    )
                features.append(arr)
        if not features:
            raise KeyError(
                f"No Assembly101 LMDB features loaded for segment "
                f"{video_id}/{view}:{segment.start_frame}-{segment.end_frame}"
            )
        return torch.from_numpy(np.stack(features, axis=0).copy()).to(torch.float32)

    def _load_take(self, sample_id: str):
        cached = self._feature_cache.get(sample_id) if self.max_take_len is None else None
        if cached is not None:
            return cached

        take = self.take_records[sample_id]
        segments = contiguous_window(self.take_to_segments[sample_id], self.max_take_len)
        take_feats = []
        take_targets = []
        segment_lengths = []
        segment_names = []

        for segment in segments:
            seg_feats = self._load_segment_features(take.video_id, take.view, segment)
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
            self._feature_cache[sample_id] = state
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
