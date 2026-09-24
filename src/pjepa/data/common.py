import csv
import os
import random
import re

import numpy as np
import torch


START_TIME_RE = re.compile(r"start([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
START_END_RE = re.compile(r"start([0-9]+(?:\.[0-9]+)?)_end([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def shuffle_in_place(items):
    random.shuffle(items)


def take_name_from_path(path):
    return os.path.basename(os.path.dirname(path))


def start_time_from_path(path):
    match = START_TIME_RE.search(os.path.basename(path))
    return float(match.group(1)) if match else -1.0


def sort_take_segments(take_to_segments):
    for take, segments in take_to_segments.items():
        segments.sort(key=lambda item: (start_time_from_path(item[0]), item[0]))
        take_to_segments[take] = segments
    return take_to_segments


def resolve_feature_path(rel_or_abs, features_root="", tolerate_missing_absolute=False):
    if os.path.isabs(rel_or_abs):
        if tolerate_missing_absolute and not os.path.exists(rel_or_abs):
            rel_or_abs = rel_or_abs.lstrip(os.sep)
        else:
            return rel_or_abs
    return os.path.join(features_root, rel_or_abs) if features_root else rel_or_abs


def load_tensor(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("features", "feats", "x"):
            if key in obj and torch.is_tensor(obj[key]):
                return obj[key]
        tensor = next((value for value in obj.values() if torch.is_tensor(value)), None)
        if tensor is None:
            raise ValueError(f"No tensor-like value found in dict at {path}")
        return tensor
    return obj


def flatten_tensor(tensor, sample_rate=1, transpose=False):
    if tensor.dim() < 2:
        raise ValueError(f"Expected >=2D tensor but got shape {tuple(tensor.shape)}")

    feature_dim = tensor.shape[-1]
    time_steps = int(np.prod(tensor.shape[:-1]))
    feats = tensor.reshape(time_steps, feature_dim).to(torch.float32).numpy()
    if sample_rate > 1:
        feats = feats[::sample_rate]
    return feats.T if transpose else feats


def load_feature_array(path, sample_rate=1, transpose=False):
    return flatten_tensor(load_tensor(path), sample_rate=sample_rate, transpose=transpose)


def detect_delimiter(sample: str) -> str:
    if "\t" in sample:
        return "\t"
    return ","


def normalize_label_path(rel_path: str) -> str:
    rel_path = rel_path.strip()
    if rel_path.endswith(".mp4"):
        return rel_path[:-4] + ".pt"
    return rel_path


def iter_label_rows(csv_file):
    with open(csv_file, "r", encoding="utf-8") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        reader = csv.reader(handle, delimiter=detect_delimiter(sample))
        for row in reader:
            if not row:
                continue
            rel_path = normalize_label_path(row[0])
            if not rel_path or rel_path.startswith("#"):
                continue
            if len(row) < 2:
                raise ValueError(f"CSV row missing label: {row}")
            yield rel_path, row[1].strip()


def parse_take_label_tsv(csv_file):
    take_to_segments = {}
    for rel_path, raw_label in iter_label_rows(csv_file):
        try:
            label = int(raw_label)
        except ValueError as exc:
            raise ValueError(
                f"Label must be an integer, got '{raw_label}' for path '{rel_path}'"
            ) from exc
        take_to_segments.setdefault(take_name_from_path(rel_path), []).append((rel_path, label))
    return sort_take_segments(take_to_segments)


def load_take_labels(csv_file):
    take_to_segments = parse_take_label_tsv(csv_file)
    examples = list(take_to_segments.keys())
    shuffle_in_place(examples)
    return take_to_segments, examples


def contiguous_window(items, max_items):
    if not max_items or len(items) <= max_items:
        return items
    start = random.randint(0, len(items) - max_items)
    end = start + max_items
    return items[start:end]


def pack_clip_batch(per_take_feats, per_take_targets, num_classes):
    batch_size = len(per_take_feats)
    clip_counts = [len(clips) for clips in per_take_feats]
    max_clips = max(clip_counts) if clip_counts else 0

    clip_len = 0
    feature_dim = 0
    for clips in per_take_feats:
        if clips:
            clip_len, feature_dim = clips[0].shape
            break

    for clips in per_take_feats:
        for clip in clips:
            if clip.shape != (clip_len, feature_dim):
                raise ValueError(
                    f"Inconsistent clip shapes, expected {(clip_len, feature_dim)} but got {clip.shape}"
                )

    batch_input = torch.zeros(batch_size, max_clips, clip_len, feature_dim, dtype=torch.float32)
    batch_target = torch.ones(batch_size, max_clips, clip_len, dtype=torch.long) * (-100)
    mask = torch.zeros(batch_size, num_classes, max_clips, clip_len, dtype=torch.float32)

    for batch_idx, (clips, targets) in enumerate(zip(per_take_feats, per_take_targets)):
        for clip_idx, (feat_arr, target_arr) in enumerate(zip(clips, targets)):
            batch_input[batch_idx, clip_idx] = torch.from_numpy(feat_arr)
            batch_target[batch_idx, clip_idx] = torch.from_numpy(target_arr)
            mask[batch_idx, :, clip_idx, :] = 1.0

    return batch_input, batch_target, mask


def pack_sequence_batch(batch_inputs, batch_targets, num_classes):
    lengths = [len(target) for target in batch_targets]
    max_time = max(lengths) if lengths else 0
    feature_dim = batch_inputs[0].shape[0] if batch_inputs else 0
    batch_size = len(batch_inputs)

    batch_input = torch.zeros(batch_size, feature_dim, max_time, dtype=torch.float32)
    batch_target = torch.ones(batch_size, max_time, dtype=torch.long) * (-100)
    mask = torch.zeros(batch_size, num_classes, max_time, dtype=torch.float32)

    for batch_idx, (feat_arr, target_arr) in enumerate(zip(batch_inputs, batch_targets)):
        time_steps = feat_arr.shape[1]
        if time_steps == 0:
            continue
        batch_input[batch_idx, :, :time_steps] = torch.from_numpy(feat_arr)
        batch_target[batch_idx, :time_steps] = torch.from_numpy(target_arr)
        mask[batch_idx, :, :time_steps] = 1.0

    return batch_input, batch_target, mask


def maybe_parse_clip_times(path):
    match = START_END_RE.search(os.path.basename(path))
    if not match:
        raise ValueError(f"Cannot parse start/end from filename: {path}")
    start = float(match.group(1))
    end = float(match.group(2))
    if end < start:
        raise ValueError(f"Invalid start/end in filename: {path}")
    return start, end
