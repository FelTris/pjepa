"""Build feature generators and preserve deterministic training subsets."""

import csv
import math
from pathlib import Path
import random
from pjepa.data.assembly101_frame_take import Assembly101FrameTakeBatchGenerator
from pjepa.data.packed_assembly101_frame_archive import PackedAssembly101FrameTakeBatchGenerator
from pjepa.utils.runtime import prepare_generator, resolve_path


def resolve_val_max_take_len(data_cfg):
    if "val_max_take_len" in data_cfg:
        return data_cfg["val_max_take_len"]
    return data_cfg.get("max_take_len")


def resolve_probe_train_max_take_len(probe_cfg, data_cfg):
    if "train_max_take_len" in probe_cfg:
        return probe_cfg["train_max_take_len"]
    return data_cfg.get("max_take_len")


def _get_generator_take_to_segments(generator):
    take_to_segments = getattr(generator, "take_to_segments", None)
    if take_to_segments is None and hasattr(generator, "dataset"):
        take_to_segments = getattr(generator.dataset, "take_to_segments", None)
    if take_to_segments is None:
        raise ValueError("Could not find take_to_segments on train generator.")
    return take_to_segments


def _set_generator_examples(generator, examples):
    updated_examples = list(examples)
    if hasattr(generator, "list_of_examples"):
        generator.list_of_examples = list(updated_examples)
    if hasattr(generator, "dataset") and hasattr(generator.dataset, "list_of_examples"):
        generator.dataset.list_of_examples = list(updated_examples)
    for attr_name, attr_value in {
        "_loader": None,
        "_iterator": None,
        "_remaining_batches": None,
        "_batch_size": None,
    }.items():
        if hasattr(generator, attr_name):
            setattr(generator, attr_name, attr_value)


def _segment_label(segment):
    if hasattr(segment, "label"):
        return int(segment.label)
    return int(segment[-1])


def _select_cover_takes(take_to_labels, rng):
    remaining_classes = set()
    for labels in take_to_labels.values():
        remaining_classes.update(labels)
    selected = []
    available = sorted(take_to_labels)
    while remaining_classes:
        rng.shuffle(available)
        best_take = None
        best_gain = set()
        for take in available:
            gain = take_to_labels[take] & remaining_classes
            if len(gain) > len(best_gain):
                best_take = take
                best_gain = gain
        if best_take is None or not best_gain:
            raise ValueError(
                "Could not build a class-covering train subset from the loaded train split."
            )
        selected.append(best_take)
        remaining_classes.difference_update(best_gain)
        available.remove(best_take)
    return selected


def maybe_apply_train_subset(generator, subset_fraction, ensure_class_coverage, seed):
    if subset_fraction is None:
        return {
            "enabled": False,
            "fraction": None,
            "requested_fraction": None,
            "ensure_class_coverage": bool(ensure_class_coverage),
        }
    fraction = float(subset_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"train_subset_fraction must be in (0, 1], got {subset_fraction!r}.")
    take_to_segments = _get_generator_take_to_segments(generator)
    all_takes = sorted(take_to_segments)
    total_takes = len(all_takes)
    if total_takes == 0:
        raise ValueError("Train split is empty; cannot apply train_subset_fraction.")
    requested_takes = min(total_takes, max(1, int(math.ceil(fraction * total_takes))))
    take_to_labels = {
        take: {_segment_label(segment) for segment in segments}
        for (take, segments) in take_to_segments.items()
    }
    available_classes = sorted({label for labels in take_to_labels.values() for label in labels})
    missing_classes = []
    num_classes = getattr(generator, "num_classes", None)
    if num_classes is None and hasattr(generator, "dataset"):
        num_classes = getattr(generator.dataset, "num_classes", None)
    if num_classes is not None:
        available_set = set(available_classes)
        missing_classes = [
            class_idx for class_idx in range(int(num_classes)) if class_idx not in available_set
        ]
    rng = random.Random(int(seed))
    cover_takes = []
    if ensure_class_coverage:
        cover_takes = _select_cover_takes(take_to_labels, rng)
        if requested_takes < len(cover_takes):
            min_fraction = len(cover_takes) / float(total_takes)
            raise ValueError(
                f"Requested train_subset_fraction is too small to keep every train-split class represented. Requested {fraction:.6f} -> {requested_takes}/{total_takes} takes, but class coverage needs at least {len(cover_takes)} takes ({min_fraction:.6f} of the train split)."
            )
    selected = list(cover_takes)
    selected_set = set(selected)
    remaining = [take for take in all_takes if take not in selected_set]
    rng.shuffle(remaining)
    selected.extend(remaining[: requested_takes - len(selected)])
    _set_generator_examples(generator, selected)
    generator.reset()
    actual_takes = len(selected)
    actual_fraction = actual_takes / float(total_takes)
    print(
        f"Train subset enabled: selected {actual_takes}/{total_takes} takes (requested_fraction={fraction:.6f}, actual_fraction={actual_fraction:.6f}, ensure_class_coverage={bool(ensure_class_coverage)}, class_cover_takes={len(cover_takes)}, classes_in_train_split={len(available_classes)})"
    )
    if missing_classes:
        preview = missing_classes[:10]
        suffix = "..." if len(missing_classes) > 10 else ""
        print(
            f"Train split is missing class ids from the full configured label space: {preview}{suffix}"
        )
    return {
        "enabled": True,
        "fraction": actual_fraction,
        "requested_fraction": fraction,
        "requested_takes": requested_takes,
        "num_takes": actual_takes,
        "total_takes": total_takes,
        "ensure_class_coverage": bool(ensure_class_coverage),
        "class_cover_takes": len(cover_takes),
        "classes_in_train_split": len(available_classes),
        "missing_class_ids": missing_classes,
        "seed": int(seed),
    }


def infer_num_classes_from_manifests(manifest_paths, label_column: str) -> int:
    max_label = None
    unique_labels = set()
    for manifest_path in manifest_paths:
        with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if label_column not in (reader.fieldnames or []):
                raise ValueError(
                    f"Manifest {manifest_path} is missing label column '{label_column}'."
                )
            for row in reader:
                label = int(row[label_column])
                unique_labels.add(label)
                max_label = label if max_label is None else max(max_label, label)
    if max_label is None:
        raise ValueError(f"Could not infer num_classes from manifests: {manifest_paths}")
    inferred = int(max_label) + 1
    print(
        f"Inferred num_classes={inferred} from label column '{label_column}' ({len(unique_labels)} unique labels observed, max={max_label})."
    )
    return inferred


def resolve_ltcontext_annotations_root(paths_cfg, labels_root):
    annotations_root = paths_cfg.get("annotations_root")
    if annotations_root:
        return resolve_path(annotations_root)
    labels_root_path = Path(resolve_path(labels_root)) if labels_root else Path(".")
    if (labels_root_path / "actions.csv").exists() and (
        labels_root_path / "coarse_labels"
    ).exists():
        return str(labels_root_path)
    return str(labels_root_path / "coarse-annotations")


def infer_num_classes_from_ltcontext_actions(
    annotations_root, action_id_column: str = "action_id"
) -> int:
    actions_path = Path(annotations_root) / "actions.csv"
    if not actions_path.exists():
        raise FileNotFoundError(
            f"Could not infer num_classes; actions file not found: {actions_path}"
        )
    max_label = None
    unique_labels = set()
    with actions_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if action_id_column not in (reader.fieldnames or []):
            raise ValueError(f"{actions_path} is missing label column '{action_id_column}'.")
        for row in reader:
            label = int(row[action_id_column])
            unique_labels.add(label)
            max_label = label if max_label is None else max(max_label, label)
    if max_label is None:
        raise ValueError(f"Could not infer num_classes from empty actions file: {actions_path}")
    inferred = int(max_label) + 1
    print(
        f"Inferred num_classes={inferred} from LTContext actions file ({len(unique_labels)} unique labels observed, max={max_label})."
    )
    return inferred


def _manifest_path_for_split(name, data_cfg, labels_root):
    split_key = "train_manifest" if name.lower() == "train" else "val_manifest"
    manifest_value = data_cfg.get(split_key, data_cfg.get("manifest"))
    if not manifest_value:
        raise ValueError(f"{split_key} is required for Assembly101 SSL training.")
    return resolve_path(manifest_value, labels_root)


def build_batch_generator(
    name, manifest_path, data_cfg, paths_cfg, loader_cfg, max_take_len=None, split_role=None
):
    split_role = str(split_role or name).lower()
    archive_root = paths_cfg.get("archive_root", "")
    backend = str(loader_cfg.get("backend", "assembly101_frame_flat")).lower()
    feature_source = str(data_cfg.get("feature_source", "")).strip().lower()
    archive_key = "train_archive_path" if split_role == "train" else "val_archive_path"
    archive_path = data_cfg.get(archive_key, data_cfg.get("archive_path"))
    if backend in {
        "assembly101_ltcontext_lmdb",
        "assembly101_ltcontext_tsm_lmdb",
    } or feature_source in {"ltcontext_tsm_lmdb", "assembly101_ltcontext_tsm_lmdb"}:
        from pjepa.data.assembly101_ltcontext_lmdb_take import (
            Assembly101LTContextLmdbTakeBatchGenerator,
        )

        features_root = resolve_path(paths_cfg["features_root"])
        annotations_root = resolve_ltcontext_annotations_root(
            paths_cfg, paths_cfg.get("labels_root", "")
        )
        return prepare_generator(
            name,
            Assembly101LTContextLmdbTakeBatchGenerator,
            manifest_path,
            num_classes=data_cfg.get("num_classes"),
            features_root=features_root,
            annotations_root=annotations_root,
            lmdb_subdir=str(data_cfg.get("lmdb_subdir", "db_TSM_features")),
            feature_dim=int(
                data_cfg.get("input_features_dim", data_cfg.get("raw_features_dim", 2048))
            ),
            sample_rate=int(data_cfg.get("sample_rate", 1)),
            max_take_len=max_take_len,
            action_id_column=str(data_cfg.get("action_id_column", "action_id")),
            action_cls_column=str(data_cfg.get("action_cls_column", "action_cls")),
            video_id_column=str(data_cfg.get("video_id_column", "video_id")),
            view_column=str(data_cfg.get("view_column", "view")),
            action_type_column=str(data_cfg.get("action_type_column", "action_type")),
            video_end_frame_column=str(data_cfg.get("video_end_frame_column", "video_end_frame")),
            cache_num_videos=int(loader_cfg.get("cache_num_videos", 4)),
            strict_missing_frames=bool(loader_cfg.get("strict_missing_frames", True)),
        )
    if backend in {"assembly101_lmdb", "assembly101_tsm_lmdb"} or feature_source in {
        "tsm_lmdb",
        "assembly101_tsm_lmdb",
    }:
        from pjepa.data.assembly101_lmdb_take import Assembly101LmdbTakeBatchGenerator

        features_root = resolve_path(paths_cfg["features_root"])
        split_key = "train_split" if split_role == "train" else "val_split"
        target_fps = data_cfg.get("target_fps", data_cfg.get("target_frame_rate"))
        return prepare_generator(
            name,
            Assembly101LmdbTakeBatchGenerator,
            manifest_path,
            num_classes=data_cfg.get("num_classes"),
            features_root=features_root,
            feature_source=feature_source or "tsm_lmdb",
            lmdb_subdir=str(data_cfg.get("lmdb_subdir", "db_TSM_features")),
            feature_dim=int(
                data_cfg.get("input_features_dim", data_cfg.get("raw_features_dim", 2048))
            ),
            sample_rate=int(data_cfg.get("sample_rate", 1)),
            target_fps=target_fps,
            source_fps=float(data_cfg.get("source_fps", 30.0)),
            max_take_len=max_take_len,
            split_name=data_cfg.get(split_key, split_role),
            label_column=str(data_cfg.get("label_column", "action_id")),
            video_path_column=str(data_cfg.get("video_path_column", "video_path")),
            split_column=str(data_cfg.get("split_column", "official_split")),
            sample_id_column=str(data_cfg.get("sample_id_column", "sample_uid")),
            action_type_column=str(data_cfg.get("action_type_column", "action_type")),
            take_key_mode=str(data_cfg.get("take_key_mode", "action_type_video_view")),
            start_frame_column=str(data_cfg.get("start_frame_column", "start_frame")),
            end_frame_column=str(data_cfg.get("end_frame_column", "end_frame")),
            source_fps_column=str(data_cfg.get("source_fps_column", "annotation_fps")),
            cache_num_videos=int(loader_cfg.get("cache_num_videos", 4)),
            strict_missing_frames=bool(loader_cfg.get("strict_missing_frames", True)),
        )
    if backend in {"assembly101_frame_flat_archive", "assembly101_frame_archive"} or archive_path:
        if not archive_path:
            raise ValueError(
                f"{name} Assembly101 archive loader requested but no archive path was configured."
            )
        archive_path = resolve_path(archive_path, archive_root)
        return prepare_generator(
            name,
            PackedAssembly101FrameTakeBatchGenerator,
            manifest_path,
            num_classes=data_cfg.get("num_classes"),
            archive_path=archive_path,
            sample_rate=int(data_cfg.get("sample_rate", 1)),
            max_take_len=max_take_len,
            split_name=data_cfg.get(
                "train_split" if split_role == "train" else "val_split", split_role
            ),
            label_column=str(data_cfg.get("label_column", "action_id")),
            video_path_column=str(data_cfg.get("video_path_column", "video_path")),
            split_column=str(data_cfg.get("split_column", "official_split")),
            sample_id_column=str(data_cfg.get("sample_id_column", "sample_uid")),
            action_type_column=str(data_cfg.get("action_type_column", "action_type")),
            take_key_mode=str(data_cfg.get("take_key_mode", "action_type_video_view")),
            start_sec_column=str(data_cfg.get("start_sec_column", "start_sec")),
            end_sec_column=str(data_cfg.get("end_sec_column", "end_sec")),
            cache_in_memory=bool(loader_cfg.get("cache_in_memory", True)),
            preload_in_parent=bool(loader_cfg.get("preload_in_parent", True)),
            max_background_segment_tokens=data_cfg.get("max_background_segment_tokens"),
            background_label_id=int(data_cfg.get("background_label_id", 0)),
            num_workers=int(loader_cfg.get("num_workers", 0)),
            persistent_workers=bool(loader_cfg.get("persistent_workers", False)),
            pin_memory=bool(loader_cfg.get("pin_memory", False)),
            prefetch_factor=loader_cfg.get("prefetch_factor", 2),
            multiprocessing_context=loader_cfg.get("multiprocessing_context"),
        )
    if backend != "assembly101_frame_flat":
        raise ValueError(
            f"Unsupported Assembly101 backend '{backend}'. Use 'assembly101_frame_flat' or 'assembly101_frame_flat_archive'."
        )
    features_root = resolve_path(paths_cfg["features_root"])
    split_key = "train_split" if split_role == "train" else "val_split"
    split_name = data_cfg.get(split_key, split_role)
    label_column = str(data_cfg.get("label_column", "action_id"))
    return prepare_generator(
        name,
        Assembly101FrameTakeBatchGenerator,
        manifest_path,
        num_classes=data_cfg.get("num_classes"),
        features_root=features_root,
        sample_rate=int(data_cfg.get("sample_rate", 1)),
        max_take_len=max_take_len,
        split_name=split_name,
        label_column=label_column,
        video_path_column=str(data_cfg.get("video_path_column", "video_path")),
        split_column=str(data_cfg.get("split_column", "official_split")),
        sample_id_column=str(data_cfg.get("sample_id_column", "sample_uid")),
        start_sec_column=str(data_cfg.get("start_sec_column", "start_sec")),
        end_sec_column=str(data_cfg.get("end_sec_column", "end_sec")),
        cache_num_videos=int(loader_cfg.get("cache_num_videos", 8)),
    )
