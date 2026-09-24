from pathlib import Path


def feature_rel_path_from_video_path(video_path: str) -> str:
    path = str(video_path).strip()
    suffix = Path(path).suffix.lower()
    if suffix in {".mp4", ".avi"}:
        return str(Path(path).with_suffix(".pt"))
    return path


def action_type_from_row(
    row,
    action_type_column: str = "action_type",
    sample_id_column: str = "sample_uid",
) -> str:
    action_type = str(row.get(action_type_column, "")).strip().lower()
    if action_type:
        return action_type

    sample_id = str(row.get(sample_id_column, "")).strip()
    if sample_id:
        parts = sample_id.split("::")
        if len(parts) >= 2:
            sample_recording_id = parts[1]
            if sample_recording_id.startswith("assembly_"):
                return "assembly"
            if sample_recording_id.startswith("disassembly_"):
                return "disassembly"

    raise ValueError(
        f"Could not infer Assembly101 action_type from row. "
        f"Expected column '{action_type_column}' or sample_uid prefixed with assembly_/disassembly_."
    )


def take_key_from_video_path(
    video_path: str,
    action_type: str,
    take_key_mode: str = "action_type_video_view",
) -> str:
    feature_rel_path = feature_rel_path_from_video_path(video_path)
    mode = str(take_key_mode).strip().lower()
    if mode in {"video_path", "feature_path"}:
        return feature_rel_path
    if mode not in {"action_type_video_view", "ltcontext"}:
        raise ValueError(
            f"Unsupported Assembly101 take_key_mode='{take_key_mode}'. "
            "Use 'action_type_video_view' or 'video_path'."
        )

    path = Path(str(video_path).strip())
    view = path.stem
    recording_id = str(path.parent).strip()
    if not recording_id or recording_id == ".":
        raise ValueError(f"Assembly101 video_path must include recording/view: {video_path}")
    action_type = str(action_type).strip().lower()
    if not action_type:
        raise ValueError(f"Assembly101 action_type is empty for video_path={video_path}")
    return f"{action_type}/{recording_id}/{view}"


def take_key_from_row(
    row,
    video_path_column: str = "video_path",
    action_type_column: str = "action_type",
    sample_id_column: str = "sample_uid",
    take_key_mode: str = "action_type_video_view",
) -> str:
    action_type = action_type_from_row(
        row,
        action_type_column=action_type_column,
        sample_id_column=sample_id_column,
    )
    return take_key_from_video_path(
        str(row[video_path_column]).strip(),
        action_type=action_type,
        take_key_mode=take_key_mode,
    )
