"""Strict loading of released and legacy feature-JEPA checkpoints.

Architecture metadata is required: attention, RoPE, and head count cannot
be reconstructed reliably from parameter shapes.
"""

from __future__ import annotations

import hashlib
import json
import os
from importlib.resources import files
from pathlib import Path
from typing import Any

import torch

from pjepa.models.feature_vjepa_rope import FeatureJEPA


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registry() -> dict[str, Any]:
    return json.loads(files("pjepa").joinpath("artifacts.json").read_text())


def artifact_hashes(entry: dict) -> set[str]:
    """Recognize a release bundle and its original, tensor-identical source."""
    return {entry[key] for key in ("sha256", "source_sha256") if entry.get(key)}


def assert_encoder_identity(head_metadata: dict, encoder_path: str | Path) -> None:
    """Validate head/encoder provenance by content, allowing files to move."""
    expected_hash = head_metadata.get("pjepa_checkpoint_sha256")
    old_path = head_metadata.get("pjepa_checkpoint")
    if not expected_hash and old_path:
        for entry in registry()["encoders"].values():
            if str(old_path).endswith(entry["source"]) or Path(old_path).name == entry["filename"]:
                expected_hash = entry["sha256"]
                break
        if not expected_hash and Path(old_path).is_file():
            expected_hash = sha256_file(old_path)
    if old_path and not expected_hash:
        raise ValueError(
            "Cannot verify this head's encoder. Supply pjepa_checkpoint_sha256 in its metadata."
        )
    if expected_hash:
        allowed_hashes = {expected_hash}
        for entry in registry()["encoders"].values():
            if expected_hash in artifact_hashes(entry):
                allowed_hashes.update(artifact_hashes(entry))
        if sha256_file(encoder_path) not in allowed_hashes:
            raise ValueError("Linear head and requested P-JEPA checkpoint do not match.")


def architecture_from_config(config: dict) -> dict:
    """Translate either the surgical or frame-feature experiment schema."""
    model, ssl = config.get("model", {}), config.get("ssl", {})
    data = config.get("data", {})
    surgical = "input_dim" in model
    return {
        "d_in": int(
            model["input_dim"] if surgical else data.get("input_features_dim", data["features_dim"])
        ),
        "d_model": int(model["dim"] if surgical else data["features_dim"]),
        "enc_depth": int(model.get("enc_depth", ssl.get("enc_depth", 4))),
        "enc_heads": int(model.get("enc_heads", ssl.get("enc_heads", 16))),
        "pred_depth": int(model.get("pred_depth", ssl.get("pred_depth", 2))),
        "pred_heads": int(model.get("pred_heads", ssl.get("pred_heads", 16))),
        "drop_path_prob": float(model.get("drop_path_prob", 0.0)),
        "student_encoder_attention": str(ssl.get("student_encoder_attention", "block_causal")),
        "student_encoder_block_size": int(
            ssl.get("student_encoder_block_size", 32 if surgical else 16)
        ),
        "rope_mode": str(ssl.get("rope_mode", "1d_flat" if surgical else "2d_clip_frame")),
    }


def resolve_checkpoint(checkpoint: str | Path, checkpoint_root: str | Path | None = None):
    catalog = registry()["encoders"]
    key = str(checkpoint)
    entry = catalog.get(key)
    if entry:
        root = Path(
            checkpoint_root or os.environ.get("PJEPA_CHECKPOINT_ROOT", "checkpoints/weights")
        )
        path = root / entry["filename"]
    else:
        path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Set PJEPA_CHECKPOINT_ROOT or pass a file path."
        )
    return path, entry


def load_model(
    checkpoint: str | Path,
    *,
    checkpoint_root: str | Path | None = None,
    architecture: dict | None = None,
    device: str | torch.device = "cpu",
) -> tuple[FeatureJEPA, dict]:
    """Load by registry ID or original path; return model and resolved metadata.

    Known original bare checkpoints are recognized by content hash. For an
    unregistered bare state dict, provide the complete FeatureJEPA constructor
    configuration.
    """
    path, entry = resolve_checkpoint(checkpoint, checkpoint_root)
    digest = sha256_file(path)
    if entry and digest != entry["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    if entry is None:
        entry = next(
            (e for e in registry()["encoders"].values() if digest in artifact_hashes(e)), None
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Expected a state dictionary or a checkpoint bundle.")
    embedded = payload.get("architecture")
    if embedded is None and "config" in payload and "model_state_dict" in payload:
        embedded = architecture_from_config(payload["config"])
    expected = dict(entry["architecture"]) if entry else embedded
    if expected is None and architecture is None:
        raise ValueError("Unregistered bare checkpoint: supply complete architecture metadata.")
    if architecture is not None and expected is not None and architecture != expected:
        differences = {
            k: (expected.get(k), architecture.get(k))
            for k in expected.keys() | architecture.keys()
            if expected.get(k) != architecture.get(k)
        }
        raise ValueError(f"Architecture conflicts with checkpoint metadata: {differences}")
    resolved = dict(expected if expected is not None else architecture)
    required = {
        "d_in",
        "d_model",
        "enc_depth",
        "enc_heads",
        "pred_depth",
        "pred_heads",
        "drop_path_prob",
        "student_encoder_attention",
        "student_encoder_block_size",
        "rope_mode",
    }
    if set(resolved) != required:
        raise ValueError(f"Architecture must provide exactly {sorted(required)}")
    model = FeatureJEPA(**resolved)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=True)
    model.to(device).eval()
    return model, {
        "architecture": resolved,
        "sha256": digest,
        "path": str(path),
        "oracle_boundaries": resolved["student_encoder_attention"] == "clip_causal",
    }
