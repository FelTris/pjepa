from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from pjepa.data.surgical_phase_plstitch import (
    SurgicalPhasePLStitchArchive,
    SurgicalPhaseVideoDataset,
    collate_surgical_phase_videos,
)
from pjepa.probes.surgical_phase_linear_probe import (
    SurgicalPhaseLinearHead,
    frame_accuracy_and_macro_f1,
)
from pjepa.data.surgical_feature_cache import (
    build_student_feature_cache,
    validate_student_feature_cache,
)
from pjepa.models.factory import build_model
from pjepa.utils.runtime import load_json, resolve_device, resolve_path, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Cholec80-trained linear heads directly on "
            "the M2CAI16 test split without target-dataset training."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-source", choices=["raw", "student", "all"], default="all")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--results", default=None)
    return parser.parse_args()


def _canonical_class_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _class_mapping(
    source_names: list[str], target_names: list[str]
) -> tuple[torch.Tensor, list[str]]:
    source_by_name = {_canonical_class_name(name): index for index, name in enumerate(source_names)}
    if len(source_by_name) != len(source_names):
        raise ValueError("Source phase names are not unique after normalization.")
    mapping = torch.full((len(target_names),), -1, dtype=torch.long)
    ignored: list[str] = []
    for target_index, name in enumerate(target_names):
        source_index = source_by_name.get(_canonical_class_name(name))
        if source_index is None:
            ignored.append(name)
        else:
            mapping[target_index] = int(source_index)
    mapped_source = sorted(int(value) for value in mapping[mapping >= 0].tolist())
    if mapped_source != list(range(len(source_names))):
        raise ValueError(
            "Target dataset does not contain every source phase exactly once: "
            f"mapping={mapping.tolist()}."
        )
    return mapping, ignored


def _loader(dataset, *, device: torch.device, config: dict[str, Any]) -> DataLoader:
    num_workers = int(config.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(config.get("batch_size", 1)),
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": bool(config.get("pin_memory", device.type == "cuda")),
        "collate_fn": collate_surgical_phase_videos,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = False
    return DataLoader(**kwargs)


def _load_linear_head(
    checkpoint_path: str,
    *,
    feature_source: str,
    source_names: list[str],
    input_dim: int,
    device: torch.device,
) -> SurgicalPhaseLinearHead:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("version") != "surgical_phase_linear_probe_v1":
        raise ValueError(f"Unsupported linear checkpoint: {checkpoint_path}")
    if checkpoint.get("source_dataset") != "cholec80":
        raise ValueError("Transfer linear head was not trained on Cholec80.")
    if checkpoint.get("feature_source") != feature_source:
        raise ValueError("Linear checkpoint feature source does not match the run.")
    if list(checkpoint.get("phase_class_names", [])) != source_names:
        raise ValueError("Linear checkpoint phase classes do not match Cholec80.")
    if int(checkpoint["input_dim"]) != int(input_dim):
        raise ValueError("Linear checkpoint and target feature dimensions differ.")
    head = SurgicalPhaseLinearHead(input_dim, len(source_names)).to(device)
    head.load_state_dict(checkpoint["model_state_dict"], strict=True)
    head.eval()
    return head


@torch.no_grad()
def _evaluate(
    model,
    loader: DataLoader,
    *,
    mapping: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> dict[str, float]:
    mapping = mapping.to(device)
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    loss_sum = 0.0
    evaluated_tokens = 0
    total_tokens = 0
    sequences = 0
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True).bool()
        target_labels = batch["targets"]["phase"].to(device, non_blocking=True)
        logits = model(features)
        mapped_targets = mapping[target_labels.clamp_min(0)]
        shared_mask = valid_mask & (mapped_targets >= 0)
        selected_logits = logits[shared_mask]
        selected_targets = mapped_targets[shared_mask]
        if not selected_targets.numel():
            continue
        loss_sum += float(
            F.cross_entropy(selected_logits, selected_targets, reduction="sum").item()
        )
        predictions.append(selected_logits.argmax(dim=-1).cpu())
        targets.append(selected_targets.cpu())
        evaluated_tokens += int(selected_targets.numel())
        total_tokens += int(valid_mask.sum().item())
        sequences += int(features.shape[0])
    if not predictions:
        raise RuntimeError("No shared target-dataset phase tokens were evaluated.")
    accuracy, macro_f1 = frame_accuracy_and_macro_f1(
        torch.cat(predictions), torch.cat(targets), num_classes=num_classes
    )
    return {
        "phase_acc": float(accuracy),
        "phase_macro_f1": float(macro_f1),
        "phase_loss": loss_sum / max(1, evaluated_tokens),
        "sequences": float(sequences),
        "total_tokens": float(total_tokens),
        "evaluated_shared_tokens": float(evaluated_tokens),
        "ignored_nonshared_tokens": float(total_tokens - evaluated_tokens),
    }


def _ensure_student_cache(config: dict[str, Any], *, device: torch.device, rebuild: bool) -> str:
    paths = config["paths"]
    source_archive = resolve_path(paths["m2cai16_archive"])
    checkpoint_path = resolve_path(paths["pjepa_checkpoint"])
    cache_path = resolve_path(paths["m2cai16_student_feature_cache"])
    if not rebuild and Path(cache_path).is_file():
        validate_student_feature_cache(
            cache_path,
            source_archive_path=source_archive,
            checkpoint_path=checkpoint_path,
        )
        return cache_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    embedded = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    model_source = embedded or config
    model = build_model(model_source["model"], model_source["ssl"])
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.to(device)
    build_student_feature_cache(
        model,
        source_archive_path=source_archive,
        checkpoint_path=checkpoint_path,
        output_path=cache_path,
        device=device,
        use_bfloat16=bool(config.get("features", {}).get("extraction_use_bfloat16", False)),
    )
    return cache_path


def main() -> None:
    args = parse_args()
    config_path = str(Path(args.config).expanduser().resolve())
    config = load_json(config_path)
    seed = seed_everything(
        int(config.get("runtime", {}).get("seed", 1538574472)),
        bool(config.get("runtime", {}).get("deterministic_cudnn", True)),
    )
    device = resolve_device(args.device or config.get("runtime", {}).get("device", "auto"))
    paths = config["paths"]
    source_archive = SurgicalPhasePLStitchArchive(
        resolve_path(paths["cholec80_archive"]), expected_dataset="cholec80"
    )
    target_archive = SurgicalPhasePLStitchArchive(
        resolve_path(paths["m2cai16_archive"]), expected_dataset="m2cai16"
    )
    source_names = list(source_archive.phase_class_names)
    target_names = list(target_archive.phase_class_names)
    mapping, ignored_names = _class_mapping(source_names, target_names)

    feature_sources = ["raw", "student"] if args.feature_source == "all" else [args.feature_source]
    results: dict[str, Any] = {}
    for feature_source in feature_sources:
        feature_archive = (
            resolve_path(paths["m2cai16_archive"])
            if feature_source == "raw"
            else _ensure_student_cache(config, device=device, rebuild=args.rebuild_cache)
        )
        dataset = SurgicalPhaseVideoDataset(
            feature_archive,
            dataset="m2cai16",
            protocol="official",
            split="test",
        )
        loader = _loader(dataset, device=device, config=config.get("loader", {}))
        input_dim = int(dataset.archive.feature_dim)
        checkpoint_path = resolve_path(paths[f"{feature_source}_linear_checkpoint"])
        model = _load_linear_head(
            checkpoint_path,
            feature_source=feature_source,
            source_names=source_names,
            input_dim=input_dim,
            device=device,
        )
        key = f"{feature_source}_linear"
        metrics = _evaluate(
            model,
            loader,
            mapping=mapping,
            num_classes=len(source_names),
            device=device,
        )
        results[key] = {
            "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
            "feature_archive": str(Path(feature_archive).expanduser().resolve()),
            "metrics": metrics,
        }
        print(f"[{key}] acc={metrics['phase_acc']:.6f} macro_f1={metrics['phase_macro_f1']:.6f}")

    serializable = {
        "source_dataset": "cholec80",
        "source_protocol": "table1",
        "target_dataset": "m2cai16",
        "target_split": "test",
        "target_training_or_tuning_used": False,
        "metrics_policy": "framewise_shared_phases_only",
        "source_phase_class_names": source_names,
        "target_phase_class_names": target_names,
        "target_to_source_class_mapping": mapping.tolist(),
        "ignored_target_phase_class_names": ignored_names,
        "seed": int(seed),
        "config_path": config_path,
        "results": results,
    }
    output_value = args.results or paths.get("results_output")
    if output_value:
        output_path = Path(resolve_path(output_value)).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2)
            handle.write("\n")
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
