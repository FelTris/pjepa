from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader


from pjepa.data.surgical_phase_plstitch import (
    SurgicalPhaseVideoDataset,
    collate_surgical_phase_videos,
)
from pjepa.probes.surgical_phase_linear_probe import run_frozen_phase_probe
from pjepa.data.surgical_feature_cache import validate_student_feature_cache
from pjepa.models.factory import build_model
from pjepa.utils.runtime import load_json, resolve_device, resolve_path, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a frozen raw PL-Stitch or P-JEPA surgical phase probe."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=["cholec80", "m2cai16"], required=True)
    parser.add_argument(
        "--protocol",
        choices=["development", "table1", "official"],
        default=None,
    )
    parser.add_argument("--feature-source", choices=["raw", "student"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--student-feature-cache",
        default=None,
        help=(
            "Optional precomputed P-JEPA archive. This avoids re-encoding all "
            "videos when retraining a student-feature linear head."
        ),
    )
    parser.add_argument("--results", default=None)
    parser.add_argument(
        "--train-video-id",
        default=None,
        help="Optional single official-train video override for one-video sweeps.",
    )
    parser.add_argument(
        "--updates-per-epoch",
        type=int,
        default=None,
        help="Optional probe updates-per-epoch override.",
    )
    parser.add_argument(
        "--model-output",
        default=None,
        help=(
            "Output path for the trained linear head. When omitted but --results "
            "is set, the checkpoint is saved beside the JSON with a .pt suffix."
        ),
    )
    parser.add_argument(
        "--probe-epochs",
        type=int,
        default=None,
        help="Optional probe-epoch override, useful for smoke tests.",
    )
    return parser.parse_args()


def _loader(
    dataset: SurgicalPhaseVideoDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": False,
        "drop_last": False,
        "num_workers": int(num_workers),
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_surgical_phase_videos,
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = False
    return DataLoader(**kwargs)


def main() -> None:
    args = parse_args()
    config_path = str(Path(args.config).expanduser().resolve())
    config = load_json(config_path)
    runtime_cfg = config.get("runtime", {})
    paths_cfg = config["paths"]
    probe_cfg = dict(config["probe"])
    data_cfg = config.get("data", {})
    if args.probe_epochs is not None:
        if args.probe_epochs <= 0:
            raise ValueError("--probe-epochs must be positive.")
        probe_cfg["epochs"] = int(args.probe_epochs)
    if args.updates_per_epoch is not None:
        if args.updates_per_epoch <= 0:
            raise ValueError("--updates-per-epoch must be positive.")
        probe_cfg["updates_per_epoch"] = int(args.updates_per_epoch)
    seed = seed_everything(
        int(runtime_cfg.get("seed", 1538574472)),
        bool(runtime_cfg.get("deterministic_cudnn", True)),
    )
    device = resolve_device(runtime_cfg.get("device", "auto"))

    if args.dataset == "cholec80":
        archive_path = resolve_path(paths_cfg["cholec80_archive"])
        protocol = args.protocol or "table1"
        if protocol not in {"development", "table1"}:
            raise ValueError("Cholec80 protocol must be development or table1.")
        eval_split = "val" if protocol == "development" else "test"
        select_best_on_eval = protocol == "development"
    else:
        archive_path = resolve_path(paths_cfg["m2cai16_archive"])
        protocol = args.protocol or "official"
        if protocol != "official":
            raise ValueError("M2CAI16 protocol must be official.")
        eval_split = "test"
        select_best_on_eval = False

    train_video_ids = (
        [str(args.train_video_id)]
        if args.train_video_id is not None
        else data_cfg.get("train_video_ids")
    )
    train_dataset = SurgicalPhaseVideoDataset(
        archive_path,
        dataset=args.dataset,
        protocol=protocol,
        split="train",
        video_ids=train_video_ids,
    )
    eval_dataset = SurgicalPhaseVideoDataset(
        archive_path,
        dataset=args.dataset,
        protocol=protocol,
        split=eval_split,
    )
    extraction_batch_size = int(probe_cfg.get("extraction_batch_size", 1))
    loader_cfg = config.get("loader", {}).get("surgical_phase", {})
    num_workers = int(loader_cfg.get("num_workers", 0))
    train_loader = _loader(
        train_dataset,
        batch_size=extraction_batch_size,
        num_workers=num_workers,
        device=device,
    )
    eval_loader = _loader(
        eval_dataset,
        batch_size=extraction_batch_size,
        num_workers=num_workers,
        device=device,
    )

    source_archive_path = archive_path
    model = None
    checkpoint_path = args.checkpoint
    checkpoint = None
    extraction_feature_source = args.feature_source
    if args.feature_source == "student":
        if not checkpoint_path:
            raise ValueError("Student probing requires --checkpoint.")
        checkpoint_path = resolve_path(checkpoint_path)
        if args.student_feature_cache:
            archive_path = resolve_path(args.student_feature_cache)
            validate_student_feature_cache(
                archive_path,
                source_archive_path=source_archive_path,
                checkpoint_path=checkpoint_path,
            )
            extraction_feature_source = "raw"
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            embedded = checkpoint.get("config") if isinstance(checkpoint, dict) else None
            model_cfg = (embedded or config)["model"]
            ssl_cfg = (embedded or config)["ssl"]
            model = build_model(model_cfg, ssl_cfg)
            state = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state, strict=True)
            model.to(device)

        train_dataset = SurgicalPhaseVideoDataset(
            archive_path,
            dataset=args.dataset,
            protocol=protocol,
            split="train",
            video_ids=train_video_ids,
        )
        eval_dataset = SurgicalPhaseVideoDataset(
            archive_path,
            dataset=args.dataset,
            protocol=protocol,
            split=eval_split,
        )
        train_loader = _loader(
            train_dataset,
            batch_size=extraction_batch_size,
            num_workers=num_workers,
            device=device,
        )
        eval_loader = _loader(
            eval_dataset,
            batch_size=extraction_batch_size,
            num_workers=num_workers,
            device=device,
        )

    result = run_frozen_phase_probe(
        model,
        train_loader,
        eval_loader,
        device=device,
        config=probe_cfg,
        num_classes=train_dataset.archive.num_classes,
        feature_source=extraction_feature_source,
        select_best_on_eval=select_best_on_eval,
        log_prefix=f"{args.dataset}/{protocol}/{args.feature_source}",
    )
    model_output = args.model_output
    if model_output is None and args.results:
        model_output = str(Path(resolve_path(args.results)).with_suffix(".pt"))
    resolved_model_output = None
    if model_output is not None:
        model_path = Path(resolve_path(model_output)).expanduser().resolve()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        classifier_weight = result["head_state_dict"]["classifier.weight"]
        torch.save(
            {
                "version": "surgical_phase_linear_probe_v1",
                "model_state_dict": result["head_state_dict"],
                "input_dim": int(classifier_weight.shape[1]),
                "num_classes": int(classifier_weight.shape[0]),
                "phase_class_names": list(train_dataset.archive.phase_class_names),
                "source_dataset": args.dataset,
                "source_protocol": protocol,
                "feature_source": args.feature_source,
                "feature_archive": str(Path(archive_path).expanduser().resolve()),
                "pjepa_checkpoint": checkpoint_path,
                "probe_config": copy.deepcopy(probe_cfg),
                "train_video_ids": list(train_dataset.video_ids),
                "config_path": config_path,
                "metrics": dict(result["metrics"]),
                "seed": int(seed),
            },
            model_path,
        )
        resolved_model_output = str(model_path)
    serializable = {
        "dataset": args.dataset,
        "protocol": protocol,
        "train_split": "train",
        "eval_split": eval_split,
        "feature_source": args.feature_source,
        "checkpoint": checkpoint_path,
        "feature_archive": str(Path(archive_path).expanduser().resolve()),
        "selection_used_eval_metrics": select_best_on_eval,
        "phase_class_names": train_dataset.archive.phase_class_names,
        "metrics": result["metrics"],
        "train_tokens": result["train_tokens"],
        "eval_tokens": result["eval_tokens"],
        "train_sequences": result["train_sequences"],
        "eval_sequences": result["eval_sequences"],
        "train_video_ids": list(train_dataset.video_ids),
        "eval_video_ids": list(eval_dataset.video_ids),
        "updates_per_epoch": result["updates_per_epoch"],
        "default_updates_per_epoch": result["default_updates_per_epoch"],
        "seed": seed,
        "model_output": resolved_model_output,
        "config_path": config_path,
    }
    if args.results:
        output_path = Path(resolve_path(args.results))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2)
            handle.write("\n")
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
