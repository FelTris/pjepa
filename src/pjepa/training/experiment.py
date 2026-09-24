"""Construct feature experiments and translate portable configuration sections."""

from pathlib import Path

from pjepa.checkpoints import architecture_from_config, load_model
from pjepa.data.experiments import (
    _manifest_path_for_split,
    build_batch_generator,
    infer_num_classes_from_manifests,
    maybe_apply_train_subset,
)
from pjepa.training.feature_ssl import Trainer


def build_trainer(config, num_classes):
    data, ssl, train = config["data"], config["ssl"], config["train"]
    exp = config["experiment"]
    return Trainer(
        dim=int(data["features_dim"]),
        enc_depth=int(ssl["enc_depth"]),
        pred_depth=int(ssl["pred_depth"]),
        ema_momentum=float(ssl["ema_momentum"]),
        lr=float(train["lr"]),
        num_classes=num_classes,
        dataset=exp["dataset"],
        split=exp["split"],
        mask_ratio=float(ssl["mask_ratio"]),
        input_dim=int(data.get("input_features_dim", data["features_dim"])),
        enc_heads=int(ssl.get("enc_heads", 16)),
        pred_heads=int(ssl.get("pred_heads", 16)),
        student_encoder_attention=ssl["student_encoder_attention"],
        student_encoder_block_size=int(ssl.get("student_encoder_block_size", 16)),
        rope_mode=ssl.get("rope_mode", "2d_clip_frame"),
        directional_mask_prob=float(ssl.get("directional_mask_prob", 0)),
        directional_future_mask_ratio=float(ssl.get("directional_future_mask_ratio", 0.75)),
        directional_min_context_clips=int(ssl.get("directional_min_context_clips", 1)),
        directional_min_future_clips=int(ssl.get("directional_min_future_clips", 1)),
        use_wandb=bool(config.get("wandb", {}).get("use_wandb", False)),
        wandb_project=config.get("wandb", {}).get("project", "pjepa"),
        run_name=exp["run_name"],
    )


def build_data(config, *, include_train=True):
    data, paths, loader = config["data"], config["paths"], config.get("loader", {})
    manifests = {
        s: _manifest_path_for_split(s, data, paths.get("labels_root", "")) for s in ("train", "val")
    }
    num_classes = data.get("num_classes") or infer_num_classes_from_manifests(
        list(manifests.values()), data["label_column"]
    )
    data = dict(data, num_classes=int(num_classes))
    generators = {}
    for role in ("train", "val") if include_train else ("val",):
        max_len = data.get("max_take_len") if role == "train" else data.get("val_max_take_len")
        generators[role] = build_batch_generator(
            role,
            manifests[role],
            data_cfg=data,
            paths_cfg=paths,
            loader_cfg=loader,
            max_take_len=max_len,
            split_role=role,
        )
    return generators, int(num_classes), data, manifests


def load_encoder_into(trainer, config, checkpoint, device):
    loaded, metadata = load_model(
        checkpoint, architecture=architecture_from_config(config), device="cpu"
    )
    trainer.model.load_state_dict(loaded.state_dict(), strict=True)
    trainer.model.to(device)
    trainer.teacher.load_state_dict(trainer.model.student_enc.state_dict(), strict=True)
    trainer.teacher.to(device)
    return metadata


def probe_arguments(config):
    probe, evaluation = config["probe"], config.get("evaluation", {})
    kind = evaluation.get("temporal_probe_kind", probe.get("mode", "linear"))
    if kind not in {"linear", "ltcontext", "causal_ltcontext"}:
        raise ValueError("Probe kind must be linear, ltcontext, or causal_ltcontext.")
    solver = probe.get("solver", probe.get("SOLVER"))
    temporal = kind != "linear"
    return dict(
        batch_size=int(evaluation.get("batch_size", 1)),
        val_batch_size=int(evaluation.get("val_batch_size", 1)),
        probe_epochs=int(probe["epochs"]),
        probe_lr_lin=float(probe.get("lr_lin", 0.001)),
        probe_lr_ltcontext=float(probe.get("lr_ltcontext", 0.00025)),
        probe_weight_decay=float(probe.get("weight_decay", 0)),
        select_metric=evaluation.get(
            "select_metric", "ltcontext_combined" if temporal else "lin_combined"
        ),
        lin_lr_milestones=probe.get("lin_lr_milestones", [5, 15]),
        ltcontext_lr_milestones=probe.get("ltcontext_lr_milestones", [10, 20]),
        ltcontext_solver_cfg=solver,
        ltcontext_cfg_overrides=probe.get("ltcontext"),
        lr_gamma=float(probe.get("lr_gamma", 0.1)),
        linear_only=not temporal,
        temporal_probe_kind=kind if temporal else None,
        linear_probe_pool=bool(probe.get("linear_probe_pool", False)),
        probe_feature_source=evaluation.get("probe_feature_source", "student"),
        background_label_id=int(probe.get("background_label_id", -100)),
        linear_probe_head_mode=probe.get("linear_head_mode", "single"),
        linear_probe_activity_loss_weight=float(probe.get("activity_loss_weight", 1)),
        linear_probe_foreground_loss_weight=float(probe.get("foreground_loss_weight", 1)),
        progress_log_every=int(probe.get("progress_log_every", 0)),
    )


def probe_training_data(config, data, manifests, generators):
    maximum = config["probe"].get("train_max_take_len")
    if maximum == data.get("max_take_len"):
        generator = generators["train"]
    else:
        generator = build_batch_generator(
            "ProbeTrain",
            manifests["train"],
            data_cfg=data,
            paths_cfg=config["paths"],
            loader_cfg=config.get("loader", {}),
            max_take_len=maximum,
            split_role="train",
        )
    evaluation = config.get("evaluation", {})
    subset = maybe_apply_train_subset(
        generator,
        evaluation.get("train_subset_fraction"),
        evaluation.get("ensure_class_coverage", True),
        int(evaluation.get("train_subset_seed", config["runtime"]["seed"])),
    )
    return generator, subset


def train_features(trainer, config, generators, probe_generator, device, output):
    train, probe, ssl = config["train"], config["probe"], config["ssl"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trainer.release_config = config
    trainer.train(
        str(output),
        generators["train"],
        num_epochs=int(train["num_epochs"]),
        batch_size=int(train["batch_size"]),
        learning_rate=float(train["lr"]),
        device=device,
        val_batch_gen=generators["val"],
        val_batch_size=int(train.get("val_batch_size", 1)),
        ckpt_metric=train.get("ckpt_metric", "lin_combined"),
        lr_milestones=train.get("lr_milestones"),
        lr_gamma=float(train.get("lr_gamma", 0.1)),
        probe_every=int(probe.get("every", 20)),
        probe_epochs=int(probe["epochs"]),
        probe_batch_size=int(probe.get("batch_size", train["batch_size"])),
        probe_val_batch_size=int(probe.get("val_batch_size", 1)),
        probe_lr_lin=float(probe.get("lr_lin", 0.001)),
        probe_lr_ltcontext=float(probe.get("lr_ltcontext", 0.00025)),
        probe_weight_decay=float(probe.get("weight_decay", 0)),
        lin_lr_milestones=probe.get("lin_lr_milestones"),
        ltcontext_lr_milestones=probe.get("ltcontext_lr_milestones"),
        ltcontext_cfg_overrides=probe.get("ltcontext"),
        val_every=int(train.get("val_every", 20)),
        probe_mode=probe.get("mode", "linear"),
        probe_batch_gen=probe_generator,
        ssl_accumulation_steps=int(ssl.get("accumulation_steps", 1)),
        linear_probe_pool=bool(probe.get("linear_probe_pool", False)),
        background_label_id=int(probe.get("background_label_id", -100)),
        linear_probe_head_mode=probe.get("linear_head_mode", "single"),
        linear_probe_activity_loss_weight=float(probe.get("activity_loss_weight", 1)),
        linear_probe_foreground_loss_weight=float(probe.get("foreground_loss_weight", 1)),
    )
