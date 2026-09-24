"""Portable encoder snapshots for inference and training warm starts."""

from pathlib import Path
import torch


def model_architecture(model):
    encoder = model.student_enc
    return dict(
        d_in=encoder.d_in,
        d_model=encoder.d_model,
        enc_depth=len(encoder.blocks),
        enc_heads=encoder.n_heads,
        pred_depth=len(model.predictor.blocks),
        pred_heads=model.predictor.n_heads,
        drop_path_prob=0.0,
        student_encoder_attention=model.student_encoder_attention,
        student_encoder_block_size=model.student_encoder_block_size,
        rope_mode=model.rope_mode,
    )


def save_snapshot(path, trainer, *, epoch, scheduler=None, selection=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "pjepa_feature_v1",
        "model_state_dict": trainer.model.state_dict(),
        "architecture": model_architecture(trainer.model),
        "epoch": epoch,
        "teacher_state_dict": trainer.teacher.state_dict(),
        "optimizer_state_dict": trainer.opt.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "selection": selection,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
