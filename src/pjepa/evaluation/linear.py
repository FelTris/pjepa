"""Score a saved frame-feature linear probe without retraining it."""

import torch
from pjepa.utils.segmentation_metrics import evaluate_sequences
from pjepa.evaluation.selection import _ltcontext_combined_metric


@torch.inference_mode()
def evaluate_linear(
    trainer,
    generator,
    checkpoint,
    *,
    device,
    batch_size=1,
    probe_feature_source="student",
    background_label_id=-100,
    linear_probe_head_mode="single",
    linear_probe_pool=False,
    **kwargs,
):
    trainer._init_linear_probe_if_needed(
        trainer.num_classes,
        device,
        0.001,
        0,
        head_mode=linear_probe_head_mode,
        background_label_id=background_label_id,
        in_dim=trainer.input_dim if probe_feature_source.startswith("raw") else trainer.dim,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    trainer.lin_probe.load_state_dict(payload.get("state_dict", payload), strict=True)
    trainer.lin_probe.eval()
    trainer.model.eval()
    predictions, labels = {}, {}
    total = correct = total_fg = correct_fg = 0
    losses = []
    generator.reset()
    while generator.has_next():
        batch = trainer._normalize_loader_batch(generator.next_batch(batch_size))
        mask = batch["valid_mask_3d"].to(device)
        lengths = batch["segment_lengths"]
        features = trainer._probe_features(
            batch["x4d"].to(device),
            mask,
            probe_feature_source=probe_feature_source,
            segment_lengths=lengths.to(device) if lengths is not None else None,
        )
        logits = trainer.lin_probe(features)
        loss, pred, targets, valid, _ = trainer._linear_probe_loss_and_predictions(
            logits,
            batch,
            mask,
            device,
            linear_probe_pool=linear_probe_pool,
            activity_loss_weight=kwargs.get("linear_probe_activity_loss_weight", 1),
            foreground_loss_weight=kwargs.get("linear_probe_foreground_loss_weight", 1),
        )
        losses.append(float(loss))
        valid = valid.bool()
        correct += int(((pred == targets) & valid).sum())
        total += int(valid.sum())
        fg = valid & (targets != background_label_id)
        correct_fg += int(((pred == targets) & fg).sum())
        total_fg += int(fg.sum())
        for p, y in trainer._dense_linear_sequences(pred, batch, mask, linear_probe_pool):
            key = str(len(predictions))
            predictions[key] = p
            labels[key] = y
    generator.reset()
    scores = evaluate_sequences(
        predictions=predictions,
        ground_truth=labels,
        label_names={},
        bg_label_id=background_label_id,
        overlaps=[0.1, 0.25, 0.5],
    )
    result = {
        "frame_acc": correct / max(1, total),
        "frame_acc_no_bg": correct_fg / max(1, total_fg),
        "loss": sum(losses) / max(1, len(losses)),
        "edit_score": scores["edit_score"],
        "f1_overlap": scores["f1_overlap"],
        "num_frames": total,
    }
    result["combined"] = _ltcontext_combined_metric(
        result["frame_acc"],
        result["edit_score"],
        *[scores["f1_overlap"][str(v)] for v in [0.1, 0.25, 0.5]],
        accuracy_no_bg=result["frame_acc_no_bg"],
    )
    return result
