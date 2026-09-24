"""Evaluate frozen-feature heads for one probe epoch; no fitting or selection."""

import torch
from pjepa.evaluation.selection import _ltcontext_combined_metric
from pjepa.utils.segmentation_metrics import evaluate_sequences
from pjepa.utils.logging import logger


def evaluate_probe_epoch(
    self,
    val_batch_gen,
    val_batch_size,
    device,
    temporal_probe_kind,
    temporal_clip_pooled,
    probe_feature_source,
    linear_probe_pool,
    background_label_id,
    linear_probe_activity_loss_weight,
    linear_probe_foreground_loss_weight,
    pe,
    probe_epochs,
    temporal_head_label,
    temporal_loss_key,
):
    if self.ltcontext_probe is not None:
        self.ltcontext_probe.eval()
    self.lin_probe.eval()
    total_temporal = total_lin = 0
    correct_temporal = correct_lin = 0
    total_temporal_non_bg = total_lin_non_bg = 0
    correct_temporal_non_bg = correct_lin_non_bg = 0
    clip_correct_temporal = clip_total_temporal = 0
    clip_pooled_correct_temporal = clip_pooled_total_temporal = 0
    clip_correct_lin = clip_total_lin = 0
    loss_sum_temporal = loss_sum_lin = 0.0
    steps = 0
    val_dict = {}
    ltcontext_seq_predictions = {}
    ltcontext_seq_ground_truth = {}
    ltcontext_seq_index = 0
    lin_seq_predictions = {}
    lin_seq_ground_truth = {}
    lin_seq_index = 0
    val_batch_gen.reset()
    with torch.no_grad():
        while val_batch_gen.has_next():
            normalized = self._normalize_loader_batch(val_batch_gen.next_batch(val_batch_size))
            x = normalized["x4d"].to(device)
            y = normalized["target_flat"].to(device)
            mask_3d = normalized["valid_mask_3d"].to(device)
            mask_f = normalized["valid_mask_flat"].to(device)
            segment_lengths = normalized["segment_lengths"]
            segment_lengths_dev = (
                segment_lengths.to(device) if segment_lengths is not None else None
            )
            (B, N, L, _) = x.shape
            t_full = self._probe_features(
                x,
                mask_3d,
                probe_feature_source=probe_feature_source,
                segment_lengths=segment_lengths_dev,
            )
            if (
                temporal_probe_kind in {"ltcontext", "causal_ltcontext"}
                and self.ltcontext_probe is not None
            ):
                predictions = (
                    self.ltcontext_probe(t_full, mask_f, segment_lengths_dev)
                    if temporal_probe_kind == "causal_ltcontext"
                    else self.ltcontext_probe(t_full, mask_f)
                )
                if temporal_clip_pooled:
                    loss_temporal = 0.0
                    for stage_logits in predictions:
                        (clip_logits, clip_labels, _) = self._prepare_clip_pooled_logits(
                            stage_logits.transpose(2, 1).contiguous(), normalized, mask_3d, device
                        )
                        loss_temporal = loss_temporal + self.ce(
                            clip_logits.contiguous().view(-1, self.num_classes),
                            clip_labels.view(-1),
                        )
                else:
                    loss_temporal = self.ltcontext_probe.loss(predictions, y)
                loss_sum_temporal += float(loss_temporal.item())
                pred_temporal = predictions[-1].argmax(dim=1)
                (clip_logits_temporal, clip_labels_temporal, clip_mask_temporal) = (
                    self._prepare_clip_pooled_logits(
                        predictions[-1].transpose(2, 1).contiguous(), normalized, mask_3d, device
                    )
                )
                pred_clip_temporal = clip_logits_temporal.argmax(dim=-1)
                clip_pooled_correct_temporal += (
                    ((pred_clip_temporal == clip_labels_temporal).float() * clip_mask_temporal)
                    .sum()
                    .item()
                )
                clip_pooled_total_temporal += torch.sum(clip_mask_temporal).item()
                if temporal_clip_pooled:
                    val_dict.update(
                        self._clip_predictions_to_dict(
                            pred_clip_temporal, normalized, clip_mask_temporal
                        )
                    )
                correct_temporal += ((pred_temporal == y).float() * mask_f).sum().item()
                total_temporal += torch.sum(mask_f).item()
                non_bg_mask = mask_f.bool() & (y != int(background_label_id))
                correct_temporal_non_bg += int(((pred_temporal == y) & non_bg_mask).sum().item())
                total_temporal_non_bg += int(non_bg_mask.sum().item())
                if normalized["flat_mode"]:
                    (correct_inc, total_inc, val_inc) = self._segment_majority_vote(
                        pred_temporal.detach().cpu(),
                        y.detach().cpu(),
                        mask_f.detach().cpu(),
                        segment_lengths,
                        normalized["names"],
                    )
                    clip_correct_temporal += correct_inc
                    clip_total_temporal += total_inc
                    val_dict.update(val_inc)
                else:
                    pred_last_3d = pred_temporal.view(B, N, L)
                    y_3d = y.view(B, N, L)
                    m_3d = mask_3d
                    for b in range(B):
                        base = (
                            normalized["names"][b]
                            if normalized["names"] is not None and len(normalized["names"]) > b
                            else []
                        )
                        for n in range(N):
                            valid_vec = m_3d[b, n]
                            if not valid_vec.any():
                                continue
                            p_clip = pred_last_3d[b, n][valid_vec].detach().cpu()
                            t_clip = y_3d[b, n][valid_vec].detach().cpu()
                            pred_label = int(torch.bincount(p_clip).argmax().item())
                            true_label = int(torch.bincount(t_clip).argmax().item())
                            if not temporal_clip_pooled and n < len(base):
                                val_dict[base[n]] = pred_label
                            clip_correct_temporal += int(pred_label == true_label)
                            clip_total_temporal += 1
                for b in range(B):
                    valid_flat = mask_f[b].bool()
                    if not valid_flat.any():
                        continue
                    seq_key = f"take_{ltcontext_seq_index}"
                    ltcontext_seq_index += 1
                    ltcontext_seq_predictions[seq_key] = (
                        pred_temporal[b][valid_flat].detach().cpu().numpy()
                    )
                    ltcontext_seq_ground_truth[seq_key] = y[b][valid_flat].detach().cpu().numpy()
            logits_token = self.lin_probe(t_full)
            (loss_lin, pred_lin, y_single_class, mask_single_class, lin_aux) = (
                self._linear_probe_loss_and_predictions(
                    logits_token,
                    normalized,
                    mask_3d,
                    device,
                    linear_probe_pool=linear_probe_pool,
                    activity_loss_weight=linear_probe_activity_loss_weight,
                    foreground_loss_weight=linear_probe_foreground_loss_weight,
                )
            )
            loss_sum_lin += loss_lin.item()
            correct_lin += ((pred_lin == y_single_class).float() * mask_single_class).sum().item()
            total_lin += torch.sum(mask_single_class).item()
            lin_non_bg_mask = mask_single_class.bool() & (
                y_single_class != int(background_label_id)
            )
            correct_lin_non_bg += int(((pred_lin == y_single_class) & lin_non_bg_mask).sum().item())
            total_lin_non_bg += int(lin_non_bg_mask.sum().item())
            if linear_probe_pool:
                clip_correct_inc = (
                    ((pred_lin == y_single_class).float() * mask_single_class).sum().item()
                )
                clip_total_inc = torch.sum(mask_single_class).item()
            else:
                (clip_correct_inc, clip_total_inc, _) = self._clip_majority_from_linear_predictions(
                    pred_lin, normalized, mask_3d
                )
            clip_correct_lin += clip_correct_inc
            clip_total_lin += clip_total_inc
            if not linear_probe_pool:
                for pred_seq, gt_seq in self._dense_linear_sequences(
                    pred_lin, normalized, mask_3d, linear_probe_pool=linear_probe_pool
                ):
                    seq_key = f"take_{lin_seq_index}"
                    lin_seq_index += 1
                    lin_seq_predictions[seq_key] = pred_seq
                    lin_seq_ground_truth[seq_key] = gt_seq
            steps += 1
    val_loss_lin = loss_sum_lin / max(1, steps)
    frame_acc_lin = correct_lin / total_lin if total_lin > 0 else 0.0
    frame_acc_lin_non_bg = correct_lin_non_bg / total_lin_non_bg if total_lin_non_bg > 0 else 0.0
    clip_acc_lin = clip_correct_lin / clip_total_lin if clip_total_lin > 0 else 0.0
    linear_seg_metrics = (
        None
        if linear_probe_pool
        else evaluate_sequences(
            predictions=lin_seq_predictions,
            ground_truth=lin_seq_ground_truth,
            label_names={},
            bg_label_id=int(background_label_id),
            overlaps=[0.1, 0.25, 0.5],
        )
    )
    logger.info(
        f"[probe][val][Linear] epoch={pe}/{probe_epochs} frame_acc={frame_acc_lin * 100:.2f}% frame_acc_no_bg={frame_acc_lin_non_bg * 100:.2f}% clip_acc={clip_acc_lin * 100:.2f}% loss={val_loss_lin:.4f}"
    )
    if linear_seg_metrics is not None:
        logger.info(
            f"[probe][val][Linear] epoch={pe}/{probe_epochs} Edit=%.4f F1@0.10=%.4f F1@0.25=%.4f F1@0.50=%.4f"
            % (
                float(linear_seg_metrics["edit_score"]),
                float(linear_seg_metrics["f1_overlap"].get("0.1", 0.0)),
                float(linear_seg_metrics["f1_overlap"].get("0.25", 0.0)),
                float(linear_seg_metrics["f1_overlap"].get("0.5", 0.0)),
            )
        )
    val_stats = {
        "lin_loss": float(val_loss_lin),
        "lin_clip_acc": float(clip_acc_lin),
        "lin_frame_acc": float(frame_acc_lin),
        "lin_frame_acc_no_bg": float(frame_acc_lin_non_bg),
        "lin_clip_majority_acc": float(clip_acc_lin),
        "lin_edit_score": float(linear_seg_metrics["edit_score"])
        if linear_seg_metrics is not None
        else float("nan"),
        "lin_f1_10": float(linear_seg_metrics["f1_overlap"].get("0.1", 0.0))
        if linear_seg_metrics is not None
        else float("nan"),
        "lin_f1_25": float(linear_seg_metrics["f1_overlap"].get("0.25", 0.0))
        if linear_seg_metrics is not None
        else float("nan"),
        "lin_f1_50": float(linear_seg_metrics["f1_overlap"].get("0.5", 0.0))
        if linear_seg_metrics is not None
        else float("nan"),
        "linear_probe_pool": bool(linear_probe_pool),
        "epoch": pe,
    }
    val_stats["lin_combined"] = _ltcontext_combined_metric(
        val_stats["lin_frame_acc"],
        val_stats["lin_edit_score"],
        val_stats["lin_f1_10"],
        val_stats["lin_f1_25"],
        val_stats["lin_f1_50"],
        accuracy_no_bg=val_stats["lin_frame_acc_no_bg"],
    )
    metric_values = {
        "lin_clip_acc": clip_acc_lin,
        "lin_frame_acc": frame_acc_lin,
        "lin_frame_acc_no_bg": frame_acc_lin_non_bg,
        "lin_combined": val_stats["lin_combined"],
        "lin_loss": val_loss_lin,
    }
    if temporal_probe_kind is not None:
        val_loss_temporal = loss_sum_temporal / max(1, steps)
        frame_acc_temporal = correct_temporal / total_temporal if total_temporal > 0 else 0.0
        frame_acc_temporal_non_bg = (
            correct_temporal_non_bg / total_temporal_non_bg if total_temporal_non_bg > 0 else 0.0
        )
        clip_majority_temporal = (
            clip_correct_temporal / clip_total_temporal if clip_total_temporal > 0 else 0.0
        )
        clip_pooled_temporal = (
            clip_pooled_correct_temporal / clip_pooled_total_temporal
            if clip_pooled_total_temporal > 0
            else clip_majority_temporal
        )
        temporal_seg_metrics = None
        if temporal_probe_kind in {"ltcontext", "causal_ltcontext"}:
            temporal_seg_metrics = evaluate_sequences(
                predictions=ltcontext_seq_predictions,
                ground_truth=ltcontext_seq_ground_truth,
                label_names={},
                bg_label_id=int(background_label_id),
                overlaps=[0.1, 0.25, 0.5],
            )
        logger.info(
            f"[probe][val][{temporal_head_label}] epoch={pe}/{probe_epochs} frame_acc={frame_acc_temporal * 100:.2f}% frame_acc_no_bg={frame_acc_temporal_non_bg * 100:.2f}% clip_majority={clip_majority_temporal * 100:.2f}% clip_pooled={clip_pooled_temporal * 100:.2f}% loss={val_loss_temporal:.4f}"
        )
        if temporal_seg_metrics is not None:
            metric_tag = (
                "CausalLTContext" if temporal_probe_kind == "causal_ltcontext" else "LTContext"
            )
            logger.info(
                f"[probe][val][{metric_tag}] epoch={pe}/{probe_epochs} Edit=%.4f F1@0.10=%.4f F1@0.25=%.4f F1@0.50=%.4f"
                % (
                    float(temporal_seg_metrics["edit_score"]),
                    float(temporal_seg_metrics["f1_overlap"].get("0.1", 0.0)),
                    float(temporal_seg_metrics["f1_overlap"].get("0.25", 0.0)),
                    float(temporal_seg_metrics["f1_overlap"].get("0.5", 0.0)),
                )
            )
        temporal_metric_prefix = "ltcontext"
        val_stats.update(
            {
                temporal_loss_key: float(val_loss_temporal),
                "loss": float(val_loss_temporal),
                "frame_acc": float(frame_acc_temporal),
                f"{temporal_metric_prefix}_frame_acc": float(frame_acc_temporal),
                f"{temporal_metric_prefix}_clip_acc": float(clip_majority_temporal),
                f"{temporal_metric_prefix}_clip_pooled_acc": float(clip_pooled_temporal),
                f"{temporal_metric_prefix}_frame_acc_no_bg": float(frame_acc_temporal_non_bg),
                "frame_acc_no_bg": float(frame_acc_temporal_non_bg),
                "clip_majority_acc": float(clip_majority_temporal),
                "val_dict": val_dict,
            }
        )
        if temporal_probe_kind in {"ltcontext", "causal_ltcontext"}:
            val_stats["ltcontext_clip_acc"] = float(clip_pooled_temporal)
        if temporal_seg_metrics is not None:
            metric_prefix = "ltcontext"
            val_stats.update(
                {
                    f"{metric_prefix}_edit_score": float(temporal_seg_metrics["edit_score"]),
                    f"{metric_prefix}_f1_10": float(
                        temporal_seg_metrics["f1_overlap"].get("0.1", 0.0)
                    ),
                    f"{metric_prefix}_f1_25": float(
                        temporal_seg_metrics["f1_overlap"].get("0.25", 0.0)
                    ),
                    f"{metric_prefix}_f1_50": float(
                        temporal_seg_metrics["f1_overlap"].get("0.5", 0.0)
                    ),
                }
            )
        if temporal_probe_kind in {"ltcontext", "causal_ltcontext"}:
            val_stats["ltcontext_combined"] = _ltcontext_combined_metric(
                val_stats["ltcontext_frame_acc"],
                val_stats.get("ltcontext_edit_score", float("nan")),
                val_stats.get("ltcontext_f1_10", float("nan")),
                val_stats.get("ltcontext_f1_25", float("nan")),
                val_stats.get("ltcontext_f1_50", float("nan")),
                accuracy_no_bg=val_stats["ltcontext_frame_acc_no_bg"],
            )
        metric_values.update(
            {
                f"{temporal_metric_prefix}_frame_acc": frame_acc_temporal,
                f"{temporal_metric_prefix}_frame_acc_no_bg": frame_acc_temporal_non_bg,
                "clip_majority_acc": clip_majority_temporal,
                temporal_loss_key: val_loss_temporal,
            }
        )
        if temporal_probe_kind in {"ltcontext", "causal_ltcontext"}:
            metric_values["ltcontext_clip_acc"] = val_stats["ltcontext_clip_acc"]
        if temporal_probe_kind in {"ltcontext", "causal_ltcontext"}:
            metric_values["ltcontext_combined"] = val_stats["ltcontext_combined"]
    else:
        val_stats.update(
            {
                "loss": float(val_loss_lin),
                "frame_acc": float(frame_acc_lin),
                "frame_acc_no_bg": float(frame_acc_lin_non_bg),
                "clip_majority_acc": float(frame_acc_lin),
                "val_dict": {},
            }
        )
    return (val_stats, metric_values)
