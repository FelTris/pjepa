"""Evaluate saved temporal and linear heads without fitting."""

from typing import Dict, Optional
import os
import torch
from pjepa.utils.segmentation_metrics import evaluate_sequences
from pjepa.evaluation.selection import _ltcontext_combined_metric


class SavedProbeMixin:
    def evaluate_loaded_ltcontext_probes(
        self,
        val_batch_gen,
        val_batch_size: int,
        device,
        temporal_probe_path: str,
        linear_probe_path: str,
        temporal_probe_kind: str = "causal_ltcontext",
        probe_feature_source: str = "student",
        probe_lr_ltcontext: float = 0.00025,
        probe_lr_lin: float = 0.001,
        probe_weight_decay: float = 0.0,
        ltcontext_cfg_overrides: Optional[Dict[str, object]] = None,
        ltcontext_solver_cfg: Optional[Dict[str, object]] = None,
        linear_probe_pool: bool = True,
        temporal_head_mode: str = "frame",
        temporal_clip_pool: str = "max",
        background_label_id: int = -100,
        linear_probe_head_mode: str = "single",
        linear_probe_activity_loss_weight: float = 1.0,
        linear_probe_foreground_loss_weight: float = 1.0,
    ):
        temporal_probe_kind = str(temporal_probe_kind).lower()
        if temporal_probe_kind not in {"ltcontext", "causal_ltcontext"}:
            raise ValueError(
                f"Loaded-probe inference currently supports only temporal_probe_kind 'ltcontext' or 'causal_ltcontext', got '{temporal_probe_kind}'."
            )
        temporal_head_mode = str(temporal_head_mode).lower()
        if temporal_head_mode not in {"frame", "clip_pooled"}:
            raise ValueError("temporal_head_mode must be one of {'frame', 'clip_pooled'}.")
        if str(temporal_clip_pool).lower() != "max":
            raise ValueError("Only temporal_clip_pool='max' is currently supported.")
        temporal_clip_pooled = temporal_head_mode == "clip_pooled"
        temporal_probe_path = str(temporal_probe_path)
        linear_probe_path = str(linear_probe_path)
        if not os.path.isfile(temporal_probe_path):
            raise FileNotFoundError(f"Temporal probe checkpoint not found: {temporal_probe_path}")
        if not os.path.isfile(linear_probe_path):
            raise FileNotFoundError(f"Linear probe checkpoint not found: {linear_probe_path}")
        was_train_student = self.model.student_enc.training
        self.model.student_enc.eval()
        for p in self.model.student_enc.parameters():
            p.requires_grad_(False)
        probe_input_dim = (
            self.input_dim if str(probe_feature_source).startswith("raw") else self.dim
        )
        self._init_LTContext_probe_if_needed(
            self.num_classes,
            device,
            probe_lr_ltcontext,
            probe_weight_decay,
            cfg_overrides=ltcontext_cfg_overrides,
            solver_cfg=ltcontext_solver_cfg,
            causal=temporal_probe_kind == "causal_ltcontext",
            in_dim=probe_input_dim,
        )
        self._init_linear_probe_if_needed(
            self.num_classes,
            device,
            probe_lr_lin,
            probe_weight_decay,
            head_mode=linear_probe_head_mode,
            background_label_id=background_label_id,
            in_dim=probe_input_dim,
        )
        try:
            self.ltcontext_probe.load_state_dict(
                torch.load(temporal_probe_path, map_location=device), strict=True
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to strictly load temporal probe checkpoint '{temporal_probe_path}'."
            ) from exc
        try:
            self.lin_probe.load_state_dict(
                torch.load(linear_probe_path, map_location=device), strict=True
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to strictly load linear probe checkpoint '{linear_probe_path}'."
            ) from exc
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
        temporal_predictions = {}
        temporal_ground_truth = {}
        temporal_frame_predictions = []
        linear_predictions = {}
        linear_ground_truth = {}
        linear_frame_predictions = []
        ltcontext_seq_predictions = {}
        ltcontext_seq_ground_truth = {}
        lin_seq_predictions = {}
        lin_seq_ground_truth = {}
        ltcontext_seq_index = 0
        lin_seq_index = 0

        def append_frame_rows(
            rows,
            pred_flat,
            target_flat,
            valid_mask_flat,
            normalized_batch,
            mask_3d_batch,
            batch_index,
        ):
            pred_cpu = pred_flat.detach().cpu()
            target_cpu = target_flat.detach().cpu()
            valid_cpu = valid_mask_flat.detach().cpu().bool()
            if normalized_batch["flat_mode"]:
                segment_lengths_cpu = normalized_batch["segment_lengths"]
                for b in range(pred_cpu.shape[0]):
                    base = (
                        normalized_batch["names"][b]
                        if normalized_batch["names"] is not None
                        and len(normalized_batch["names"]) > b
                        else []
                    )
                    offset = 0
                    for s, seg_len in enumerate(segment_lengths_cpu[b].tolist()):
                        seg_len = int(seg_len)
                        if seg_len <= 0 or offset >= pred_cpu.shape[1]:
                            break
                        seg_len = min(seg_len, pred_cpu.shape[1] - offset)
                        for i in range(seg_len):
                            token_index = offset + i
                            if not bool(valid_cpu[b, token_index].item()):
                                continue
                            sample_id = (
                                base[s]
                                if s < len(base)
                                else f"batch{batch_index}_item{b}_segment{s}"
                            )
                            pred_label = int(pred_cpu[b, token_index].item())
                            gt_label = int(target_cpu[b, token_index].item())
                            rows.append(
                                {
                                    "sample_id": sample_id,
                                    "batch_index": int(batch_index),
                                    "item_index": int(b),
                                    "segment_index": int(s),
                                    "frame_index": int(token_index),
                                    "segment_frame_index": int(i),
                                    "predicted_class": pred_label,
                                    "gt_class": gt_label,
                                    "correct": int(pred_label == gt_label),
                                }
                            )
                        offset += seg_len
                return
            (B, N, L) = mask_3d_batch.shape
            pred_3d = pred_cpu.view(B, N, L)
            target_3d = target_cpu.view(B, N, L)
            valid_3d = mask_3d_batch.detach().cpu().bool()
            for b in range(B):
                base = (
                    normalized_batch["names"][b]
                    if normalized_batch["names"] is not None and len(normalized_batch["names"]) > b
                    else []
                )
                for n in range(N):
                    sample_id = base[n] if n < len(base) else f"batch{batch_index}_item{b}_clip{n}"
                    for l in range(L):
                        if not bool(valid_3d[b, n, l].item()):
                            continue
                        pred_label = int(pred_3d[b, n, l].item())
                        gt_label = int(target_3d[b, n, l].item())
                        rows.append(
                            {
                                "sample_id": sample_id,
                                "batch_index": int(batch_index),
                                "item_index": int(b),
                                "segment_index": int(n),
                                "frame_index": int(n * L + l),
                                "segment_frame_index": int(l),
                                "predicted_class": pred_label,
                                "gt_class": gt_label,
                                "correct": int(pred_label == gt_label),
                            }
                        )

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
                append_frame_rows(
                    temporal_frame_predictions, pred_temporal, y, mask_f, normalized, mask_3d, steps
                )
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
                    temporal_predictions.update(
                        self._clip_predictions_to_dict(
                            pred_clip_temporal, normalized, clip_mask_temporal
                        )
                    )
                    temporal_ground_truth.update(
                        self._clip_predictions_to_dict(
                            clip_labels_temporal, normalized, clip_mask_temporal
                        )
                    )
                correct_temporal += ((pred_temporal == y).float() * mask_f).sum().item()
                total_temporal += torch.sum(mask_f).item()
                non_bg_mask = mask_f.bool() & (y != int(background_label_id))
                correct_temporal_non_bg += int(((pred_temporal == y) & non_bg_mask).sum().item())
                total_temporal_non_bg += int(non_bg_mask.sum().item())
                if normalized["flat_mode"]:
                    pred_temporal_cpu = pred_temporal.detach().cpu()
                    y_cpu = y.detach().cpu()
                    mask_f_cpu = mask_f.detach().cpu()
                    for b in range(pred_temporal_cpu.shape[0]):
                        base = (
                            normalized["names"][b]
                            if normalized["names"] is not None and len(normalized["names"]) > b
                            else []
                        )
                        offset = 0
                        for s, seg_len in enumerate(segment_lengths[b].tolist()):
                            seg_len = int(seg_len)
                            if seg_len <= 0 or offset >= pred_temporal_cpu.shape[1]:
                                break
                            seg_len = min(seg_len, pred_temporal_cpu.shape[1] - offset)
                            seg_valid = mask_f_cpu[b, offset : offset + seg_len]
                            if seg_valid.any():
                                pred_seg = pred_temporal_cpu[b, offset : offset + seg_len][
                                    seg_valid
                                ]
                                tgt_seg = y_cpu[b, offset : offset + seg_len][seg_valid]
                                pred_label = int(torch.bincount(pred_seg).argmax().item())
                                true_label = int(torch.bincount(tgt_seg).argmax().item())
                                if s < len(base):
                                    sample_id = base[s]
                                    temporal_predictions[sample_id] = pred_label
                                    temporal_ground_truth[sample_id] = true_label
                                clip_correct_temporal += int(pred_label == true_label)
                                clip_total_temporal += 1
                            offset += seg_len
                else:
                    pred_last_3d = pred_temporal.view(B, N, L)
                    y_3d = y.view(B, N, L)
                    for b in range(B):
                        base = (
                            normalized["names"][b]
                            if normalized["names"] is not None and len(normalized["names"]) > b
                            else []
                        )
                        for n in range(N):
                            valid_vec = mask_3d[b, n]
                            if not valid_vec.any():
                                continue
                            p_clip = pred_last_3d[b, n][valid_vec].detach().cpu()
                            t_clip = y_3d[b, n][valid_vec].detach().cpu()
                            pred_label = int(torch.bincount(p_clip).argmax().item())
                            true_label = int(torch.bincount(t_clip).argmax().item())
                            if not temporal_clip_pooled and n < len(base):
                                sample_id = base[n]
                                temporal_predictions[sample_id] = pred_label
                                temporal_ground_truth[sample_id] = true_label
                            clip_correct_temporal += int(pred_label == true_label)
                            clip_total_temporal += 1
                for b in range(B):
                    valid_flat = mask_f[b].bool()
                    if valid_flat.any():
                        seq_key = f"take_{ltcontext_seq_index}"
                        ltcontext_seq_index += 1
                        ltcontext_seq_predictions[seq_key] = (
                            pred_temporal[b][valid_flat].detach().cpu().numpy()
                        )
                        ltcontext_seq_ground_truth[seq_key] = (
                            y[b][valid_flat].detach().cpu().numpy()
                        )
                logits_token = self.lin_probe(t_full)
                if isinstance(logits_token, dict):
                    pred_lin_frame = self._foreground_activity_predictions(
                        logits_token["foreground"], logits_token["activity"]
                    )
                else:
                    pred_lin_frame = logits_token.argmax(dim=-1)
                append_frame_rows(
                    linear_frame_predictions, pred_lin_frame, y, mask_f, normalized, mask_3d, steps
                )
                (loss_lin, pred_lin, y_single_class, mask_single_class, _) = (
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
                correct_lin += (
                    ((pred_lin == y_single_class).float() * mask_single_class).sum().item()
                )
                total_lin += torch.sum(mask_single_class).item()
                lin_non_bg_mask = mask_single_class.bool() & (
                    y_single_class != int(background_label_id)
                )
                correct_lin_non_bg += int(
                    ((pred_lin == y_single_class) & lin_non_bg_mask).sum().item()
                )
                total_lin_non_bg += int(lin_non_bg_mask.sum().item())
                if linear_probe_pool:
                    clip_correct_lin += (
                        ((pred_lin == y_single_class).float() * mask_single_class).sum().item()
                    )
                    clip_total_lin += torch.sum(mask_single_class).item()
                    linear_predictions.update(
                        self._clip_predictions_to_dict(pred_lin, normalized, mask_single_class)
                    )
                    linear_ground_truth.update(
                        self._clip_predictions_to_dict(
                            y_single_class, normalized, mask_single_class
                        )
                    )
                else:
                    if normalized["flat_mode"]:
                        pred_lin_cpu = pred_lin.detach().cpu()
                        target_flat_cpu = normalized["target_flat"].detach().cpu()
                        valid_mask_flat_cpu = normalized["valid_mask_flat"].detach().cpu()
                        for b in range(pred_lin_cpu.shape[0]):
                            base = (
                                normalized["names"][b]
                                if normalized["names"] is not None and len(normalized["names"]) > b
                                else []
                            )
                            offset = 0
                            for s, seg_len in enumerate(normalized["segment_lengths"][b].tolist()):
                                seg_len = int(seg_len)
                                if seg_len <= 0 or offset >= pred_lin_cpu.shape[1]:
                                    break
                                seg_len = min(seg_len, pred_lin_cpu.shape[1] - offset)
                                seg_valid = valid_mask_flat_cpu[b, offset : offset + seg_len]
                                if seg_valid.any():
                                    pred_seg = pred_lin_cpu[b, offset : offset + seg_len][seg_valid]
                                    tgt_seg = target_flat_cpu[b, offset : offset + seg_len][
                                        seg_valid
                                    ]
                                    pred_label = int(torch.bincount(pred_seg).argmax().item())
                                    true_label = int(torch.bincount(tgt_seg).argmax().item())
                                    if s < len(base):
                                        sample_id = base[s]
                                        linear_predictions[sample_id] = pred_label
                                        linear_ground_truth[sample_id] = true_label
                                    clip_correct_lin += int(pred_label == true_label)
                                    clip_total_lin += 1
                                offset += seg_len
                    else:
                        target_3d = normalized["target_3d"]
                        for b in range(B):
                            base = (
                                normalized["names"][b]
                                if normalized["names"] is not None and len(normalized["names"]) > b
                                else []
                            )
                            for n in range(N):
                                valid_vec = mask_3d[b, n]
                                if not valid_vec.any():
                                    continue
                                if pred_lin.ndim == 2:
                                    pred_label = int(pred_lin[b, n].item())
                                    true_label = int(
                                        torch.bincount(target_3d[b, n][valid_vec].detach().cpu())
                                        .argmax()
                                        .item()
                                    )
                                else:
                                    p_clip = pred_lin[b, n][valid_vec].detach().cpu()
                                    t_clip = target_3d[b, n][valid_vec].detach().cpu()
                                    pred_label = int(torch.bincount(p_clip).argmax().item())
                                    true_label = int(torch.bincount(t_clip).argmax().item())
                                if n < len(base):
                                    sample_id = base[n]
                                    linear_predictions[sample_id] = pred_label
                                    linear_ground_truth[sample_id] = true_label
                                clip_correct_lin += int(pred_label == true_label)
                                clip_total_lin += 1
                    for pred_seq, gt_seq in self._dense_linear_sequences(
                        pred_lin, normalized, mask_3d, linear_probe_pool=linear_probe_pool
                    ):
                        seq_key = f"take_{lin_seq_index}"
                        lin_seq_index += 1
                        lin_seq_predictions[seq_key] = pred_seq
                        lin_seq_ground_truth[seq_key] = gt_seq
                steps += 1
        val_batch_gen.reset()
        self.model.student_enc.train(was_train_student)
        for p in self.model.student_enc.parameters():
            p.requires_grad_(True)
        val_loss_lin = loss_sum_lin / max(1, steps)
        frame_acc_lin = correct_lin / total_lin if total_lin > 0 else 0.0
        frame_acc_lin_non_bg = (
            correct_lin_non_bg / total_lin_non_bg if total_lin_non_bg > 0 else 0.0
        )
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
        temporal_seg_metrics = evaluate_sequences(
            predictions=ltcontext_seq_predictions,
            ground_truth=ltcontext_seq_ground_truth,
            label_names={},
            bg_label_id=int(background_label_id),
            overlaps=[0.1, 0.25, 0.5],
        )
        result = {
            "epoch": 0,
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
            "ltcontext_loss": float(val_loss_temporal),
            "loss": float(val_loss_temporal),
            "frame_acc": float(frame_acc_temporal),
            "ltcontext_frame_acc": float(frame_acc_temporal),
            "ltcontext_clip_acc": float(clip_majority_temporal),
            "ltcontext_clip_pooled_acc": float(clip_pooled_temporal),
            "ltcontext_frame_acc_no_bg": float(frame_acc_temporal_non_bg),
            "frame_acc_no_bg": float(frame_acc_temporal_non_bg),
            "clip_majority_acc": float(clip_majority_temporal),
            "ltcontext_edit_score": float(temporal_seg_metrics["edit_score"]),
            "ltcontext_f1_10": float(temporal_seg_metrics["f1_overlap"].get("0.1", 0.0)),
            "ltcontext_f1_25": float(temporal_seg_metrics["f1_overlap"].get("0.25", 0.0)),
            "ltcontext_f1_50": float(temporal_seg_metrics["f1_overlap"].get("0.5", 0.0)),
            "temporal_predictions": temporal_predictions,
            "temporal_ground_truth": temporal_ground_truth,
            "temporal_frame_predictions": temporal_frame_predictions,
            "linear_predictions": linear_predictions,
            "linear_ground_truth": linear_ground_truth,
            "linear_frame_predictions": linear_frame_predictions,
            "temporal_probe_path": temporal_probe_path,
            "linear_probe_path": linear_probe_path,
        }
        result["lin_combined"] = _ltcontext_combined_metric(
            result["lin_frame_acc"],
            result["lin_edit_score"],
            result["lin_f1_10"],
            result["lin_f1_25"],
            result["lin_f1_50"],
            accuracy_no_bg=result["lin_frame_acc_no_bg"],
        )
        result["ltcontext_combined"] = _ltcontext_combined_metric(
            result["ltcontext_frame_acc"],
            result["ltcontext_edit_score"],
            result["ltcontext_f1_10"],
            result["ltcontext_f1_25"],
            result["ltcontext_f1_50"],
            accuracy_no_bg=result["ltcontext_frame_acc_no_bg"],
        )
        return result
