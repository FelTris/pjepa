from pjepa.evaluation.probe_epoch import evaluate_probe_epoch

"""Fit frozen-feature linear and optional temporal heads."""
from typing import Dict, Optional
import math
import time
import torch
import torch.nn as nn
from torch import optim
from pjepa.utils.logging import logger


class ProbeFitMixin:
    def run_linear_probe(
        self,
        batch_gen,
        batch_size,
        device,
        val_batch_gen=None,
        val_batch_size=None,
        probe_epochs: int = 1,
        probe_lr_lin: float = 0.001,
        probe_lr_ltcontext: float = 0.00025,
        probe_weight_decay: float = 0.0,
        select_metric: str = "lin_clip_acc",
        lin_lr_milestones=[5, 15],
        ltcontext_lr_milestones=None,
        lr_gamma=0.1,
        ltcontext_cfg_overrides: Optional[Dict[str, object]] = None,
        ltcontext_solver_cfg: Optional[Dict[str, object]] = None,
        linear_only: bool = False,
        temporal_probe_kind: Optional[str] = None,
        linear_probe_pool: bool = True,
        temporal_head_mode: str = "frame",
        temporal_clip_pool: str = "max",
        probe_feature_source: str = "student",
        background_label_id: int = -100,
        linear_probe_head_mode: str = "single",
        linear_probe_activity_loss_weight: float = 1.0,
        linear_probe_foreground_loss_weight: float = 1.0,
        progress_log_every: int = 0,
    ):
        was_train_student = self.model.student_enc.training
        self.model.student_enc.eval()
        for p in self.model.student_enc.parameters():
            p.requires_grad_(False)
        if linear_only:
            temporal_probe_kind = None
        elif temporal_probe_kind is None:
            temporal_probe_kind = "ltcontext"
        temporal_probe_kind = (
            None if temporal_probe_kind is None else str(temporal_probe_kind).lower()
        )
        if temporal_probe_kind not in {None, "ltcontext", "causal_ltcontext"}:
            raise ValueError(
                f"Unsupported temporal_probe_kind='{temporal_probe_kind}'. Use None, 'ltcontext', or 'causal_ltcontext'."
            )
        temporal_head_mode = str(temporal_head_mode).lower()
        if temporal_head_mode not in {"frame", "clip_pooled"}:
            raise ValueError("temporal_head_mode must be one of {'frame', 'clip_pooled'}.")
        temporal_clip_pool = str(temporal_clip_pool).lower()
        if temporal_clip_pool != "max":
            raise ValueError("Only temporal_clip_pool='max' is currently supported.")
        temporal_clip_pooled = temporal_head_mode == "clip_pooled"
        probe_input_dim = (
            self.input_dim if str(probe_feature_source).startswith("raw") else self.dim
        )
        progress_log_every = max(0, int(progress_log_every))
        self.probe_session += 1
        curr_session = self.probe_session
        local_base = self._probe_global_epoch_base
        temporal_results = None
        if temporal_probe_kind in {"ltcontext", "causal_ltcontext"}:
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
        else:
            self.ltcontext_probe = None
            self.ltcontext_probe_opt = None
        self._init_linear_probe_if_needed(
            self.num_classes,
            device,
            probe_lr_lin,
            probe_weight_decay,
            head_mode=linear_probe_head_mode,
            background_label_id=background_label_id,
            in_dim=probe_input_dim,
        )
        scheduler_temporal = scheduler_lin = None
        if lin_lr_milestones:
            ms = sorted((int(m) for m in lin_lr_milestones))
            scheduler_lin = optim.lr_scheduler.MultiStepLR(
                self.lin_probe_opt, milestones=ms, gamma=lr_gamma
            )
        if temporal_probe_kind is not None:
            temporal_milestones = ltcontext_lr_milestones
            if temporal_probe_kind in {"ltcontext", "causal_ltcontext"} and ltcontext_solver_cfg:
                scheduler_temporal = self._construct_ltcontext_scheduler(
                    self.ltcontext_probe_opt,
                    solver_cfg=ltcontext_solver_cfg,
                    fallback_milestones=temporal_milestones,
                    fallback_gamma=lr_gamma,
                )
            elif temporal_milestones:
                ms = sorted((int(m) for m in temporal_milestones))
                temporal_opt = self.ltcontext_probe_opt
                scheduler_temporal = optim.lr_scheduler.MultiStepLR(
                    temporal_opt, milestones=ms, gamma=lr_gamma
                )
        select_metric = {"linacc": "lin_clip_acc"}.get(select_metric, select_metric)
        if temporal_probe_kind in {"ltcontext", "causal_ltcontext"}:
            select_metric = {
                "clip_acc": "ltcontext_clip_acc",
                "frame_acc": "ltcontext_frame_acc",
                "frame_acc_no_bg": "ltcontext_frame_acc_no_bg",
            }.get(select_metric, select_metric)
        best_track = {
            "lin_clip_acc": {"value": float("-inf"), "epoch": -1, "stats": None},
            "lin_frame_acc": {"value": float("-inf"), "epoch": -1, "stats": None},
            "lin_frame_acc_no_bg": {"value": float("-inf"), "epoch": -1, "stats": None},
            "lin_combined": {"value": float("-inf"), "epoch": -1, "stats": None},
            "lin_loss": {"value": float("inf"), "epoch": -1, "stats": None},
        }
        metric_direction = {"lin_loss": "min"}
        if temporal_probe_kind in {"ltcontext", "causal_ltcontext"}:
            best_track.update(
                {
                    "ltcontext_clip_acc": {"value": float("-inf"), "epoch": -1, "stats": None},
                    "ltcontext_frame_acc": {"value": float("-inf"), "epoch": -1, "stats": None},
                    "ltcontext_frame_acc_no_bg": {
                        "value": float("-inf"),
                        "epoch": -1,
                        "stats": None,
                    },
                    "clip_majority_acc": {"value": float("-inf"), "epoch": -1, "stats": None},
                    "ltcontext_combined": {"value": float("-inf"), "epoch": -1, "stats": None},
                    "ltcontext_loss": {"value": float("inf"), "epoch": -1, "stats": None},
                }
            )
            metric_direction["ltcontext_loss"] = "min"
        if temporal_probe_kind in {"ltcontext", "causal_ltcontext"}:
            temporal_head_label = (
                "CausalLTContext" if temporal_probe_kind == "causal_ltcontext" else "LTContext"
            )
            temporal_metric_key = (
                "ltcontext_clip_acc" if temporal_clip_pooled else "ltcontext_frame_acc"
            )
            temporal_loss_key = "ltcontext_loss"
        else:
            temporal_head_label = temporal_metric_key = temporal_loss_key = None

        def _copy_module_state(module: Optional[nn.Module]):
            if module is None:
                return None
            return {
                key: value.detach().cpu().clone() for (key, value) in module.state_dict().items()
            }

        def _current_temporal_probe():
            if temporal_probe_kind in {"ltcontext", "causal_ltcontext"}:
                return self.ltcontext_probe
            return None

        selected_probe_state = None
        for pe in range(1, probe_epochs + 1):
            if self.ltcontext_probe is not None:
                self.ltcontext_probe.train()
            if self.lin_probe is not None:
                self.lin_probe.train()
            total_temporal = total_lin = 0
            correct_temporal = correct_lin = 0
            total_temporal_non_bg = total_lin_non_bg = 0
            correct_temporal_non_bg = correct_lin_non_bg = 0
            loss_sum_temporal = loss_sum_lin = 0.0
            steps = 0
            epoch_start_time = time.perf_counter()
            total_batches = max(1, math.ceil(len(batch_gen.list_of_examples) / float(batch_size)))
            batch_gen.reset()
            while batch_gen.has_next():
                batch_start_time = time.perf_counter()
                normalized = self._normalize_loader_batch(batch_gen.next_batch(batch_size))
                x = normalized["x4d"].to(device)
                (B, N, L, _) = x.shape
                y = normalized["target_flat"].to(device)
                mask_3d = normalized["valid_mask_3d"].to(device)
                mask_f = normalized["valid_mask_flat"].to(device)
                segment_lengths = normalized["segment_lengths"]
                segment_lengths_dev = (
                    segment_lengths.to(device) if segment_lengths is not None else None
                )
                with torch.no_grad():
                    t_full = self._probe_features(
                        x,
                        mask_3d,
                        probe_feature_source=probe_feature_source,
                        segment_lengths=segment_lengths_dev,
                    )
                if (
                    temporal_probe_kind in {"ltcontext", "causal_ltcontext"}
                    and self.ltcontext_probe is not None
                    and (self.ltcontext_probe_opt is not None)
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
                                stage_logits.transpose(2, 1).contiguous(),
                                normalized,
                                mask_3d,
                                device,
                            )
                            loss_temporal = loss_temporal + self.ce(
                                clip_logits.contiguous().view(-1, self.num_classes),
                                clip_labels.view(-1),
                            )
                    else:
                        loss_temporal = self.ltcontext_probe.loss(predictions, y)
                    self.ltcontext_probe_opt.zero_grad(set_to_none=True)
                    loss_temporal.backward()
                    self.ltcontext_probe_opt.step()
                    loss_sum_temporal += float(loss_temporal.item())
                    if temporal_clip_pooled:
                        (clip_logits, clip_labels, clip_mask) = self._prepare_clip_pooled_logits(
                            predictions[-1].transpose(2, 1).contiguous(),
                            normalized,
                            mask_3d,
                            device,
                        )
                        pred_clip = clip_logits.argmax(dim=-1)
                        correct_temporal += (
                            ((pred_clip == clip_labels).float() * clip_mask).sum().item()
                        )
                        total_temporal += torch.sum(clip_mask).item()
                        non_bg_mask = clip_mask.bool() & (clip_labels != int(background_label_id))
                        correct_temporal_non_bg += int(
                            ((pred_clip == clip_labels) & non_bg_mask).sum().item()
                        )
                        total_temporal_non_bg += int(non_bg_mask.sum().item())
                    else:
                        pred_temporal = predictions[-1].argmax(dim=1)
                        correct_temporal += ((pred_temporal == y).float() * mask_f).sum().item()
                        total_temporal += torch.sum(mask_f).item()
                        non_bg_mask = mask_f.bool() & (y != int(background_label_id))
                        correct_temporal_non_bg += int(
                            ((pred_temporal == y) & non_bg_mask).sum().item()
                        )
                        total_temporal_non_bg += int(non_bg_mask.sum().item())
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
                self.lin_probe_opt.zero_grad(set_to_none=True)
                loss_lin.backward()
                self.lin_probe_opt.step()
                loss_sum_lin += loss_lin.item()
                correct_lin += (
                    ((pred_lin == y_single_class).float() * mask_single_class).sum().item()
                )
                total_lin += torch.sum(mask_single_class).item()
                non_bg_mask = mask_single_class.bool() & (
                    y_single_class != int(background_label_id)
                )
                correct_lin_non_bg += int(((pred_lin == y_single_class) & non_bg_mask).sum().item())
                total_lin_non_bg += int(non_bg_mask.sum().item())
                steps += 1
                if progress_log_every > 0 and (
                    steps == 1 or steps % progress_log_every == 0 or (not batch_gen.has_next())
                ):
                    now = time.perf_counter()
                    elapsed = now - epoch_start_time
                    batch_sec = now - batch_start_time
                    avg_sec = elapsed / max(1, steps)
                    remaining = max(0, total_batches - steps)
                    eta_min = remaining * avg_sec / 60.0
                    logger.info(
                        f"[probe][progress][{temporal_head_label}] epoch={pe}/{probe_epochs} batch={steps}/{total_batches} tokens={int(mask_f.sum().item())} batch_sec={batch_sec:.2f} avg_sec={avg_sec:.2f} eta={eta_min:.1f}m"
                    )
            train_loss_temporal = loss_sum_temporal / max(1, steps)
            train_acc_temporal = correct_temporal / total_temporal if total_temporal > 0 else 0.0
            train_acc_temporal_non_bg = (
                correct_temporal_non_bg / total_temporal_non_bg
                if total_temporal_non_bg > 0
                else 0.0
            )
            train_loss_lin = loss_sum_lin / max(1, steps)
            train_acc_lin = correct_lin / total_lin if total_lin > 0 else 0.0
            train_acc_lin_non_bg = (
                correct_lin_non_bg / total_lin_non_bg if total_lin_non_bg > 0 else 0.0
            )
            if (
                temporal_probe_kind in {"ltcontext", "causal_ltcontext"}
                and self.ltcontext_probe_opt is not None
            ):
                curr_lr_temporal = float(self.ltcontext_probe_opt.param_groups[0]["lr"])
            else:
                curr_lr_temporal = 0.0
            curr_lr_lin = float(self.lin_probe_opt.param_groups[0]["lr"])
            if scheduler_temporal is not None:
                scheduler_temporal.step()
                curr_lr_temporal = float(scheduler_temporal.get_last_lr()[0])
            if scheduler_lin is not None:
                scheduler_lin.step()
                curr_lr_lin = float(scheduler_lin.get_last_lr()[0])
            if temporal_probe_kind is not None:
                logger.info(
                    f"[probe][train][{temporal_head_label}] epoch={pe}/{probe_epochs} loss={train_loss_temporal:.4f} acc={train_acc_temporal * 100:.2f}% acc_no_bg={train_acc_temporal_non_bg * 100:.2f}% lr={curr_lr_temporal:.2e}"
                )
            logger.info(
                f"[probe][train][Linear] epoch={pe}/{probe_epochs} loss={train_loss_lin:.4f} acc={train_acc_lin * 100:.2f}% acc_no_bg={train_acc_lin_non_bg * 100:.2f}% lr={curr_lr_lin:.2e}"
            )
            if (
                val_batch_gen is not None
                and val_batch_size is not None
                and (self.lin_probe is not None)
            ):
                (val_stats, metric_values) = evaluate_probe_epoch(
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
                )
                for key, value in metric_values.items():
                    direction = metric_direction.get(key, "max")
                    baseline = best_track[key]["value"]
                    better = value > baseline if direction == "max" else value < baseline
                    if better:
                        best_track[key] = {"value": float(value), "epoch": pe, "stats": val_stats}
                        if key == select_metric:
                            selected_probe_state = {
                                "temporal": _copy_module_state(_current_temporal_probe()),
                                "linear": _copy_module_state(self.lin_probe),
                                "epoch": int(pe),
                            }
                            pretty_val = (
                                value * 100.0 if "acc" in key or "clip_majority" in key else value
                            )
                            suffix = "%" if "acc" in key or "clip_majority" in key else ""
                            logger.info(
                                f"[probe][val] New best for {key} at epoch {pe}: {pretty_val:.2f}{suffix}"
                            )
        self.model.student_enc.train(was_train_student)
        for p in self.model.student_enc.parameters():
            p.requires_grad_(True)
        batch_gen.reset()
        self._probe_global_epoch_base += probe_epochs
        selected_stats = best_track.get(select_metric, {}).get("stats")
        best_metrics = {k: v["value"] for (k, v) in best_track.items() if v["epoch"] != -1}
        best_epochs = {k: v["epoch"] for (k, v) in best_track.items() if v["epoch"] != -1}
        if selected_probe_state is not None:
            temporal_module = _current_temporal_probe()
            if temporal_module is not None and selected_probe_state["temporal"] is not None:
                temporal_module.load_state_dict(selected_probe_state["temporal"], strict=True)
            if self.lin_probe is not None and selected_probe_state["linear"] is not None:
                self.lin_probe.load_state_dict(selected_probe_state["linear"], strict=True)
            logger.info(
                f"[probe][val] Restored probe weights from selected {select_metric} epoch {selected_probe_state['epoch']} before returning."
            )
        if best_metrics and self.wandb_run is not None:
            step_epoch = best_track.get(select_metric, {}).get("epoch", probe_epochs)
            if step_epoch == -1:
                step_epoch = probe_epochs
            payload = {
                "probe/session": int(curr_session),
                "probe/epoch": int(step_epoch),
                "probe/global_epoch": int(local_base + step_epoch),
                "probe_lin/val/acc": float(best_track["lin_clip_acc"]["value"]),
                "probe_lin/val/clip_majority": float(best_track["lin_clip_acc"]["value"]),
                "probe_lin/val/frame_acc": float(best_track["lin_frame_acc"]["value"]),
                "probe_lin/val/loss": float(best_track["lin_loss"]["value"]),
            }
            lin_stats = (
                best_track.get("lin_clip_acc", {}).get("stats")
                or best_track.get("lin_frame_acc", {}).get("stats")
                or {}
            )
            if "lin_frame_acc_no_bg" in lin_stats:
                payload["probe_lin/val/frame_acc_no_bg"] = float(lin_stats["lin_frame_acc_no_bg"])
            if not math.isnan(float(lin_stats.get("lin_edit_score", float("nan")))):
                payload.update(
                    {
                        "probe_lin/val/edit_score": float(lin_stats.get("lin_edit_score")),
                        "probe_lin/val/f1_10": float(lin_stats.get("lin_f1_10", float("nan"))),
                        "probe_lin/val/f1_25": float(lin_stats.get("lin_f1_25", float("nan"))),
                        "probe_lin/val/f1_50": float(lin_stats.get("lin_f1_50", float("nan"))),
                    }
                )
            if temporal_probe_kind is not None:
                payload.update(
                    {
                        "probe/val/acc": float(best_track[temporal_metric_key]["value"]),
                        "probe/val/clip_majority": float(best_track["clip_majority_acc"]["value"]),
                        "probe/val/loss": float(best_track[temporal_loss_key]["value"]),
                    }
                )
                temporal_stats = best_track.get(temporal_metric_key, {}).get("stats") or {}
                metric_prefix = "ltcontext"
                if not math.isnan(
                    float(temporal_stats.get(f"{metric_prefix}_edit_score", float("nan")))
                ):
                    payload.update(
                        {
                            "probe/val/edit_score": float(
                                temporal_stats.get(f"{metric_prefix}_edit_score")
                            ),
                            "probe/val/f1_10": float(
                                temporal_stats.get(f"{metric_prefix}_f1_10", float("nan"))
                            ),
                            "probe/val/f1_25": float(
                                temporal_stats.get(f"{metric_prefix}_f1_25", float("nan"))
                            ),
                            "probe/val/f1_50": float(
                                temporal_stats.get(f"{metric_prefix}_f1_50", float("nan"))
                            ),
                        }
                    )
            self._wb_log(payload)
        if selected_stats is not None:
            selected_stats = dict(selected_stats)
            selected_stats["best_metrics"] = best_metrics
            selected_stats["best_epochs"] = best_epochs
            if temporal_results is not None:
                selected_stats["temporal_continuity"] = temporal_results["summary"]
        return selected_stats
