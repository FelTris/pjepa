"""Probe construction, label mapping, pooling, and frame predictions."""

from typing import Dict, Optional
import torch
from torch import optim
from pjepa.utils.logging import logger
from pjepa.probes.linear import LinearProbe, ForegroundActivityLinearProbe
from pjepa.probes.ltcontext_probe import LTContextProbe
from pjepa.probes.causal_ltcontext import CausalLTContextProbe


class ProbeOperationsMixin:
    def _init_linear_probe_if_needed(
        self,
        num_classes: int,
        device,
        lr: float,
        weight_decay: float,
        head_mode: str = "single",
        background_label_id: int = -100,
        in_dim: Optional[int] = None,
    ):
        if self.lin_probe is not None:
            del self.lin_probe
            del self.lin_probe_opt
        self.linear_probe_head_mode = str(head_mode).lower()
        self.linear_probe_background_label_id = int(background_label_id)
        self.linear_probe_foreground_labels = [
            label
            for label in range(int(num_classes))
            if label != self.linear_probe_background_label_id
        ]
        label_to_foreground_index = torch.full((int(num_classes),), -100, dtype=torch.long)
        for fg_index, label in enumerate(self.linear_probe_foreground_labels):
            label_to_foreground_index[int(label)] = int(fg_index)
        self.linear_probe_label_to_foreground_index = label_to_foreground_index.to(device)
        probe_dim = int(in_dim or self.dim)
        if self.linear_probe_head_mode == "foreground_activity":
            if not 0 <= self.linear_probe_background_label_id < int(num_classes):
                raise ValueError(
                    f"foreground_activity linear probe requires a valid background_label_id in [0, {int(num_classes)}), got {background_label_id}."
                )
            self.lin_probe = ForegroundActivityLinearProbe(
                probe_dim, num_foreground_classes=len(self.linear_probe_foreground_labels)
            ).to(device)
        elif self.linear_probe_head_mode == "single":
            self.lin_probe = LinearProbe(probe_dim, num_classes).to(device)
        else:
            raise ValueError(
                f"Unsupported linear probe head_mode='{head_mode}'. Use 'single' or 'foreground_activity'."
            )
        self.lin_probe_opt = torch.optim.AdamW(
            self.lin_probe.parameters(), lr=lr, weight_decay=weight_decay
        )

    def _init_LTContext_probe_if_needed(
        self,
        num_classes: int,
        device,
        lr: float,
        weight_decay: float,
        cfg_overrides: Optional[Dict[str, object]] = None,
        solver_cfg: Optional[Dict[str, object]] = None,
        causal: bool = False,
        in_dim: Optional[int] = None,
    ):
        if self.ltcontext_probe is not None:
            del self.ltcontext_probe
            del self.ltcontext_probe_opt
        probe_cls = CausalLTContextProbe if causal else LTContextProbe
        self.ltcontext_probe = probe_cls(
            in_dim=int(in_dim or self.dim), num_classes=num_classes, cfg_overrides=cfg_overrides
        ).to(device)
        logger.info(
            "[probe][init] class=%s causal_attention_mode=%s"
            % (
                probe_cls.__name__,
                getattr(self.ltcontext_probe, "causal_attention_mode", "non_causal"),
            )
        )
        self.ltcontext_probe_opt = self._construct_ltcontext_optimizer(
            lr=lr, weight_decay=weight_decay, solver_cfg=solver_cfg
        )

    def _construct_ltcontext_optimizer(
        self, lr: float, weight_decay: float, solver_cfg: Optional[Dict[str, object]] = None
    ) -> torch.optim.Optimizer:
        if self.ltcontext_probe is None:
            raise RuntimeError(
                "LTContext probe must be initialized before constructing its optimizer."
            )
        solver_cfg = solver_cfg or {}
        method = str(
            solver_cfg.get("OPTIMIZING_METHOD", solver_cfg.get("optimizing_method", "adamw"))
        ).lower()
        base_lr = float(solver_cfg.get("BASE_LR", solver_cfg.get("base_lr", lr)))
        wd = float(solver_cfg.get("WEIGHT_DECAY", solver_cfg.get("weight_decay", weight_decay)))
        params = filter(lambda p: p.requires_grad, self.ltcontext_probe.parameters())
        if method == "adam":
            return torch.optim.Adam(params, lr=base_lr, betas=(0.9, 0.999), weight_decay=wd)
        if method == "adamw":
            return torch.optim.AdamW(params, lr=base_lr, betas=(0.9, 0.999), weight_decay=wd)
        if method == "sgd":
            return torch.optim.SGD(
                params,
                lr=base_lr,
                momentum=float(solver_cfg.get("MOMENTUM", solver_cfg.get("momentum", 0.9))),
                weight_decay=wd,
                dampening=float(solver_cfg.get("DAMPENING", solver_cfg.get("dampening", 0.0))),
                nesterov=bool(solver_cfg.get("NESTEROV", solver_cfg.get("nesterov", True))),
            )
        raise ValueError(f"Unsupported LTContext optimizer '{method}'.")

    @staticmethod
    def _construct_ltcontext_scheduler(
        optimizer: torch.optim.Optimizer,
        solver_cfg: Optional[Dict[str, object]] = None,
        fallback_milestones=None,
        fallback_gamma: float = 0.1,
    ):
        solver_cfg = solver_cfg or {}
        policy = str(solver_cfg.get("LR_POLICY", solver_cfg.get("lr_policy", ""))).lower()
        if policy == "constant_cosine_decay":
            warmup_epochs = int(solver_cfg.get("WARMUP_EPOCHS", solver_cfg.get("warmup_epochs", 0)))
            t_max = int(
                solver_cfg.get("T_MAX", solver_cfg.get("t_max", solver_cfg.get("MAX_EPOCH", 1)))
            )
            eta_min = float(solver_cfg.get("ETA_MIN", solver_cfg.get("eta_min", 0.0)))
            warmup_epochs = max(0, warmup_epochs)
            cosine_epochs = max(1, t_max - warmup_epochs)
            if warmup_epochs > 0:
                return optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[
                        optim.lr_scheduler.ConstantLR(
                            optimizer, factor=1.0, total_iters=warmup_epochs
                        ),
                        optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=cosine_epochs, eta_min=eta_min
                        ),
                    ],
                    milestones=[warmup_epochs],
                )
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_epochs, eta_min=eta_min
            )
        if policy in {"", "identity"}:
            if fallback_milestones:
                ms = sorted((int(m) for m in fallback_milestones))
                return optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=ms, gamma=fallback_gamma
                )
            return None
        raise ValueError(f"Unsupported LTContext LR policy '{policy}'.")

    def _aggregate_segment_logits(self, logits_flat, target_flat, valid_mask_flat, segment_lengths):
        if segment_lengths is None:
            raise ValueError("segment_lengths are required for flat-sequence clip aggregation.")
        (B, _, C) = logits_flat.shape
        S = int(segment_lengths.shape[1])
        seg_logits = logits_flat.new_zeros(B, S, C)
        seg_targets = torch.ones(B, S, dtype=torch.long, device=logits_flat.device) * -100
        seg_mask = torch.zeros(B, S, dtype=torch.bool, device=logits_flat.device)
        for b in range(B):
            offset = 0
            for s, seg_len in enumerate(segment_lengths[b].tolist()):
                seg_len = int(seg_len)
                if seg_len <= 0 or offset >= logits_flat.shape[1]:
                    break
                seg_len = min(seg_len, logits_flat.shape[1] - offset)
                seg_valid = valid_mask_flat[b, offset : offset + seg_len]
                if seg_valid.any():
                    seg_logits[b, s] = logits_flat[b, offset : offset + seg_len][seg_valid].amax(
                        dim=0
                    )
                    seg_targets[b, s] = int(
                        target_flat[b, offset : offset + seg_len][seg_valid][0].item()
                    )
                    seg_mask[b, s] = True
                offset += seg_len
        return (seg_logits, seg_targets, seg_mask)

    def _prepare_linear_probe_batch(
        self, logits_token, normalized, valid_mask_3d, device, linear_probe_pool: bool
    ):
        target_flat = normalized["target_flat"].to(device)
        valid_mask_flat = normalized["valid_mask_flat"].to(device)
        if linear_probe_pool:
            if normalized["flat_mode"]:
                segment_lengths = normalized["segment_lengths"]
                segment_lengths_dev = (
                    segment_lengths.to(device) if segment_lengths is not None else None
                )
                return self._aggregate_segment_logits(
                    logits_token, target_flat, valid_mask_flat, segment_lengths_dev
                )
            (B, N, L) = valid_mask_3d.shape
            return (
                logits_token.reshape(B, N, L, self.num_classes).amax(dim=-2),
                normalized["target_3d"].to(device)[:, :, 0],
                valid_mask_3d[:, :, 0],
            )
        if normalized["flat_mode"]:
            return (logits_token, target_flat, valid_mask_flat)
        (B, N, L) = valid_mask_3d.shape
        return (
            logits_token.reshape(B, N, L, self.num_classes),
            normalized["target_3d"].to(device),
            valid_mask_3d,
        )

    def _prepare_clip_pooled_logits(self, logits_token, normalized, valid_mask_3d, device):
        target_flat = normalized["target_flat"].to(device)
        valid_mask_flat = normalized["valid_mask_flat"].to(device)
        if normalized["flat_mode"]:
            segment_lengths = normalized["segment_lengths"]
            segment_lengths_dev = (
                segment_lengths.to(device) if segment_lengths is not None else None
            )
            return self._aggregate_segment_logits(
                logits_token, target_flat, valid_mask_flat, segment_lengths_dev
            )
        (B, N, L) = valid_mask_3d.shape
        return (
            logits_token.reshape(B, N, L, self.num_classes).amax(dim=-2),
            normalized["target_3d"].to(device)[:, :, 0],
            valid_mask_3d[:, :, 0],
        )

    def _clip_predictions_to_dict(self, pred_clip, normalized, clip_mask):
        val_dict = {}
        names = normalized["names"]
        if names is None:
            return val_dict
        for b in range(pred_clip.shape[0]):
            base = names[b] if len(names) > b else []
            for s in range(pred_clip.shape[1]):
                if not bool(clip_mask[b, s].item()):
                    continue
                if s < len(base):
                    val_dict[base[s]] = int(pred_clip[b, s].detach().cpu().item())
        return val_dict

    def _prepare_foreground_activity_linear_batch(
        self, logits_token, normalized, valid_mask_3d, device, linear_probe_pool: bool
    ):
        if not isinstance(logits_token, dict):
            raise TypeError("foreground_activity linear probe expects dict logits.")
        if self.linear_probe_label_to_foreground_index is None:
            raise RuntimeError("Foreground label mapping has not been initialized.")
        (fg_logits, labels, mask) = self._prepare_linear_probe_batch(
            logits_token["foreground"],
            normalized,
            valid_mask_3d,
            device,
            linear_probe_pool=linear_probe_pool,
        )
        (activity_logits, activity_labels_source, activity_mask) = self._prepare_linear_probe_batch(
            logits_token["activity"],
            normalized,
            valid_mask_3d,
            device,
            linear_probe_pool=linear_probe_pool,
        )
        fg_targets = self.linear_probe_label_to_foreground_index[labels.clamp_min(0)]
        fg_targets = fg_targets.masked_fill(~mask.bool(), -100)
        activity_targets = (
            activity_labels_source != int(self.linear_probe_background_label_id)
        ).long()
        activity_targets = activity_targets.masked_fill(~activity_mask.bool(), -100)
        return (fg_logits, fg_targets, activity_logits, activity_targets, labels, mask)

    def _foreground_activity_predictions(self, fg_logits, activity_logits):
        foreground_index = fg_logits.argmax(dim=-1)
        foreground_labels = torch.as_tensor(
            self.linear_probe_foreground_labels, dtype=torch.long, device=fg_logits.device
        )
        pred_foreground = foreground_labels[foreground_index]
        pred_activity = activity_logits.argmax(dim=-1)
        background = torch.full_like(pred_foreground, int(self.linear_probe_background_label_id))
        return torch.where(pred_activity == 1, pred_foreground, background)

    def _linear_probe_loss_and_predictions(
        self,
        logits_token,
        normalized,
        mask_3d,
        device,
        linear_probe_pool: bool,
        activity_loss_weight: float = 1.0,
        foreground_loss_weight: float = 1.0,
    ):
        if isinstance(logits_token, dict):
            (fg_logits, fg_targets, activity_logits, activity_targets, labels, mask) = (
                self._prepare_foreground_activity_linear_batch(
                    logits_token, normalized, mask_3d, device, linear_probe_pool=linear_probe_pool
                )
            )
            loss_fg = self.ce(
                fg_logits.contiguous().view(-1, len(self.linear_probe_foreground_labels)),
                fg_targets.view(-1),
            )
            loss_activity = self.ce(
                activity_logits.contiguous().view(-1, 2), activity_targets.view(-1)
            )
            loss = (
                float(foreground_loss_weight) * loss_fg
                + float(activity_loss_weight) * loss_activity
            )
            pred = self._foreground_activity_predictions(fg_logits, activity_logits)
            aux = {
                "loss_foreground": float(loss_fg.detach().item()),
                "loss_activity": float(loss_activity.detach().item()),
            }
            return (loss, pred, labels, mask, aux)
        (logits_lin_flat, labels, mask) = self._prepare_linear_probe_batch(
            logits_token, normalized, mask_3d, device, linear_probe_pool=linear_probe_pool
        )
        loss = self.ce(logits_lin_flat.contiguous().view(-1, self.num_classes), labels.view(-1))
        (_, pred) = torch.max(logits_lin_flat.data, -1)
        return (loss, pred, labels, mask, {})

    def _segment_majority_vote(
        self, pred_flat, target_flat, valid_mask_flat, segment_lengths, names
    ):
        if segment_lengths is None:
            raise ValueError("segment_lengths are required for flat-sequence clip aggregation.")
        correct = 0
        total = 0
        val_dict = {}
        B = pred_flat.shape[0]
        for b in range(B):
            base = names[b] if names is not None and len(names) > b else []
            offset = 0
            for s, seg_len in enumerate(segment_lengths[b].tolist()):
                seg_len = int(seg_len)
                if seg_len <= 0 or offset >= pred_flat.shape[1]:
                    break
                seg_len = min(seg_len, pred_flat.shape[1] - offset)
                seg_valid = valid_mask_flat[b, offset : offset + seg_len]
                if seg_valid.any():
                    pred_seg = pred_flat[b, offset : offset + seg_len][seg_valid]
                    tgt_seg = target_flat[b, offset : offset + seg_len][seg_valid]
                    pred_label = int(torch.bincount(pred_seg).argmax().item())
                    true_label = int(torch.bincount(tgt_seg).argmax().item())
                    if s < len(base):
                        val_dict[base[s]] = pred_label
                    correct += int(pred_label == true_label)
                    total += 1
                offset += seg_len
        return (correct, total, val_dict)

    def _clip_majority_from_linear_predictions(self, pred_lin, normalized, mask_3d):
        if normalized["flat_mode"]:
            if pred_lin.ndim != 2 or normalized["segment_lengths"] is None:
                raise ValueError(
                    "Flat linear clip-majority expects [B, T] predictions with segment_lengths."
                )
            return self._segment_majority_vote(
                pred_lin.detach().cpu(),
                normalized["target_flat"].detach().cpu(),
                normalized["valid_mask_flat"].detach().cpu(),
                normalized["segment_lengths"],
                normalized["names"],
            )
        target_3d = normalized["target_3d"]
        (B, N, _) = mask_3d.shape
        correct = 0
        total = 0
        val_dict = {}
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
                        torch.bincount(target_3d[b, n][valid_vec].detach().cpu()).argmax().item()
                    )
                else:
                    p_clip = pred_lin[b, n][valid_vec].detach().cpu()
                    t_clip = target_3d[b, n][valid_vec].detach().cpu()
                    pred_label = int(torch.bincount(p_clip).argmax().item())
                    true_label = int(torch.bincount(t_clip).argmax().item())
                if n < len(base):
                    val_dict[base[n]] = pred_label
                correct += int(pred_label == true_label)
                total += 1
        return (correct, total, val_dict)

    def _dense_linear_sequences(self, pred_lin, normalized, mask_3d, linear_probe_pool: bool):
        if linear_probe_pool:
            raise ValueError(
                "Linear segmentation metrics are only defined for token-wise linear probe outputs."
            )
        sequences = []
        if normalized["flat_mode"]:
            target_flat = normalized["target_flat"]
            valid_mask_flat = normalized["valid_mask_flat"].bool()
            if pred_lin.ndim != 2:
                raise ValueError("Flat token-wise linear predictions expect [B, T] predictions.")
            for b in range(pred_lin.shape[0]):
                valid = valid_mask_flat[b]
                sequences.append(
                    (
                        pred_lin[b][valid].detach().cpu().numpy(),
                        target_flat[b][valid].detach().cpu().numpy(),
                    )
                )
            return sequences
        target_3d = normalized["target_3d"]
        (B, N, L) = mask_3d.shape
        for b in range(B):
            preds = []
            labels = []
            for n in range(N):
                valid = mask_3d[b, n].bool()
                if not valid.any():
                    continue
                labels.append(target_3d[b, n][valid].detach().cpu())
                preds.append(pred_lin[b, n][valid].detach().cpu())
            if preds:
                sequences.append(
                    (torch.cat(preds, dim=0).numpy(), torch.cat(labels, dim=0).numpy())
                )
        return sequences
