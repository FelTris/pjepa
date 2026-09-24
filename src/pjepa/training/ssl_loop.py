"""Masked feature prediction, optimizer/EMA steps, and encoder selection.

The loss, accumulation scaling, masking, and scheduler convention follow the
retained feature experiments. Probe fitting is delegated to training.probe_fit.
"""

from pathlib import Path
import math
import torch
from pjepa.training.snapshots import save_snapshot
from pjepa.utils.logging import logger


class SSLTrainingMixin:
    def _ssl_batch(self, raw_batch, device):
        normalized = self._normalize_loader_batch(raw_batch)
        valid = normalized["valid_mask_3d"].to(device)
        features = normalized["x4d"].to(device)
        lengths = self._build_segment_lengths_for_model(
            normalized["segment_lengths"], normalized["x4d"], normalized["flat_mode"], device
        )
        context, target, _ = self._make_ssl_masks(normalized, valid, device)
        if not target.any():
            raise ValueError(
                "SSL batch has no target tokens; sequences need at least two valid tokens."
            )
        batch = {"x": features, "valid_mask": valid, "context_mask": context, "target_mask": target}
        if lengths is not None:
            batch["segment_lengths"] = lengths
        return batch

    def train_epoch(self, generator, batch_size, device, accumulation_steps):
        self.model.train()
        self.teacher.eval()
        loss_sum = batches = pending = 0
        self.opt.zero_grad(set_to_none=True)
        while generator.has_next():
            batch = self._ssl_batch(generator.next_batch(batch_size), device)
            outputs = self.model(batch, teacher=self.teacher)
            loss = (outputs["preds_at_targets"] - outputs["teacher_at_targets"]).abs().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite SSL loss.")
            (loss / accumulation_steps).backward()
            loss_sum += float(loss.detach())
            batches += 1
            pending += 1
            if pending >= accumulation_steps or not generator.has_next():
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                pending = 0
                self.ema.update(teacher=self.teacher, student=self.model.student_enc)
        generator.reset()
        if not batches:
            raise ValueError("Training split is empty.")
        return loss_sum / batches

    @torch.no_grad()
    def validate(self, val_batch_gen, val_batch_size, device):
        model_mode, teacher_mode = self.model.training, self.teacher.training
        self.model.eval()
        self.teacher.eval()
        error = 0.0
        count = 0
        val_batch_gen.reset()
        while val_batch_gen.has_next():
            out = self.model(
                self._ssl_batch(val_batch_gen.next_batch(val_batch_size), device),
                teacher=self.teacher,
            )
            error += float((out["preds_at_targets"] - out["teacher_at_targets"]).abs().sum())
            count += out["preds_at_targets"].numel()
        val_batch_gen.reset()
        self.model.train(model_mode)
        self.teacher.train(teacher_mode)
        value = error / max(1, count)
        self._wb_log_val_epoch({"val/l1": value})
        return value

    def train(
        self,
        save_dir,
        batch_gen,
        num_epochs,
        batch_size,
        learning_rate,
        device,
        val_batch_gen=None,
        val_batch_size=1,
        ckpt_metric="lin_combined",
        lr_milestones=None,
        lr_gamma=0.1,
        probe_every=10,
        probe_epochs=5,
        probe_batch_size=None,
        probe_val_batch_size=None,
        probe_lr_lin=0.001,
        probe_lr_ltcontext=0.00025,
        probe_weight_decay=0.0,
        lin_lr_milestones=None,
        ltcontext_lr_milestones=None,
        ltcontext_cfg_overrides=None,
        val_every=10,
        probe_mode="linear",
        probe_batch_gen=None,
        probe_val_batch_gen=None,
        ssl_accumulation_steps=1,
        linear_probe_pool=False,
        background_label_id=-100,
        linear_probe_head_mode="single",
        linear_probe_activity_loss_weight=1.0,
        linear_probe_foreground_loss_weight=1.0,
        progress_log_every=0,
    ):
        if probe_mode not in {"linear", "ltcontext", "causal_ltcontext", "none"}:
            raise ValueError("Use linear, ltcontext, causal_ltcontext, or none.")
        if ssl_accumulation_steps < 1 or num_epochs < 1:
            raise ValueError("Epochs and accumulation steps must be positive.")
        if learning_rate != self.opt.param_groups[0]["lr"]:
            raise ValueError("Training LR must match the optimizer initialization.")
        self.model.to(device)
        self.teacher.to(device)
        output = Path(save_dir)
        output.mkdir(parents=True, exist_ok=True)
        scheduler = (
            torch.optim.lr_scheduler.MultiStepLR(
                self.opt, milestones=[m - 1 for m in lr_milestones], gamma=lr_gamma
            )
            if lr_milestones
            else None
        )
        self._wandb_init_if_needed(
            {
                "train/num_epochs": num_epochs,
                "train/batch_size": batch_size,
                "model/student_encoder_attention": self.student_encoder_attention,
            }
        )
        best = float("inf") if ckpt_metric.endswith("loss") else float("-inf")
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            loss = self.train_epoch(batch_gen, batch_size, device, int(ssl_accumulation_steps))
            if scheduler:
                scheduler.step()
            logger.info(f"SSL epoch {epoch}/{num_epochs}: masked L1={loss:.6f}")
            self._wb_log({"epoch": epoch, "train/loss": loss})
            if val_batch_gen is not None and val_every and epoch % val_every == 0:
                self.validate(val_batch_gen, val_batch_size, device)
            selection = None
            if probe_mode != "none" and probe_every and epoch % probe_every == 0:
                stats = self.run_linear_probe(
                    probe_batch_gen or batch_gen,
                    probe_batch_size or batch_size,
                    device,
                    val_batch_gen=probe_val_batch_gen or val_batch_gen,
                    val_batch_size=probe_val_batch_size or val_batch_size,
                    probe_epochs=probe_epochs,
                    probe_lr_lin=probe_lr_lin,
                    probe_lr_ltcontext=probe_lr_ltcontext,
                    probe_weight_decay=probe_weight_decay,
                    select_metric=ckpt_metric,
                    lin_lr_milestones=lin_lr_milestones
                    if lin_lr_milestones is not None
                    else [5, 15],
                    ltcontext_lr_milestones=ltcontext_lr_milestones,
                    ltcontext_cfg_overrides=ltcontext_cfg_overrides,
                    linear_only=probe_mode == "linear",
                    temporal_probe_kind=None if probe_mode == "linear" else probe_mode,
                    linear_probe_pool=linear_probe_pool,
                    background_label_id=background_label_id,
                    linear_probe_head_mode=linear_probe_head_mode,
                    linear_probe_activity_loss_weight=linear_probe_activity_loss_weight,
                    linear_probe_foreground_loss_weight=linear_probe_foreground_loss_weight,
                    progress_log_every=progress_log_every,
                )
                if stats is None or ckpt_metric not in stats:
                    raise ValueError(f"Probe did not produce selection metric {ckpt_metric!r}.")
                value = float(stats[ckpt_metric])
                selection = {"metric": ckpt_metric, "value": value}
                better = value < best if ckpt_metric.endswith("loss") else value > best
                if math.isfinite(value) and better:
                    best = value
                    save_snapshot(
                        output / "best.pt",
                        self,
                        epoch=epoch,
                        scheduler=scheduler,
                        selection=selection,
                    )
            save_snapshot(
                output / "last.pt", self, epoch=epoch, scheduler=scheduler, selection=selection
            )
