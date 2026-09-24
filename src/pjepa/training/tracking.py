try:
    import wandb
except ImportError:
    wandb = None

"""Optional experiment tracking."""
from typing import Dict, Optional
import os
from pjepa.utils.logging import logger


class TrackingMixin:
    def _wandb_init_if_needed(self, train_hparams: Dict):
        if not self.use_wandb:
            logger.info("[wandb] Disabled by use_wandb=False.")
            return
        if wandb is None:
            logger.info("[wandb] Not installed; skipping logging.")
            return
        if self.wandb_run is not None:
            try:
                wandb.config.update(train_hparams, allow_val_change=True)
            except Exception:
                pass
            return
        if self.wandb_run_name is None:
            run_name = f"{self.dataset}-{self.split}".strip("-")
        else:
            run_name = self.wandb_run_name
        mode = self.wandb_mode or os.getenv("WANDB_MODE")
        self.wandb_run = wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            config=train_hparams,
            name=run_name,
            mode=mode,
        )
        logger.info(
            f"[wandb] Initialized run: {(self.wandb_run.name if self.wandb_run else 'N/A')}"
        )
        if self.wandb_run:
            try:
                wandb.define_metric("train/epoch")
                wandb.define_metric("train/*", step_metric="train/epoch")
                wandb.define_metric("val/*", step_metric="train/epoch")
                wandb.define_metric("probe/session")
                wandb.define_metric("probe/epoch")
                wandb.define_metric("probe/global_epoch")
                wandb.define_metric("probe/train/*", step_metric="probe/epoch")
                wandb.define_metric("probe/val/*", step_metric="probe/epoch")
                wandb.define_metric("probe_lin/train/*", step_metric="probe/epoch")
                wandb.define_metric("probe_lin/val/*", step_metric="probe/epoch")
                wandb.define_metric("probe_temporal/*", step_metric="train/epoch")
            except Exception:
                pass

    def _wb_log(self, data: Dict, *, step: Optional[int] = None):
        if self.wandb_run is not None and wandb is not None:
            try:
                if step is None:
                    wandb.log(data)
                else:
                    wandb.log(data, step=int(step))
            except Exception:
                pass

    def _wb_log_val_epoch(self, data: Dict):
        if self.wandb_run is not None and wandb is not None:
            payload = {"train/epoch": int(self.current_epoch)}
            payload.update(data)
            try:
                wandb.log(payload)
            except Exception:
                pass
