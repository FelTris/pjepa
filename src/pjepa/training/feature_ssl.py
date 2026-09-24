"""Compose the feature SSL trainer from focused training, data, and probe components."""

from typing import List, Optional
import torch
import torch.nn as nn
from pjepa.training.tracking import TrackingMixin
from pjepa.data.feature_batches import FeatureBatchMixin
from pjepa.probes.operations import ProbeOperationsMixin
from pjepa.training.ssl_loop import SSLTrainingMixin
from pjepa.training.probe_fit import ProbeFitMixin
from pjepa.evaluation.saved_probes import SavedProbeMixin
from pjepa.models.feature_vjepa_rope import FeatureJEPA, normalize_rope_mode
from pjepa.models.ema import EMA


class Trainer(
    TrackingMixin,
    FeatureBatchMixin,
    ProbeOperationsMixin,
    SSLTrainingMixin,
    ProbeFitMixin,
    SavedProbeMixin,
):
    def __init__(
        self,
        dim,
        enc_depth,
        pred_depth,
        ema_momentum,
        lr,
        num_classes,
        dataset,
        split,
        mask_ratio,
        **kwargs,
    ):
        self.dim = dim
        self.input_dim = int(kwargs.pop("input_dim", dim))
        self.enc_depth = enc_depth
        self.pred_depth = pred_depth
        student_encoder_attention = kwargs.pop("student_encoder_attention", None)
        legacy_causal_student_encoder = kwargs.pop("causal_student_encoder", None)
        if student_encoder_attention is None:
            student_encoder_attention = (
                "causal" if legacy_causal_student_encoder else "bidirectional"
            )
        self.student_encoder_attention = str(student_encoder_attention)
        self.student_encoder_block_size = max(1, int(kwargs.pop("student_encoder_block_size", 16)))
        self.rope_mode = normalize_rope_mode(kwargs.pop("rope_mode", "2d_clip_frame"))
        self.num_classes = num_classes
        self.dataset = dataset
        self.split = split
        self.mask_ratio = mask_ratio
        self.directional_mask_prob = max(
            0.0, min(1.0, float(kwargs.pop("directional_mask_prob", 0.0)))
        )
        self.directional_future_mask_ratio = max(
            0.0, min(1.0, float(kwargs.pop("directional_future_mask_ratio", 0.75)))
        )
        self.directional_min_context_clips = max(
            1, int(kwargs.pop("directional_min_context_clips", 1))
        )
        self.directional_min_future_clips = max(
            1, int(kwargs.pop("directional_min_future_clips", 1))
        )
        self.best_val_metric = float("-inf")
        self.best_model_path = None
        self.best_opt_path = None
        self.best_csv_path = None
        self.best_json_path = None
        self.use_wandb: bool = kwargs.pop("use_wandb", True)
        self.wandb_project: str = kwargs.pop("wandb_project", "vjepa-training")
        self.wandb_entity: Optional[str] = kwargs.pop("wandb_entity", None)
        self.wandb_mode: Optional[str] = kwargs.pop("wandb_mode", None)
        self.wandb_run = None
        self.wandb_run_name: Optional[str] = kwargs.pop("run_name", None)
        self.current_epoch: Optional[int] = None
        self.probe_session = 0
        self._probe_global_epoch_base = 0
        enc_heads = int(kwargs.pop("enc_heads", 16))
        pred_heads = int(kwargs.pop("pred_heads", 16))
        if kwargs:
            raise TypeError(f"Unknown trainer options: {sorted(kwargs)}")
        self.model = FeatureJEPA(
            enc_heads=enc_heads,
            pred_heads=pred_heads,
            d_in=self.input_dim,
            d_model=dim,
            enc_depth=enc_depth,
            pred_depth=pred_depth,
            student_encoder_attention=self.student_encoder_attention,
            student_encoder_block_size=self.student_encoder_block_size,
            rope_mode=self.rope_mode,
        )
        self.teacher = self.model.build_teacher()
        self.m_start = ema_momentum
        self.ema = EMA(momentum=ema_momentum)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.04)
        self.ltcontext_probe: Optional[nn.Module] = None
        self.ltcontext_probe_opt: Optional[torch.optim.Optimizer] = None
        self.lin_probe: Optional[nn.Module] = None
        self.lin_probe_opt: Optional[torch.optim.Optimizer] = None
        self.linear_probe_head_mode = "single"
        self.linear_probe_background_label_id = -100
        self.linear_probe_foreground_labels: List[int] = []
        self.linear_probe_label_to_foreground_index: Optional[torch.Tensor] = None
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        print(sum((p.numel() for p in self.model.parameters() if p.requires_grad)))
