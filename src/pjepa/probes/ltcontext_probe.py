from types import SimpleNamespace
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_ltc_model(model_cfg):
    from pjepa._vendor.ltcontext.ltcontext import LTC

    return LTC(model_cfg)


def _flatten_overrides(updates: Optional[Dict[str, object]]) -> Dict[str, object]:
    flat = {}
    for key, value in (updates or {}).items():
        normalized_key = str(key).lower()
        if isinstance(value, dict):
            if normalized_key in {"ltc", "model"}:
                flat.update(_flatten_overrides(value))
            elif normalized_key == "attention":
                for child_key, child_value in value.items():
                    child_key = str(child_key).lower()
                    aliases = {
                        "num_attn_heads": "num_attn_heads",
                        "dropout": "attention_dropout",
                    }
                    flat[aliases.get(child_key, child_key)] = child_value
            else:
                flat.update(_flatten_overrides(value))
        else:
            aliases = {
                "num_layers": "num_layers",
                "num_stages": "num_stages",
                "model_dim": "model_dim",
                "windowed_attn_w": "windowed_attn_w",
                "long_term_attn_g": "long_term_attn_g",
                "conv_dilation_factor": "conv_dilation_factor",
                "dim_reduction": "dim_reduction",
                "channel_masking_prob": "channel_masking_prob",
                "dropout_prob": "dropout_prob",
                "use_instance_norm": "use_instance_norm",
                "num_attn_heads": "num_attn_heads",
                "attention_dropout": "attention_dropout",
                "dropout": "dropout_prob",
                "ce_loss_weight": "ce_loss_weight",
                "mse_loss_weight": "mse_loss_weight",
                "mse_loss_fraction": "mse_loss_fraction",
                "mse_loss_clip_val": "mse_loss_clip_val",
            }
            flat[aliases.get(normalized_key, normalized_key)] = value
    return flat


def _merge_dicts(
    base: Dict[str, object], updates: Optional[Dict[str, object]]
) -> Dict[str, object]:
    out = dict(base)
    for key, value in _flatten_overrides(updates).items():
        out[key] = value
    return out


class LTContextProbe(nn.Module):
    """pjepa-facing wrapper for the external LTContext architecture."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        cfg_overrides: Optional[Dict[str, object]] = None,
    ):
        super().__init__()
        cfg = _merge_dicts(
            {
                "num_layers": 9,
                "num_stages": 4,
                "model_dim": 64,
                "windowed_attn_w": 64,
                "long_term_attn_g": 64,
                "conv_dilation_factor": 2,
                "dim_reduction": 2.0,
                "channel_masking_prob": 0.3,
                "dropout_prob": 0.2,
                "use_instance_norm": True,
                "num_attn_heads": 1,
                "attention_dropout": 0.2,
                "ce_loss_weight": 1.0,
                "mse_loss_weight": 0.15,
                "mse_loss_clip_val": 16.0,
            },
            cfg_overrides,
        )
        if "mse_loss_fraction" in cfg:
            cfg["mse_loss_weight"] = cfg["mse_loss_fraction"]
        self.cfg = cfg
        self.num_classes = int(num_classes)
        self.ce_loss_weight = float(cfg["ce_loss_weight"])
        self.mse_loss_weight = float(cfg["mse_loss_weight"])
        self.mse_loss_clip_val = float(cfg["mse_loss_clip_val"])

        model_cfg = SimpleNamespace(
            INPUT_DIM=int(in_dim),
            NUM_CLASSES=int(num_classes),
            LTC=SimpleNamespace(
                NUM_LAYERS=int(cfg["num_layers"]),
                NUM_STAGES=int(cfg["num_stages"]),
                MODEL_DIM=int(cfg["model_dim"]),
                WINDOWED_ATTN_W=int(cfg["windowed_attn_w"]),
                LONG_TERM_ATTN_G=int(cfg["long_term_attn_g"]),
                CONV_DILATION_FACTOR=int(cfg["conv_dilation_factor"]),
                DIM_REDUCTION=float(cfg["dim_reduction"]),
                CHANNEL_MASKING_PROB=float(cfg["channel_masking_prob"]),
                DROPOUT_PROB=float(cfg["dropout_prob"]),
                USE_INSTANCE_NORM=bool(cfg["use_instance_norm"]),
            ),
            ATTENTION=SimpleNamespace(
                NUM_ATTN_HEADS=int(cfg["num_attn_heads"]),
                DROPOUT=float(cfg["attention_dropout"]),
            ),
        )
        self.model = _build_ltc_model(model_cfg)

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError(f"Expected features with shape [B, T, D], got {tuple(features.shape)}")
        if valid_mask.shape != features.shape[:2]:
            raise ValueError(
                f"Expected valid_mask with shape [B, T], got {tuple(valid_mask.shape)} for features {tuple(features.shape)}"
            )
        masks = valid_mask.bool().unsqueeze(1)
        logits = self.model(features.movedim(-1, 1), masks)
        return logits

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 4:
            raise ValueError(f"Expected logits with shape [S, B, C, T], got {tuple(logits.shape)}")
        if targets.shape != (logits.shape[1], logits.shape[3]):
            raise ValueError(f"Expected targets with shape [B, T], got {tuple(targets.shape)}")

        total = logits.new_tensor(0.0)
        for stage_logits in logits:
            ce = F.cross_entropy(
                stage_logits.transpose(2, 1).contiguous().view(-1, self.num_classes),
                targets.contiguous().view(-1),
                ignore_index=-100,
            )
            if stage_logits.shape[-1] > 1 and self.mse_loss_weight != 0.0:
                mse = F.mse_loss(
                    F.log_softmax(stage_logits[:, :, 1:], dim=1),
                    F.log_softmax(stage_logits.detach()[:, :, :-1], dim=1),
                    reduction="none",
                )
                mse = torch.clamp(mse, min=0.0, max=self.mse_loss_clip_val).mean()
            else:
                mse = stage_logits.new_tensor(0.0)
            total = total + self.ce_loss_weight * ce + self.mse_loss_weight * mse
        return total
