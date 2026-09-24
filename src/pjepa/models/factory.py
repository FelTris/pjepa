from typing import Any
from pjepa.models.feature_vjepa_rope import FeatureJEPA


def build_model(model_cfg: dict[str, Any], ssl_cfg: dict[str, Any]) -> FeatureJEPA:
    return FeatureJEPA(
        d_in=int(model_cfg.get("input_dim", 768)),
        d_model=int(model_cfg.get("dim", 768)),
        enc_depth=int(model_cfg.get("enc_depth", 4)),
        enc_heads=int(model_cfg.get("enc_heads", 16)),
        pred_depth=int(model_cfg.get("pred_depth", 2)),
        pred_heads=int(model_cfg.get("pred_heads", 16)),
        drop_path_prob=float(model_cfg.get("drop_path_prob", 0.0)),
        student_encoder_attention=str(ssl_cfg.get("student_encoder_attention", "block_causal")),
        student_encoder_block_size=int(ssl_cfg.get("student_encoder_block_size", 32)),
        rope_mode=str(ssl_cfg.get("rope_mode", "1d_flat")),
    )
