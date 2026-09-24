"""Feature-sequence inference independent of trainers, probes, and optimizers."""

from __future__ import annotations
import torch


@torch.inference_mode()
def encode(
    model,
    features: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    segment_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encode [batch, time, input_dim] features, retaining padded positions.

    Clip-causal/2D-RoPE historical models require oracle segment lengths.
    Block-causal models need no segment boundaries. The whole sequence is
    retained; arbitrary chunking would change its temporal context.
    """
    if features.ndim != 3 or features.shape[-1] != model.student_enc.d_in:
        raise ValueError(f"Expected [B,T,{model.student_enc.d_in}] features.")
    device = next(model.parameters()).device
    features = features.to(device)
    valid = (
        torch.ones(features.shape[:2], dtype=torch.bool, device=device)
        if valid_mask is None
        else valid_mask.to(device=device, dtype=torch.bool)
    )
    if valid.shape != features.shape[:2] or not valid.any(dim=1).all():
        raise ValueError("Mask must match [B,T] and contain valid tokens in every sequence.")
    oracle = model.student_encoder_attention == "clip_causal" or model.rope_mode == "2d_clip_frame"
    if oracle and segment_lengths is None:
        raise ValueError(
            "This historical oracle model requires segment_lengths; use block-causal for boundary-free inference."
        )
    if segment_lengths is not None:
        segment_lengths = segment_lengths.to(device=device, dtype=torch.long)
        if (
            segment_lengths.ndim != 2
            or segment_lengths.shape[0] != features.shape[0]
            or (segment_lengths < 0).any()
            or not torch.equal(segment_lengths.sum(dim=1), valid.sum(dim=1))
        ):
            raise ValueError(
                "Segment lengths must be nonnegative [B,S] and sum to valid token counts."
            )
    model.eval()
    return model.student_enc(
        features.unsqueeze(1),
        valid_mask=valid.unsqueeze(1),
        context_mask=valid.unsqueeze(1),
        N=1,
        L=features.shape[1],
        segment_lengths=segment_lengths,
    )
