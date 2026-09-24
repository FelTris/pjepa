"""Configurable rotary positions for the feature-level P-JEPA model.

The upstream feature-JEPA implementation uses segment-aware 2D RoPE and also
reuses its clip-axis positions to build the clip-causal attention mask.  This
module keeps those two concerns separate so that flattened 1D RoPE can be
ablated without changing the attention pattern.
"""

from typing import Optional

import torch
import torch.nn as nn

from pjepa.models import feature_base as upstream


VALID_ROPE_MODES = {"2d_clip_frame", "1d_flat"}


def normalize_rope_mode(rope_mode: str) -> str:
    mode = str(rope_mode).strip().lower().replace("-", "_")
    aliases = {
        "2d": "2d_clip_frame",
        "clip_frame": "2d_clip_frame",
        "1d": "1d_flat",
        "flat": "1d_flat",
        "flat_1d": "1d_flat",
    }
    mode = aliases.get(mode, mode)
    if mode not in VALID_ROPE_MODES:
        raise ValueError(
            f"Unsupported rope_mode='{rope_mode}'. Expected one of {sorted(VALID_ROPE_MODES)}."
        )
    return mode


def build_flat_rope_positions(T: int, device, batch: int = 1, heads: int = 1) -> torch.Tensor:
    """Return absolute token positions [0, ..., T-1], broadcast over B and H."""
    return torch.arange(T, device=device, dtype=torch.long).view(1, 1, T).expand(batch, heads, T)


def _build_segment_positions(
    *,
    segment_lengths: Optional[torch.Tensor],
    N: int,
    L: int,
    device,
    batch: int,
    heads: int,
):
    if segment_lengths is not None:
        return upstream._build_rope_pos_from_segment_lengths(
            segment_lengths=segment_lengths.to(device=device),
            T=N * L,
            device=device,
            heads=heads,
        )
    return upstream._build_rope_pos_NL(N, L, device=device, batch=batch, heads=heads)


class FeatureEncoder(upstream.FeatureEncoder):
    def __init__(
        self,
        *args,
        rope_mode: str = "2d_clip_frame",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rope_mode = normalize_rope_mode(rope_mode)

    def forward(
        self,
        x4d: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
        context_mask: torch.Tensor,
        N: int,
        L: int,
        segment_lengths: Optional[torch.Tensor] = None,
        position_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N_in, L_in, _ = x4d.shape
        assert N_in == N and L_in == L
        T = N * L

        x = self.proj(x4d.reshape(B, T, -1))
        ctx_only = (context_mask & valid_mask).reshape(B, T)

        # Structural clip ids are used only by clip-causal attention.  They are
        # intentionally independent of the positions supplied to RoPE.
        clip_ids, frame_ids = _build_segment_positions(
            segment_lengths=segment_lengths,
            N=N,
            L=L,
            device=x.device,
            batch=B,
            heads=self.n_heads,
        )

        if self.rope_mode == "1d_flat":
            flat_positions = build_flat_rope_positions(T, x.device, batch=B, heads=self.n_heads)
            if position_offsets is not None:
                offsets = torch.as_tensor(
                    position_offsets, device=x.device, dtype=torch.long
                ).reshape(B, 1, 1)
                flat_positions = flat_positions + offsets
            # The upstream attention splits each head into two rotary chunks.
            # Supplying the same absolute position to both chunks rotates the
            # full head using token time, with no clip-boundary contribution.
            rope_n = flat_positions
            rope_l = flat_positions
        else:
            if position_offsets is not None:
                raise ValueError("position_offsets is only supported with 1d_flat RoPE.")
            rope_n = clip_ids
            rope_l = frame_ids

        packed_x, packed_rn, packed_rl, _, valid_key_mask = upstream._pack_by_mask_with_pos(
            x, ctx_only, rope_n, rope_l
        )

        if self.attention_mode == "clip_causal":
            packed_clip_ids = upstream._pack_by_mask_2d(clip_ids[:, 0, :], ctx_only)
            attn_mask = upstream._build_clip_causal_mask(valid_key_mask, packed_clip_ids)
        elif self.attention_mode == "block_causal":
            block_ids = torch.arange(T, device=x.device, dtype=torch.long) // self.block_causal_size
            block_ids = block_ids.view(1, -1).expand(B, -1)
            if position_offsets is not None:
                offsets = torch.as_tensor(
                    position_offsets, device=x.device, dtype=torch.long
                ).reshape(B, 1)
                if bool((offsets % self.block_causal_size != 0).any().item()):
                    raise ValueError(
                        "Block-causal position offsets must align to block boundaries."
                    )
                block_ids = block_ids + offsets // self.block_causal_size
            packed_block_ids = upstream._pack_by_mask_2d(block_ids, ctx_only)
            attn_mask = upstream._build_block_causal_mask(valid_key_mask, packed_block_ids)
        else:
            attn_mask = upstream._broadcast_mask_for_sdpa(valid_key_mask)

        h = packed_x
        attn_mask = ~attn_mask
        for blk in self.blocks:
            h = blk(h, attn_mask=attn_mask, rope_n=packed_rn, rope_l=packed_rl)
        h = self.norm(h)
        return upstream._scatter_back(T, h, ctx_only)


class CausalPredictor(upstream.CausalPredictor):
    def __init__(self, *args, rope_mode: str = "2d_clip_frame", **kwargs):
        super().__init__(*args, **kwargs)
        self.rope_mode = normalize_rope_mode(rope_mode)

    def forward(
        self,
        x: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
        N: int,
        L: int,
        segment_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        assert T == N * L
        attn_mask = ~upstream._broadcast_mask_for_sdpa(valid_mask)

        if self.rope_mode == "1d_flat":
            flat_positions = build_flat_rope_positions(T, x.device, batch=B, heads=self.n_heads)
            rope_n = flat_positions
            rope_l = flat_positions
        else:
            rope_n, rope_l = _build_segment_positions(
                segment_lengths=segment_lengths,
                N=N,
                L=L,
                device=x.device,
                batch=B,
                heads=self.n_heads,
            )

        h = x
        for blk in self.blocks:
            h = blk(h, attn_mask=attn_mask, rope_n=rope_n, rope_l=rope_l)
        return self.norm(h)


class FeatureJEPA(upstream.FeatureJEPA):
    def __init__(
        self,
        d_in: int = 1024,
        d_model: int = 768,
        enc_depth: int = 12,
        enc_heads: int = 16,
        pred_depth: int = 8,
        pred_heads: int = 16,
        drop_path_prob: float = 0.0,
        student_encoder_attention: str = "bidirectional",
        student_encoder_block_size: int = 16,
        rope_mode: str = "2d_clip_frame",
    ):
        nn.Module.__init__(self)
        self.rope_mode = normalize_rope_mode(rope_mode)
        self.student_encoder_attention = upstream._normalize_encoder_attention_mode(
            student_encoder_attention
        )
        self.student_encoder_block_size = max(1, int(student_encoder_block_size))
        self.student_enc = FeatureEncoder(
            d_in,
            d_model,
            enc_depth,
            enc_heads,
            drop_path_prob,
            attention_mode=self.student_encoder_attention,
            block_causal_size=self.student_encoder_block_size,
            rope_mode=self.rope_mode,
        )
        self.predictor = CausalPredictor(
            d_model,
            pred_depth,
            pred_heads,
            drop_path_prob,
            rope_mode=self.rope_mode,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    @torch.no_grad()
    def build_teacher(self) -> nn.Module:
        teacher = FeatureEncoder(
            d_in=self.student_enc.d_in,
            d_model=self.student_enc.d_model,
            depth=len(self.student_enc.blocks),
            n_heads=self.student_enc.blocks[0].attn.num_heads,
            drop_path_prob=0.0,
            attention_mode=self.student_encoder_attention,
            block_causal_size=self.student_encoder_block_size,
            rope_mode=self.rope_mode,
        )
        teacher.load_state_dict(self.student_enc.state_dict(), strict=True)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        return teacher
