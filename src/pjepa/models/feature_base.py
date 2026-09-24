from typing import Dict, Optional
import torch
import torch.nn as nn

from pjepa.models.modules_2d import Block, ACBlock


VALID_ENCODER_ATTENTION_MODES = {"bidirectional", "causal", "clip_causal", "block_causal"}


# -----------------------------
# Small helpers
# -----------------------------


def _build_rope_pos_NL(N: int, L: int, device, batch: int = 1, heads: int = 1):
    # flatten order is clip-major, then frames within a clip
    clip_ids = torch.arange(N, device=device).repeat_interleave(L)  # [N*L]
    frame_ids = torch.arange(L, device=device).repeat(N)  # [N*L]
    # broadcast to [B,H,T]
    clip_ids = clip_ids.view(1, 1, -1).expand(batch, heads, -1)
    frame_ids = frame_ids.view(1, 1, -1).expand(batch, heads, -1)
    return clip_ids, frame_ids


def _build_rope_pos_from_segment_lengths(
    segment_lengths: torch.Tensor,
    T: int,
    device,
    heads: int = 1,
):
    """
    Build per-token clip ids and local token ids from padded segment lengths.

    segment_lengths: [B, S] where zeros indicate padded segment slots
    returns:
      clip_ids:  [B, H, T]
      frame_ids: [B, H, T]
    """
    if segment_lengths.dim() != 2:
        raise ValueError(
            f"segment_lengths must have shape [B, S], got {tuple(segment_lengths.shape)}"
        )

    B, _ = segment_lengths.shape
    clip_ids = torch.zeros(B, T, dtype=torch.long, device=device)
    frame_ids = torch.zeros(B, T, dtype=torch.long, device=device)

    for b in range(B):
        offset = 0
        clip_idx = 0
        for seg_len in segment_lengths[b].tolist():
            seg_len = int(seg_len)
            if seg_len <= 0:
                break
            if offset >= T:
                break
            seg_len = min(seg_len, T - offset)
            clip_ids[b, offset : offset + seg_len] = clip_idx
            frame_ids[b, offset : offset + seg_len] = torch.arange(
                seg_len, device=device, dtype=torch.long
            )
            offset += seg_len
            clip_idx += 1

    clip_ids = clip_ids.view(B, 1, T).expand(B, heads, T)
    frame_ids = frame_ids.view(B, 1, T).expand(B, heads, T)
    return clip_ids, frame_ids


def _pack_by_mask_with_pos(
    x: torch.Tensor, keep_mask: torch.Tensor, rope_n: torch.Tensor, rope_l: torch.Tensor
):
    """
    x:       [B, T, D]
    keep_mask: [B, T] boolean (True keeps)
    rope_n/l: [B, H, T] (we keep H dimension but we don't need to change it)
    Returns: packed_x [B, Tmax, D], packed_rope_n/l [B, H, Tmax], lengths [B]
    """
    B, T, D = x.shape
    H = rope_n.shape[1]
    outs_x, outs_rn, outs_rl, lens = [], [], [], []
    for b in range(B):
        idx = keep_mask[b].nonzero(as_tuple=False).squeeze(-1)
        xb = x[b, idx]  # [Tb, D]
        rn = rope_n[b, :, idx]  # [H, Tb]
        rl = rope_l[b, :, idx]  # [H, Tb]
        outs_x.append(xb)
        outs_rn.append(rn)
        outs_rl.append(rl)
        lens.append(idx.numel())
    Lmax = max(lens) if lens else 0
    packed_x = x.new_zeros(B, Lmax, D)
    packed_rn = rope_n.new_zeros(B, H, Lmax)
    packed_rl = rope_l.new_zeros(B, H, Lmax)
    lengths = torch.tensor(lens, device=x.device, dtype=torch.long)
    for b in range(B):
        Lb = lens[b]
        if Lb > 0:
            packed_x[b, :Lb] = outs_x[b]
            packed_rn[b, :, :Lb] = outs_rn[b]
            packed_rl[b, :, :Lb] = outs_rl[b]
    # valid-key mask for SDPA
    valid_key_mask = torch.arange(Lmax, device=x.device).unsqueeze(0) < lengths.unsqueeze(
        1
    )  # [B, Lmax]
    return packed_x, packed_rn, packed_rl, lengths, valid_key_mask


def _pack_by_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pack tokens by boolean mask per batch sample.
    x: (B, N, D), mask: (B, N) -> True keeps the token.
    Returns packed (B, N_keep_max, D) padded with zeros; also returns lengths & indices if needed.
    We’ll use a simpler per-sample gather + pad since N differs per sample.
    """
    B, N, D = x.shape
    outs = []
    for b in range(B):
        xb = x[b]
        mb = mask[b]
        outs.append(xb[mb])  # (N_keep_b, D)
    # pad to max length
    max_len = max(o.shape[0] for o in outs)
    packed = x.new_zeros(B, max_len, D)
    lengths = torch.tensor([o.shape[0] for o in outs], device=x.device)
    for b, o in enumerate(outs):
        packed[b, : o.shape[0]] = o
    return packed, lengths


def _scatter_back(full_len: int, packed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Scatter a packed (B, L, D) back to (B, N, D) using mask positions; zeros elsewhere."""
    B, N = mask.shape
    D = packed.shape[-1]
    out = packed.new_zeros(B, full_len, D)
    for b in range(B):
        idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
        L = idx.numel()
        out[b, idx] = packed[b, :L]
    return out


def _broadcast_mask_for_sdpa(valid_key_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Return an SDPA-compatible boolean attn_mask of shape (B, 1, Nq, Nk)
    where True means "allow". Here we allow valid **keys** only.
    """
    if valid_key_mask is None:
        return None
    B, N = valid_key_mask.shape
    # True where key is a PAD (invalid)
    key_is_pad = ~valid_key_mask  # (B, N)
    attn_mask = key_is_pad[:, None, None, :].expand(B, 1, N, N).contiguous()
    return attn_mask  # bool, True = blocked


def _normalize_encoder_attention_mode(attention_mode: str) -> str:
    mode = str(attention_mode).strip().lower().replace("-", "_")
    aliases = {
        "noncausal": "bidirectional",
        "non_causal": "bidirectional",
        "full_causal": "causal",
        "clipcausal": "clip_causal",
        "blockcausal": "block_causal",
    }
    mode = aliases.get(mode, mode)
    if mode not in VALID_ENCODER_ATTENTION_MODES:
        raise ValueError(
            f"Unsupported encoder attention mode '{attention_mode}'. "
            f"Expected one of {sorted(VALID_ENCODER_ATTENTION_MODES)}."
        )
    return mode


def _build_clip_causal_mask(valid_key_mask: torch.Tensor, clip_ids: torch.Tensor) -> torch.Tensor:
    query_clips = clip_ids[:, :, None]
    key_clips = clip_ids[:, None, :]
    allow = valid_key_mask[:, None, :] & (key_clips <= query_clips)
    return (~allow)[:, None, :, :].contiguous()


def _build_block_causal_mask(valid_key_mask: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
    query_blocks = block_ids[:, :, None]
    key_blocks = block_ids[:, None, :]
    allow = valid_key_mask[:, None, :] & (key_blocks <= query_blocks)
    return (~allow)[:, None, :, :].contiguous()


def _pack_by_mask_2d(values: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    B, T = values.shape
    if keep_mask.shape != (B, T):
        raise ValueError(f"Expected keep_mask with shape {(B, T)}, got {tuple(keep_mask.shape)}")
    lengths = keep_mask.sum(dim=1)
    Lmax = int(lengths.max().item()) if lengths.numel() > 0 else 0
    packed = values.new_zeros(B, Lmax)
    for b in range(B):
        idx = keep_mask[b].nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() > 0:
            packed[b, : idx.numel()] = values[b, idx]
    return packed


# -----------------------------
# Encoder (bidirectional, context-only)
# -----------------------------
class FeatureEncoder(nn.Module):
    """Transformer over feature tokens using the baseline Block.

    We feed **context tokens only** through the transformer to match JEPA.
    After encoding, we scatter them back to the original positions (zeros elsewhere).
    """

    def __init__(
        self,
        d_in: int = 1024,
        d_model: int = 1024,
        depth: int = 12,
        n_heads: int = 16,
        drop_path_prob: float = 0.0,
        attention_mode: str = "bidirectional",
        block_causal_size: int = 16,
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.d_model = int(d_model)
        # When upstream features already match the model width, preserve their geometry
        # instead of forcing a learned LayerNorm+Linear remapping.
        self.proj = nn.Sequential(nn.LayerNorm(self.d_in), nn.Linear(self.d_in, self.d_model))
        self.n_heads = n_heads
        self.attention_mode = _normalize_encoder_attention_mode(attention_mode)
        self.block_causal_size = max(1, int(block_causal_size))
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=d_model,
                    num_heads=n_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=drop_path_prob,
                    use_sdpa=True,
                    is_causal=(self.attention_mode == "causal"),
                    use_rope=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x4d: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
        context_mask: torch.Tensor,
        N: int,
        L: int,
        segment_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x4d: [B, N, L, d_in]
        valid_mask:   [B, N, L]  True for real tokens
        context_mask: [B, N, L]  True for tokens visible to the encoder
        returns: [B, N*L, d_model] with encoded features at context positions; zeros elsewhere
        """
        B, N_in, L_in, _ = x4d.shape
        assert N_in == N and L_in == L
        x = x4d.reshape(B, N * L, -1)  # flatten to [B, T, d_in]
        x = self.proj(x)

        valid_mask.reshape(B, N * L)
        ctx_only = (context_mask & valid_mask).reshape(B, N * L)

        # 2D RoPE positions for the full (flattened) sequence
        if segment_lengths is not None:
            rope_n, rope_l = _build_rope_pos_from_segment_lengths(
                segment_lengths=segment_lengths.to(device=x.device),
                T=N * L,
                device=x.device,
                heads=self.n_heads,
            )
        else:
            rope_n, rope_l = _build_rope_pos_NL(N, L, device=x.device, batch=B, heads=self.n_heads)

        # pack context tokens + carry RoPE positions through the packing
        packed_x, packed_rn, packed_rl, lengths, valid_key_mask = _pack_by_mask_with_pos(
            x, ctx_only, rope_n, rope_l
        )
        if self.attention_mode == "clip_causal":
            attn_mask = _build_clip_causal_mask(valid_key_mask, packed_rn[:, 0, :])
        elif self.attention_mode == "block_causal":
            block_ids = (
                torch.arange(N * L, device=x.device, dtype=torch.long) // self.block_causal_size
            )
            block_ids = block_ids.view(1, -1).expand(B, -1)
            packed_block_ids = _pack_by_mask_2d(block_ids, ctx_only)
            attn_mask = _build_block_causal_mask(valid_key_mask, packed_block_ids)
        else:
            attn_mask = _broadcast_mask_for_sdpa(
                valid_key_mask
            )  # mask padded keys created by packing
        # print(attn_mask[0,0,0,:32])
        h = packed_x
        attn_mask = ~attn_mask
        # print(attn_mask[0,0,0,:32])
        for blk in self.blocks:
            h = blk(h, attn_mask=attn_mask, rope_n=packed_rn, rope_l=packed_rl)
        h = self.norm(h)

        # scatter back to full flattened length (zeros on non-context positions)
        enc_full = _scatter_back(N * L, h, ctx_only)  # [B, N*L, d_model]
        return enc_full


# -----------------------------
# Predictor (causal over full sequence)
# -----------------------------
class CausalPredictor(nn.Module):
    """Causal transformer over full sequence (context encodings + mask tokens)."""

    def __init__(
        self, d_model: int = 1024, depth: int = 8, n_heads: int = 16, drop_path_prob: float = 0.0
    ):
        super().__init__()
        self.n_heads = n_heads
        self.blocks = nn.ModuleList(
            [
                ACBlock(
                    dim=d_model,
                    num_heads=n_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=drop_path_prob,
                    use_sdpa=True,
                    is_causal=True,  # <- causal inside SDPA
                    use_rope=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
        N: int,
        L: int,
        segment_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, N*L, d_model]
        valid_mask: [B, N*L] True for real tokens (pads masked as keys)
        """
        B, T, _ = x.shape
        assert T == N * L
        attn_mask = _broadcast_mask_for_sdpa(valid_mask)  # mask-out PAD keys
        attn_mask = ~attn_mask

        # full-length RoPE positions
        if segment_lengths is not None:
            rope_n, rope_l = _build_rope_pos_from_segment_lengths(
                segment_lengths=segment_lengths.to(device=x.device),
                T=T,
                device=x.device,
                heads=self.n_heads,
            )
        else:
            rope_n, rope_l = _build_rope_pos_NL(N, L, device=x.device, batch=B, heads=self.n_heads)

        h = x
        for blk in self.blocks:
            h = blk(h, attn_mask=attn_mask, rope_n=rope_n, rope_l=rope_l)
        h = self.norm(h)
        return h


# -----------------------------
# Top-level wrapper (student/teacher/predictor)
# -----------------------------
class FeatureJEPA(nn.Module):
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
    ):
        super().__init__()
        self.student_encoder_attention = _normalize_encoder_attention_mode(
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
        )
        self.predictor = CausalPredictor(d_model, pred_depth, pred_heads, drop_path_prob)
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
        )
        teacher.load_state_dict(self.student_enc.state_dict(), strict=True)
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher

    def forward(
        self, batch: Dict[str, torch.Tensor], *, teacher: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Required batch keys:
          - x:            [B, N, L, d_in]
          - valid_mask:   [B, N, L]  True for real tokens
          - context_mask: [B, N, L]  True for encoder-visible tokens
          - target_mask:  [B, N, L]  True for tokens to predict
        """
        x = batch["x"]
        valid_mask_3d = batch["valid_mask"]
        context_mask_3d = batch["context_mask"]
        target_mask_3d = batch["target_mask"]
        segment_lengths = batch.get("segment_lengths")

        B, N, L, _ = x.shape
        T = N * L

        # 1) Student encoder on context-only (with RoPE 2D)
        enc_full = self.student_enc(
            x,
            valid_mask=valid_mask_3d,
            context_mask=context_mask_3d,
            N=N,
            L=L,
            segment_lengths=segment_lengths,
        )  # [B, T, d_model]

        # 2) Assemble predictor input: replace targets with mask_token
        target_mask = target_mask_3d.reshape(B, T)
        pred_in = enc_full.clone()
        mask_tok = self.mask_token.expand(B, T, pred_in.size(-1))
        pred_in = torch.where(target_mask.unsqueeze(-1), mask_tok, pred_in)

        # 3) Causal predictor over full sequence
        pred_out = self.predictor(
            pred_in,
            valid_mask=valid_mask_3d.reshape(B, T),
            N=N,
            L=L,
            segment_lengths=segment_lengths,
        )  # [B, T, d_model]

        # 4) Collect predictions at target ids
        preds = pred_out[target_mask]  # [num_masked, d_model]
        out = {
            "preds_at_targets": preds,
            "predictor_full": pred_out,
        }

        # 5) Teacher (bidirectional encoder, all valid tokens)
        if teacher is not None:
            with torch.no_grad():
                teacher_context = valid_mask_3d  # teacher sees all real tokens
                t_full = teacher(
                    x,
                    valid_mask=valid_mask_3d,
                    context_mask=teacher_context,
                    N=N,
                    L=L,
                    segment_lengths=segment_lengths,
                )
                t_targets = t_full[target_mask]
            out["teacher_at_targets"] = t_targets
            out["teacher_full"] = t_full

        return out
