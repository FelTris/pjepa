from types import SimpleNamespace
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pjepa.probes.ltcontext_probe import _merge_dicts


def _segment_ids_from_lengths(
    segment_lengths: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    if segment_lengths is None:
        raise ValueError(
            "CausalLTContextProbe requires segment_lengths for segment-causal attention."
        )
    if segment_lengths.dim() != 2:
        raise ValueError(
            f"Expected segment_lengths with shape [B, S], got {tuple(segment_lengths.shape)}"
        )
    if valid_mask.dim() != 2:
        raise ValueError(f"Expected valid_mask with shape [B, T], got {tuple(valid_mask.shape)}")

    B, T = valid_mask.shape
    segment_ids = torch.full((B, T), -1, dtype=torch.long, device=valid_mask.device)
    for b in range(B):
        offset = 0
        seg_idx = 0
        for raw_len in segment_lengths[b].tolist():
            seg_len = int(raw_len)
            if seg_len <= 0:
                continue
            if offset >= T:
                break
            end = min(T, offset + seg_len)
            segment_ids[b, offset:end] = seg_idx
            offset = end
            seg_idx += 1
    segment_ids = segment_ids.masked_fill(~valid_mask.bool(), -1)
    return segment_ids


def _block_ids_from_positions(
    valid_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Return fixed block IDs aligned to position zero in every take."""
    if valid_mask.dim() != 2:
        raise ValueError(f"Expected valid_mask with shape [B, T], got {tuple(valid_mask.shape)}")
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    positions = torch.arange(valid_mask.shape[1], device=valid_mask.device)
    block_ids = positions.div(block_size, rounding_mode="floor")
    block_ids = block_ids.unsqueeze(0).expand(valid_mask.shape[0], -1).clone()
    return block_ids.masked_fill(~valid_mask.bool(), -1)


class CausalMultiHeadAttention1d(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if model_dim % int(num_heads) != 0:
            raise ValueError(f"model_dim={model_dim} must be divisible by num_heads={num_heads}")
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        inner_dim = self.model_dim // 2
        if inner_dim % self.num_heads != 0:
            inner_dim = self.model_dim
        self.inner_dim = int(inner_dim)
        self.head_dim = self.inner_dim // self.num_heads

        self.proj_q = nn.Conv1d(self.model_dim, self.inner_dim, kernel_size=1, bias=True)
        self.proj_k = nn.Conv1d(self.model_dim, self.inner_dim, kernel_size=1, bias=True)
        self.proj_v = nn.Conv1d(self.model_dim, self.inner_dim, kernel_size=1, bias=True)
        self.out_proj = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(self.inner_dim, self.model_dim, kernel_size=1, bias=True),
        )
        self.dropout = float(dropout)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        B, D, T = x.shape
        x = x.view(B, self.num_heads, self.head_dim, T)
        return x.transpose(-2, -1).contiguous()

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor, allowed_mask: torch.Tensor
    ) -> torch.Tensor:
        if allowed_mask.dim() != 3:
            raise ValueError(
                f"Expected allowed_mask with shape [B, Q, K], got {tuple(allowed_mask.shape)}"
            )
        if allowed_mask.shape[0] != query.shape[0]:
            raise ValueError("allowed_mask batch dimension must match query batch dimension.")
        if allowed_mask.shape[1] != query.shape[-1] or allowed_mask.shape[2] != key_value.shape[-1]:
            raise ValueError(
                "allowed_mask must match query/key lengths: "
                f"mask={tuple(allowed_mask.shape)}, query={tuple(query.shape)}, key_value={tuple(key_value.shape)}"
            )
        q = self._heads(self.proj_q(query))
        k = self._heads(self.proj_k(key_value))
        v = self._heads(self.proj_v(key_value))
        no_keys = ~allowed_mask.any(dim=-1)
        if no_keys.any():
            safe_mask = allowed_mask.clone()
            safe_mask[:, :, 0] |= no_keys
            allowed_mask = safe_mask
        mask = allowed_mask[:, None, :, :].to(device=q.device, dtype=torch.bool)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = (
            out.transpose(-2, -1).contiguous().view(query.shape[0], self.inner_dim, query.shape[-1])
        )
        return self.out_proj(out)


def _base_segment_causal_mask(
    query_segment_ids: torch.Tensor,
    key_segment_ids: torch.Tensor,
    query_valid_mask: torch.Tensor,
    key_valid_mask: torch.Tensor,
) -> torch.Tensor:
    query_seg = query_segment_ids[:, :, None]
    key_seg = key_segment_ids[:, None, :]
    valid_q = query_valid_mask[:, :, None].bool()
    valid_k = key_valid_mask[:, None, :].bool()
    return valid_q & valid_k & (query_seg >= 0) & (key_seg >= 0) & (key_seg <= query_seg)


class CausalWindowedAttention(nn.Module):
    def __init__(
        self,
        windowed_attn_w: int,
        model_dim: int,
        num_heads: int,
        dropout: float,
        max_parallel_chunks: int = 1,
    ):
        super().__init__()
        self.windowed_attn_w = max(1, int(windowed_attn_w))
        self.max_parallel_chunks = max(1, int(max_parallel_chunks))
        self.attn = CausalMultiHeadAttention1d(model_dim, num_heads, dropout)

    def forward(
        self,
        qk: torch.Tensor,
        v: Optional[torch.Tensor],
        valid_mask: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        T = qk.shape[-1]
        value = qk if v is None else v
        B, D, _ = qk.shape
        out = qk.new_zeros(qk.shape)
        block = self.windowed_attn_w
        specs = []
        for q_start in range(0, T, block):
            q_end = min(T, q_start + block)
            k_start = max(0, q_start - block)
            k_end = min(T, q_end + block)
            specs.append((q_start, q_end, k_start, k_end))

        for group_start in range(0, len(specs), self.max_parallel_chunks):
            group = specs[group_start : group_start + self.max_parallel_chunks]
            q_max = max(q_end - q_start for q_start, q_end, _, _ in group)
            k_max = max(k_end - k_start for _, _, k_start, k_end in group)
            q_group = qk.new_zeros(B * len(group), D, q_max)
            kv_group = value.new_zeros(B * len(group), D, k_max)
            allowed_group = torch.zeros(
                B * len(group), q_max, k_max, dtype=torch.bool, device=qk.device
            )

            for chunk_idx, (q_start, q_end, k_start, k_end) in enumerate(group):
                q_len = q_end - q_start
                k_len = k_end - k_start
                row = slice(chunk_idx * B, (chunk_idx + 1) * B)
                q_group[row, :, :q_len] = qk[:, :, q_start:q_end]
                kv_group[row, :, :k_len] = value[:, :, k_start:k_end]

                q_pos = torch.arange(q_start, q_end, device=qk.device)
                k_pos = torch.arange(k_start, k_end, device=qk.device)
                local = (k_pos[None, :] - q_pos[:, None]).abs() <= self.windowed_attn_w
                allowed = (
                    _base_segment_causal_mask(
                        segment_ids[:, q_start:q_end],
                        segment_ids[:, k_start:k_end],
                        valid_mask[:, q_start:q_end],
                        valid_mask[:, k_start:k_end],
                    )
                    & local[None, :, :]
                )
                allowed_group[row, :q_len, :k_len] = allowed

            group_out = self.attn(q_group, kv_group, allowed_group)
            for chunk_idx, (q_start, q_end, _, _) in enumerate(group):
                q_len = q_end - q_start
                row = slice(chunk_idx * B, (chunk_idx + 1) * B)
                out[:, :, q_start:q_end] = group_out[row, :, :q_len]
        return out * valid_mask[:, None, :].to(qk.dtype)


class CausalLTContextAttention(nn.Module):
    def __init__(
        self,
        long_term_attn_g: int,
        model_dim: int,
        num_heads: int,
        dropout: float,
        max_parallel_chunks: int = 1,
    ):
        super().__init__()
        self.long_term_attn_g = max(1, int(long_term_attn_g))
        self.max_parallel_chunks = max(1, int(max_parallel_chunks))
        self.attn = CausalMultiHeadAttention1d(model_dim, num_heads, dropout)

    def forward(
        self,
        qk: torch.Tensor,
        v: Optional[torch.Tensor],
        valid_mask: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        T = qk.shape[-1]
        value = qk if v is None else v
        B, D, _ = qk.shape
        out = qk.new_zeros(qk.shape)
        residue_indices = [
            torch.arange(residue, T, self.long_term_attn_g, device=qk.device)
            for residue in range(min(self.long_term_attn_g, T))
        ]
        residue_indices = [idx for idx in residue_indices if idx.numel() > 0]

        for group_start in range(0, len(residue_indices), self.max_parallel_chunks):
            group = residue_indices[group_start : group_start + self.max_parallel_chunks]
            q_max = max(int(idx.numel()) for idx in group)
            q_group = qk.new_zeros(B * len(group), D, q_max)
            kv_group = value.new_zeros(B * len(group), D, q_max)
            allowed_group = torch.zeros(
                B * len(group), q_max, q_max, dtype=torch.bool, device=qk.device
            )

            for chunk_idx, idx in enumerate(group):
                q_len = int(idx.numel())
                row = slice(chunk_idx * B, (chunk_idx + 1) * B)
                q_group[row, :, :q_len] = qk.index_select(-1, idx)
                kv_group[row, :, :q_len] = value.index_select(-1, idx)
                allowed_group[row, :q_len, :q_len] = _base_segment_causal_mask(
                    segment_ids.index_select(1, idx),
                    segment_ids.index_select(1, idx),
                    valid_mask.index_select(1, idx),
                    valid_mask.index_select(1, idx),
                )

            group_out = self.attn(q_group, kv_group, allowed_group)
            for chunk_idx, idx in enumerate(group):
                q_len = int(idx.numel())
                row = slice(chunk_idx * B, (chunk_idx + 1) * B)
                out.index_copy_(-1, idx, group_out[row, :, :q_len])
        return out * valid_mask[:, None, :].to(qk.dtype)


class CausalDilatedConv(nn.Module):
    def __init__(self, n_channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.left_pad = int(dilation) * (int(kernel_size) - 1)
        self.dilated_conv = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=int(dilation),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return self.activation(self.dilated_conv(x)) * mask


class CausalLTCBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        dilation: int,
        windowed_attn_w: int,
        long_term_attn_g: int,
        num_heads: int,
        attention_dropout: float,
        max_parallel_chunks: int,
        use_instance_norm: bool,
        dropout_prob: float,
    ):
        super().__init__()
        self.dilated_conv = CausalDilatedConv(model_dim, dilation=dilation, kernel_size=3)
        # Preserve the original inverted flag for pretrained checkpoint compatibility.
        # InstanceNorm uses all padded time steps; see docs/protocols.md before
        # interpreting this legacy head as strictly causal or padding-invariant.
        self.instance_norm = nn.Identity() if use_instance_norm else nn.InstanceNorm1d(model_dim)
        self.windowed_attn = CausalWindowedAttention(
            windowed_attn_w, model_dim, num_heads, attention_dropout, max_parallel_chunks
        )
        self.ltc_attn = CausalLTContextAttention(
            long_term_attn_g, model_dim, num_heads, attention_dropout, max_parallel_chunks
        )
        self.out_linear = nn.Conv1d(model_dim, model_dim, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        segment_ids: torch.Tensor,
        prev_stage_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        valid_mask = mask[:, 0, :].bool()
        out = self.dilated_conv(inputs, mask)
        out = (
            self.windowed_attn(self.instance_norm(out), prev_stage_feat, valid_mask, segment_ids)
            + out
        )
        out = self.ltc_attn(self.instance_norm(out), prev_stage_feat, valid_mask, segment_ids) + out
        out = self.dropout(self.out_linear(out))
        return (out + inputs) * mask


class CausalLTCModule(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        model_dim: int,
        num_classes: int,
        dilation_factor: int,
        windowed_attn_w: int,
        long_term_attn_g: int,
        num_heads: int,
        attention_dropout: float,
        max_parallel_chunks: int,
        use_instance_norm: bool,
        dropout_prob: float,
        channel_dropout_prob: float,
    ):
        super().__init__()
        self.channel_dropout = nn.Dropout1d(channel_dropout_prob)
        self.input_proj = nn.Conv1d(input_dim, model_dim, kernel_size=1, bias=True)
        self.layers = nn.ModuleList(
            [
                CausalLTCBlock(
                    model_dim=model_dim,
                    dilation=int(dilation_factor) ** i,
                    windowed_attn_w=windowed_attn_w,
                    long_term_attn_g=long_term_attn_g,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    max_parallel_chunks=max_parallel_chunks,
                    use_instance_norm=use_instance_norm,
                    dropout_prob=dropout_prob,
                )
                for i in range(int(num_layers))
            ]
        )
        self.out_proj = nn.Conv1d(model_dim, num_classes, kernel_size=1, bias=True)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        segment_ids: torch.Tensor,
        prev_stage_feat: Optional[torch.Tensor] = None,
    ):
        feature = self.input_proj(self.channel_dropout(inputs))
        feature = feature * mask
        for layer in self.layers:
            feature = layer(feature, mask, segment_ids, prev_stage_feat)
        out = self.out_proj(feature) * mask
        return out, feature


class CausalLTC(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        ltc_cfg = model_cfg.LTC
        attn_cfg = model_cfg.ATTENTION
        self.stage1 = CausalLTCModule(
            num_layers=ltc_cfg.NUM_LAYERS,
            input_dim=model_cfg.INPUT_DIM,
            model_dim=ltc_cfg.MODEL_DIM,
            num_classes=model_cfg.NUM_CLASSES,
            dilation_factor=ltc_cfg.CONV_DILATION_FACTOR,
            windowed_attn_w=ltc_cfg.WINDOWED_ATTN_W,
            long_term_attn_g=ltc_cfg.LONG_TERM_ATTN_G,
            num_heads=attn_cfg.NUM_ATTN_HEADS,
            attention_dropout=attn_cfg.DROPOUT,
            max_parallel_chunks=ltc_cfg.MAX_PARALLEL_CHUNKS,
            use_instance_norm=ltc_cfg.USE_INSTANCE_NORM,
            dropout_prob=ltc_cfg.DROPOUT_PROB,
            channel_dropout_prob=ltc_cfg.CHANNEL_MASKING_PROB,
        )
        reduced_dim = int(ltc_cfg.MODEL_DIM // ltc_cfg.DIM_REDUCTION)
        self.dim_reduction = nn.Conv1d(ltc_cfg.MODEL_DIM, reduced_dim, kernel_size=1, bias=True)
        self.stages = nn.ModuleList(
            [
                CausalLTCModule(
                    num_layers=ltc_cfg.NUM_LAYERS,
                    input_dim=model_cfg.NUM_CLASSES,
                    model_dim=reduced_dim,
                    num_classes=model_cfg.NUM_CLASSES,
                    dilation_factor=ltc_cfg.CONV_DILATION_FACTOR,
                    windowed_attn_w=ltc_cfg.WINDOWED_ATTN_W,
                    long_term_attn_g=ltc_cfg.LONG_TERM_ATTN_G,
                    num_heads=attn_cfg.NUM_ATTN_HEADS,
                    attention_dropout=attn_cfg.DROPOUT,
                    max_parallel_chunks=ltc_cfg.MAX_PARALLEL_CHUNKS,
                    use_instance_norm=ltc_cfg.USE_INSTANCE_NORM,
                    dropout_prob=ltc_cfg.DROPOUT_PROB,
                    channel_dropout_prob=ltc_cfg.CHANNEL_MASKING_PROB,
                )
                for _ in range(1, ltc_cfg.NUM_STAGES)
            ]
        )

    def forward(
        self, inputs: torch.Tensor, masks: torch.Tensor, segment_ids: torch.Tensor
    ) -> torch.Tensor:
        out, feature = self.stage1(inputs, masks, segment_ids)
        output_list = [out]
        feature = self.dim_reduction(feature)
        for stage in self.stages:
            out, feature = stage(
                F.softmax(out, dim=1) * masks,
                masks,
                segment_ids,
                prev_stage_feat=feature * masks,
            )
            output_list.append(out)
        return torch.stack(output_list, dim=0)


class CausalLTContextProbe(nn.Module):
    """Segment- or fixed-block-causal LTContext for flat take sequences."""

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
                "max_parallel_chunks": 1,
                "ce_loss_weight": 1.0,
                "mse_loss_weight": 0.15,
                "mse_loss_clip_val": 16.0,
                "causal_attention_mode": "segment_causal",
                "causal_block_size": 64,
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
        self.causal_attention_mode = str(cfg["causal_attention_mode"]).lower()
        if self.causal_attention_mode not in {"segment_causal", "block_causal"}:
            raise ValueError(
                "causal_attention_mode must be 'segment_causal' or "
                f"'block_causal', got {self.causal_attention_mode!r}."
            )
        self.causal_block_size = int(cfg["causal_block_size"])
        if self.causal_block_size <= 0:
            raise ValueError("causal_block_size must be positive.")

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
                MAX_PARALLEL_CHUNKS=int(cfg["max_parallel_chunks"]),
            ),
            ATTENTION=SimpleNamespace(
                NUM_ATTN_HEADS=int(cfg["num_attn_heads"]),
                DROPOUT=float(cfg["attention_dropout"]),
            ),
        )
        self.model = CausalLTC(model_cfg)

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        segment_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError(f"Expected features with shape [B, T, D], got {tuple(features.shape)}")
        if valid_mask.shape != features.shape[:2]:
            raise ValueError(
                f"Expected valid_mask with shape [B, T], got {tuple(valid_mask.shape)} for features {tuple(features.shape)}"
            )
        valid_mask = valid_mask.bool()
        if self.causal_attention_mode == "segment_causal":
            if segment_lengths is None:
                raise ValueError("segment_causal LTContext requires ground-truth segment_lengths.")
            segment_ids = _segment_ids_from_lengths(
                segment_lengths.to(valid_mask.device), valid_mask
            )
        else:
            segment_ids = _block_ids_from_positions(valid_mask, block_size=self.causal_block_size)
        masks = valid_mask.unsqueeze(1)
        return self.model(features.movedim(-1, 1), masks, segment_ids)

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
