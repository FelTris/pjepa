from __future__ import annotations


import torch


from pjepa.models.feature_vjepa_rope import FeatureJEPA  # noqa: E402
from pjepa.cli.surgical_phase_context_sweep import (  # noqa: E402
    block_window_specs,
    encode_with_context_limit,
)


def test_block_windows_cover_each_query_block_with_hard_context_cap() -> None:
    specs = block_window_specs(total_tokens=11, block_size=3, max_past_blocks=2)
    assert specs == [
        (0, 3, 0, 3),
        (0, 6, 3, 6),
        (0, 9, 6, 9),
        (3, 11, 6, 8),
    ]
    for start, _, target_start, _ in specs:
        assert target_start <= 2 * 3
        assert start % 3 == 0


@torch.no_grad()
def test_batched_context_windows_match_independent_window_encoding() -> None:
    torch.manual_seed(11)
    model = FeatureJEPA(
        d_in=32,
        d_model=32,
        enc_depth=2,
        enc_heads=4,
        pred_depth=1,
        pred_heads=4,
        drop_path_prob=0.0,
        student_encoder_attention="block_causal",
        student_encoder_block_size=3,
        rope_mode="1d_flat",
    ).eval()
    features = torch.randn(1, 11, 32)
    valid = torch.ones(1, 11, dtype=torch.bool)
    actual = encode_with_context_limit(
        model,
        features,
        valid,
        max_past_blocks=2,
        block_size=3,
        max_attention_pairs=1_000_000,
    )

    expected = torch.zeros_like(actual)
    for start, end, target_start, target_end in block_window_specs(11, 3, 2):
        window = features[:, start:end]
        mask = torch.ones(1, 1, end - start, dtype=torch.bool)
        encoded = model.student_enc(
            window.unsqueeze(1),
            valid_mask=mask,
            context_mask=mask,
            N=1,
            L=end - start,
            segment_lengths=None,
            position_offsets=torch.tensor([start]),
        )
        global_start = start + target_start
        global_end = start + target_end
        expected[:, global_start:global_end] = encoded[:, target_start:target_end]
    assert torch.allclose(actual, expected, atol=1e-6)


@torch.no_grad()
def test_aligned_window_rope_reset_matches_masked_absolute_positions() -> None:
    torch.manual_seed(19)
    model = FeatureJEPA(
        d_in=32,
        d_model=32,
        enc_depth=2,
        enc_heads=4,
        pred_depth=1,
        pred_heads=4,
        drop_path_prob=0.0,
        student_encoder_attention="block_causal",
        student_encoder_block_size=3,
        rope_mode="1d_flat",
    ).eval()
    features = torch.randn(1, 1, 12, 32)
    full_valid = torch.ones(1, 1, 12, dtype=torch.bool)
    suffix_context = full_valid.clone()
    suffix_context[:, :, :6] = False
    absolute = model.student_enc(
        features,
        valid_mask=full_valid,
        context_mask=suffix_context,
        N=1,
        L=12,
        segment_lengths=None,
    )

    window = features[:, :, 6:]
    window_valid = torch.ones(1, 1, 6, dtype=torch.bool)
    reset = model.student_enc(
        window,
        valid_mask=window_valid,
        context_mask=window_valid,
        N=1,
        L=6,
        segment_lengths=None,
        position_offsets=torch.tensor([6]),
    )
    assert torch.allclose(absolute[:, 6:], reset, atol=1e-5)
