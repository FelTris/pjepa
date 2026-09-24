from __future__ import annotations


import torch


from pjepa.cli.surgical_phase_order_ablation import (  # noqa: E402
    contiguous_segments,
    retained_adjacency_fraction,
    transformation_indices,
)


LABELS = torch.tensor([0, 0, 1, 1, 1, 0])


def test_contiguous_segments_preserve_repeated_nonadjacent_phases() -> None:
    assert contiguous_segments(LABELS) == [(0, 2), (2, 5), (5, 6)]


def test_controlled_reversal_indices() -> None:
    original, _ = transformation_indices(LABELS, "original")
    within, _ = transformation_indices(LABELS, "within_segment_reverse")
    segment_reverse, order = transformation_indices(LABELS, "segment_order_reverse")
    full, _ = transformation_indices(LABELS, "full_reverse")
    assert original.tolist() == [0, 1, 2, 3, 4, 5]
    assert within.tolist() == [1, 0, 4, 3, 2, 5]
    assert segment_reverse.tolist() == [5, 2, 3, 4, 0, 1]
    assert full.tolist() == [5, 4, 3, 2, 1, 0]
    assert order == [2, 1, 0]
    assert retained_adjacency_fraction(order) == 0.0


def test_random_segment_permutation_is_deterministic_and_internal_order_is_preserved() -> None:
    first, first_order = transformation_indices(
        LABELS, "random_segment_order", seed=7, video_id="video01", repeat=3
    )
    second, second_order = transformation_indices(
        LABELS, "random_segment_order", seed=7, video_id="video01", repeat=3
    )
    assert torch.equal(first, second)
    assert first_order == second_order
    assert sorted(first.tolist()) == list(range(len(LABELS)))
    segments = contiguous_segments(LABELS)
    expected_pieces = [list(range(*segments[index])) for index in first_order]
    cursor = 0
    for piece in expected_pieces:
        assert first[cursor : cursor + len(piece)].tolist() == piece
        cursor += len(piece)
