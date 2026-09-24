import unittest

import torch

from pjepa.probes.causal_ltcontext import (
    CausalLTContextProbe,
    _base_segment_causal_mask,
    _block_ids_from_positions,
)


class BlockCausalLTContextTest(unittest.TestCase):
    def _probe(self) -> CausalLTContextProbe:
        torch.manual_seed(7)
        probe = CausalLTContextProbe(
            in_dim=4,
            num_classes=3,
            cfg_overrides={
                "num_layers": 1,
                "num_stages": 1,
                "model_dim": 4,
                "windowed_attn_w": 2,
                "long_term_attn_g": 2,
                "conv_dilation_factor": 2,
                "dim_reduction": 1.0,
                "channel_masking_prob": 0.0,
                "dropout_prob": 0.0,
                "use_instance_norm": True,
                "num_attn_heads": 1,
                "attention_dropout": 0.0,
                "causal_attention_mode": "block_causal",
                "causal_block_size": 4,
            },
        )
        return probe.eval()

    def test_block_ids_are_take_local_and_mask_padding(self):
        valid = torch.tensor([[True, True, True, True, True, True, True, False]])
        actual = _block_ids_from_positions(valid, block_size=3)
        expected = torch.tensor([[0, 0, 0, 1, 1, 1, 2, -1]])
        self.assertTrue(torch.equal(actual, expected))

    def test_current_block_is_bidirectional_and_future_blocks_are_hidden(self):
        valid = torch.ones(1, 8, dtype=torch.bool)
        block_ids = _block_ids_from_positions(valid, block_size=4)
        allowed = _base_segment_causal_mask(block_ids, block_ids, valid, valid)
        self.assertTrue(bool(allowed[0, 4, 7]))
        self.assertFalse(bool(allowed[0, 3, 4]))

    def test_block_mode_ignores_segment_lengths(self):
        probe = self._probe()
        features = torch.randn(1, 8, 4)
        valid = torch.ones(1, 8, dtype=torch.bool)
        with torch.no_grad():
            first = probe(features, valid, torch.tensor([[2, 6]]))
            second = probe(features, valid, torch.tensor([[8]]))
            without_boundaries = probe(features, valid)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, without_boundaries))

    def test_appending_future_block_does_not_change_previous_blocks(self):
        probe = self._probe()
        prefix = torch.randn(1, 8, 4)
        extended = torch.cat([prefix, torch.randn(1, 4, 4)], dim=1)
        with torch.no_grad():
            prefix_logits = probe(prefix, torch.ones(1, 8, dtype=torch.bool))
            extended_logits = probe(extended, torch.ones(1, 12, dtype=torch.bool))
        self.assertTrue(torch.allclose(prefix_logits, extended_logits[..., :8], atol=1e-6))

    def test_segment_mode_still_requires_boundaries(self):
        probe = CausalLTContextProbe(
            in_dim=4,
            num_classes=3,
            cfg_overrides={
                "num_layers": 1,
                "num_stages": 1,
                "model_dim": 4,
                "causal_attention_mode": "segment_causal",
            },
        )
        with self.assertRaisesRegex(ValueError, "requires ground-truth"):
            probe(torch.randn(1, 4, 4), torch.ones(1, 4, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
