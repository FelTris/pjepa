"""Batch preparation, masking, and frozen feature encoding."""

import numpy as np
import torch
from pjepa.models.mask import (
    make_random_masks_from_valid_idx,
    make_directional_future_masks_from_valid_idx,
    make_directional_future_masks_from_segment_lengths,
)


class FeatureBatchMixin:
    @torch.no_grad()
    def _normalize_loader_batch(self, batch):
        if len(batch) == 4:
            (batch_input, batch_target, mask, names) = batch
            segment_lengths = None
        elif len(batch) == 5:
            (batch_input, batch_target, mask, names, segment_lengths) = batch
        else:
            raise ValueError(f"Unsupported batch tuple of length {len(batch)}")
        if batch_input.dim() == 4:
            valid_mask_3d = mask[:, 0, ...].bool()
            return {
                "x4d": batch_input,
                "target_3d": batch_target,
                "valid_mask_3d": valid_mask_3d,
                "valid_mask_flat": valid_mask_3d.flatten(1, 2),
                "target_flat": batch_target.flatten(1, 2),
                "names": names,
                "segment_lengths": segment_lengths,
                "flat_mode": False,
            }
        if batch_input.dim() == 3:
            if mask.dim() != 2:
                raise ValueError(
                    f"Flat batches expect valid_mask of shape [B, T], got {tuple(mask.shape)}"
                )
            x4d = batch_input.unsqueeze(1)
            target_3d = batch_target.unsqueeze(1)
            valid_mask_3d = mask.bool().unsqueeze(1)
            return {
                "x4d": x4d,
                "target_3d": target_3d,
                "valid_mask_3d": valid_mask_3d,
                "valid_mask_flat": mask.bool(),
                "target_flat": batch_target,
                "names": names,
                "segment_lengths": segment_lengths,
                "flat_mode": True,
            }
        raise ValueError(f"Unsupported batch_input shape {tuple(batch_input.shape)}")

    def _build_segment_lengths_for_model(self, segment_lengths, x4d, flat_mode, device):
        if not flat_mode or segment_lengths is None:
            return None
        return segment_lengths.to(device)

    def _make_ssl_masks(self, normalized, valid_mask_3d, device):
        (B, N, L) = valid_mask_3d.shape
        used_directional = False
        if np.random.rand() < self.directional_mask_prob:
            if normalized["flat_mode"] and normalized["segment_lengths"] is not None:
                (context_mask_flat, target_mask_flat) = (
                    make_directional_future_masks_from_segment_lengths(
                        valid_mask_3d.reshape(B, N * L),
                        segment_lengths=normalized["segment_lengths"].to(device),
                        future_mask_ratio=self.directional_future_mask_ratio,
                        min_context_segments=self.directional_min_context_clips,
                        min_future_segments=self.directional_min_future_clips,
                    )
                )
                context_mask_3d = context_mask_flat.view(B, N, L)
                target_mask_3d = target_mask_flat.view(B, N, L)
            else:
                (context_mask_3d, target_mask_3d) = make_directional_future_masks_from_valid_idx(
                    valid_mask_3d,
                    future_mask_ratio=self.directional_future_mask_ratio,
                    min_context_clips=self.directional_min_context_clips,
                    min_future_clips=self.directional_min_future_clips,
                )
            if target_mask_3d.any():
                used_directional = True
            else:
                (context_mask_3d, target_mask_3d) = make_random_masks_from_valid_idx(
                    valid_mask_3d, mask_ratio=self.mask_ratio
                )
        else:
            (context_mask_3d, target_mask_3d) = make_random_masks_from_valid_idx(
                valid_mask_3d, mask_ratio=self.mask_ratio
            )
        return (context_mask_3d, target_mask_3d, used_directional)

    @torch.no_grad()
    def _student_features(self, x, valid_mask_3d, segment_lengths=None):
        (B, N, L, _) = x.shape
        t_full = self.model.student_enc(
            x,
            valid_mask=valid_mask_3d,
            context_mask=valid_mask_3d,
            N=N,
            L=L,
            segment_lengths=segment_lengths,
        )
        return t_full

    @torch.no_grad()
    def _probe_features(
        self, x, valid_mask_3d, probe_feature_source: str = "student", segment_lengths=None
    ):
        if probe_feature_source == "student":
            return self._student_features(x, valid_mask_3d, segment_lengths=segment_lengths)
        if probe_feature_source == "raw":
            (B, N, L, _) = x.shape
            return x.reshape(B, N * L, -1)
        raise ValueError(
            f"Unsupported probe_feature_source='{probe_feature_source}'. Use 'student' or 'raw'."
        )
