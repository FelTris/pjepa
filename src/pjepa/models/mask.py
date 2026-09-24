import torch


def make_random_masks_from_valid_idx(valid_mask_3d, mask_ratio=0.6, min_keep=1):
    B, N, L = valid_mask_3d.shape
    vm = valid_mask_3d.view(B, -1)  # (B, T)
    tgt = torch.zeros_like(vm, dtype=torch.bool)
    device = vm.device

    for b in range(B):
        idx = torch.nonzero(vm[b], as_tuple=False).flatten()  # valid positions
        Lb = idx.numel()
        if Lb <= min_keep:
            continue
        k = min(Lb - min_keep, max(1, int(round(Lb * mask_ratio))))
        sel = idx[torch.randperm(Lb, device=device)[:k]]
        tgt[b, sel] = True

    ctx = vm & ~tgt
    return ctx.view(B, N, L), tgt.view(B, N, L)


def make_directional_future_masks_from_valid_idx(
    valid_mask_3d,
    future_mask_ratio=0.8,
    min_context_clips=1,
    min_future_clips=1,
    min_keep_future_unmasked=0,
):
    """
    Build directional masks where context is restricted to earlier clips and targets
    are sampled randomly from later ("future") clips.

    Args:
        valid_mask_3d: Bool tensor (B, N, L), True = valid token.
        future_mask_ratio: Fraction of valid future tokens to mask.
        min_context_clips: Minimum number of valid clips kept as context.
        min_future_clips: Minimum number of valid clips reserved as future.
        min_keep_future_unmasked: Keep at least this many future tokens unmasked.

    Returns:
        ctx_3d: Bool tensor (B, N, L), True = visible context token.
        tgt_3d: Bool tensor (B, N, L), True = masked target token.
    """
    B, N, L = valid_mask_3d.shape
    ctx = torch.zeros_like(valid_mask_3d, dtype=torch.bool)
    tgt = torch.zeros_like(valid_mask_3d, dtype=torch.bool)
    device = valid_mask_3d.device

    for b in range(B):
        valid_clip = valid_mask_3d[b].any(dim=1)  # (N,)
        clip_idx = torch.nonzero(valid_clip, as_tuple=False).flatten()
        n_valid_clips = int(clip_idx.numel())
        if n_valid_clips < (min_context_clips + min_future_clips):
            continue

        # Randomly choose how many earliest valid clips are context.
        c_min = int(min_context_clips)
        c_max = int(n_valid_clips - min_future_clips)
        if c_max < c_min:
            continue
        n_ctx = int(torch.randint(c_min, c_max + 1, (1,), device=device).item())

        ctx_clips = clip_idx[:n_ctx]
        fut_clips = clip_idx[n_ctx:]
        if fut_clips.numel() == 0:
            continue

        # Context tokens are strictly from earlier clips.
        ctx[b, ctx_clips] = valid_mask_3d[b, ctx_clips]

        future_vm = valid_mask_3d[b, fut_clips].reshape(-1)  # (F*L,)
        fut_idx = torch.nonzero(future_vm, as_tuple=False).flatten()
        n_future_tokens = int(fut_idx.numel())
        if n_future_tokens <= 0:
            continue

        max_k = n_future_tokens - int(min_keep_future_unmasked)
        if max_k <= 0:
            continue
        k = min(max_k, max(1, int(round(n_future_tokens * float(future_mask_ratio)))))
        sel = fut_idx[torch.randperm(n_future_tokens, device=device)[:k]]

        future_tgt = torch.zeros_like(future_vm, dtype=torch.bool)
        future_tgt[sel] = True
        tgt[b, fut_clips] = future_tgt.view(fut_clips.numel(), L)

    # Safety: targets must always be valid tokens.
    tgt &= valid_mask_3d
    ctx &= valid_mask_3d
    # Also ensure no overlap.
    ctx &= ~tgt
    return ctx, tgt


def make_directional_future_masks_from_segment_lengths(
    valid_mask_2d,
    segment_lengths,
    future_mask_ratio=0.8,
    min_context_segments=1,
    min_future_segments=1,
    min_keep_future_unmasked=0,
):
    """
    Flat-sequence variant of directional masking.

    Args:
        valid_mask_2d: Bool tensor (B, T), True = valid token.
        segment_lengths: Long tensor (B, S), padded with zeros.
        future_mask_ratio: Fraction of valid future tokens to mask.
        min_context_segments: Minimum number of earlier segments kept as context.
        min_future_segments: Minimum number of later segments reserved as future.
        min_keep_future_unmasked: Keep at least this many future tokens unmasked.

    Returns:
        ctx_2d: Bool tensor (B, T), True = visible context token.
        tgt_2d: Bool tensor (B, T), True = masked target token.
    """
    B, T = valid_mask_2d.shape
    ctx = torch.zeros_like(valid_mask_2d, dtype=torch.bool)
    tgt = torch.zeros_like(valid_mask_2d, dtype=torch.bool)
    device = valid_mask_2d.device

    for b in range(B):
        lengths = [int(length) for length in segment_lengths[b].tolist() if int(length) > 0]
        n_segments = len(lengths)
        if n_segments < (min_context_segments + min_future_segments):
            continue

        c_min = int(min_context_segments)
        c_max = int(n_segments - min_future_segments)
        if c_max < c_min:
            continue
        n_ctx = int(torch.randint(c_min, c_max + 1, (1,), device=device).item())

        offset = 0
        future_indices = []
        for seg_idx, seg_len in enumerate(lengths):
            if offset >= T:
                break
            seg_len = min(int(seg_len), T - offset)
            seg_slice = slice(offset, offset + seg_len)
            seg_valid = valid_mask_2d[b, seg_slice]
            seg_token_idx = torch.nonzero(seg_valid, as_tuple=False).flatten() + offset
            if seg_idx < n_ctx:
                ctx[b, seg_token_idx] = True
            else:
                future_indices.append(seg_token_idx)
            offset += seg_len

        if not future_indices:
            continue

        fut_idx = torch.cat(future_indices)
        n_future_tokens = int(fut_idx.numel())
        if n_future_tokens <= 0:
            continue

        max_k = n_future_tokens - int(min_keep_future_unmasked)
        if max_k <= 0:
            continue
        k = min(max_k, max(1, int(round(n_future_tokens * float(future_mask_ratio)))))
        sel = fut_idx[torch.randperm(n_future_tokens, device=device)[:k]]
        tgt[b, sel] = True

    tgt &= valid_mask_2d
    ctx &= valid_mask_2d
    ctx &= ~tgt
    return ctx, tgt


def make_block_masks_from_valid_idx(
    valid_mask_3d,
    mask_ratio=0.6,
    block_len=8,
    min_keep_run=1,
    max_tries=1000,
):
    """
    Args:
        valid_mask_3d: Bool tensor (B, N, L). True = valid token.
        mask_ratio: Fraction of *valid* tokens you want masked (rounded to blocks).
        block_len: Length of each masked consecutive block.
        min_keep_run: Ensure at least one *unmasked* run of this many consecutive valid tokens remains.
        max_tries: Safety cap for random placement attempts.

    Returns:
        ctx_3d: Bool tensor (B, N, L). True = keep (context).
        tgt_3d: Bool tensor (B, N, L). True = mask (target).
    """
    B, N, L = valid_mask_3d.shape
    vm = valid_mask_3d.view(B, -1)  # (B, T)
    tgt = torch.zeros_like(vm, dtype=torch.bool)
    device = vm.device
    T = vm.size(1)

    for b in range(B):
        vb = vm[b]  # (T,)
        idx = torch.nonzero(vb, as_tuple=False).flatten()
        Lb = idx.numel()
        if Lb == 0:
            continue

        # If we can't place even one block while keeping a protected run, skip.
        if Lb < block_len + max(0, min_keep_run):
            continue

        # ---- Find contiguous runs of valid positions (in flattened T space) ----
        if Lb == 1:
            diffs = torch.empty(0, device=device, dtype=idx.dtype)
        else:
            diffs = idx[1:] - idx[:-1]
        split_points = torch.nonzero(diffs != 1, as_tuple=False).flatten()
        starts_i = torch.cat([torch.tensor([0], device=device), split_points + 1])
        ends_i = torch.cat([split_points, torch.tensor([Lb - 1], device=device)])
        runs = torch.stack([idx[starts_i], idx[ends_i]], dim=1)  # [R, 2] inclusive
        run_lengths = runs[:, 1] - runs[:, 0] + 1

        # ---- Choose a protected (unmasked) consecutive segment so at least one survives ----
        protected = torch.zeros(T, dtype=torch.bool, device=device)
        if min_keep_run > 0:
            candidates = torch.nonzero(run_lengths >= min_keep_run, as_tuple=False).flatten()
            if candidates.numel() > 0:
                r_idx = candidates[torch.randint(0, candidates.numel(), (1,), device=device)].item()
                r_start, r_end = runs[r_idx].tolist()
                keep_start = torch.randint(
                    r_start, r_end - min_keep_run + 2, (1,), device=device
                ).item()
                protected[keep_start : keep_start + min_keep_run] = True
            else:
                # Fallback: protect min_keep_run random valid tokens (may not be consecutive)
                sel = idx[torch.randperm(Lb, device=device)[:min_keep_run]]
                protected[sel] = True

        # ---- Decide how many blocks to place (respecting protected budget) ----
        # Total masked tokens = n_blocks * block_len
        max_blocks_by_budget = max(0, (Lb - min_keep_run) // block_len)
        n_blocks = max(1, int(round((Lb * mask_ratio) / block_len)))
        n_blocks = min(n_blocks, max_blocks_by_budget)
        if n_blocks <= 0:
            continue

        # Only runs that can hold at least one block
        blockable_runs = torch.nonzero(run_lengths >= block_len, as_tuple=False).flatten().tolist()
        if not blockable_runs:
            continue

        masked = torch.zeros(T, dtype=torch.bool, device=device)
        placed = 0
        tries = 0

        # ---- Greedy random placement of blocks without overlap or touching the protected window ----
        # We sample a run uniformly, then a start uniformly within that run.
        while placed < n_blocks and tries < max_tries:
            tries += 1
            r_idx = blockable_runs[
                torch.randint(0, len(blockable_runs), (1,), device=device).item()
            ]
            r_start, r_end = runs[r_idx].tolist()
            if r_end - r_start + 1 < block_len:
                continue

            start = torch.randint(r_start, r_end - block_len + 2, (1,), device=device).item()
            seg_idx = torch.arange(start, start + block_len, device=device)

            # Check: all valid (by construction yes), not protected, not already masked
            if protected[seg_idx].any():
                continue
            if masked[seg_idx].any():
                continue

            # Safe to place this block
            masked[start : start + block_len] = True
            placed += 1

        # Ensure we never mask the protected region
        masked = masked & ~protected
        tgt[b] = masked

    ctx = vm & ~tgt
    return ctx.view(B, N, L), tgt.view(B, N, L)
