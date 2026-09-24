# Protocol and checkpoint semantics

## Oracle versus boundary-free attention

Clip-causal attention and clip-relative 2D RoPE use true segment boundaries.
They are oracle comparisons. They are not the recommended route for future
work when those annotations are unavailable. Segment-causal LT-Causal has the
same limitation. Fixed-block attention/1D RoPE requires no such boundaries.

Archive manifests still contain labels and segment intervals for downstream
supervision and historical sequence construction. Removing oracle attention
does not make every historical preprocessing or selection choice label-free.
The distinction between SSL fitting, checkpoint selection, and supervised
probe fitting must remain explicit.

## LEMON and surgery

- SSL fits on LEMON features. The released block-causal encoder uses 768-wide
  tokens, encoder depth 4, predictor depth 2, 8 attention heads, block size 32,
  1D RoPE. Its saved configuration overrides the
  old local 16-head configuration when loading it.
- Encoder selection uses a Cholec80 development split: videos 1–32 fit the
  selection probe, 33–40 select the encoder. Videos 41–80 are held out.
- The released LEMON checkpoint is epoch 70. The selection criterion is
  development phase accuracy.
- Cholec80 full-data probing uses videos 1–40 for fitting and 41–80 for testing.
  M2CAI16 uses its complete official 27/14 train/test split. One-video adaptation
  recipes are not part of the release.
- Cross-dataset evaluation reuses Cholec80 heads on the seven shared phases.
  Trocar Placement is ignored for scoring; other sequence frames stay aligned.
- Surgical evaluation retains raw PL-Stitch and P-JEPA linear heads. Surgical
  LTContext fitting, evaluation, and pretrained heads are excluded.
- Context ablations use the same frozen full-context head. Order ablations that
  permute phase segments use label-defined boundaries as a diagnostic; they
  should not be described as an unlabeled inference procedure.
- No bidirectional pretrained weight was found in the retained artifacts.
  The bidirectional recipe is included for training; its historical 16-head
  architecture differs from the released 8-head block-causal checkpoint.

## Assembly101

The retained inputs are TSM features (2048 dimensions), encoded to 1408
features. Two historical families are retained: fixed block-causal/1D RoPE
(epoch 360), and oracle clip-causal/2D RoPE (epoch 940).

Each encoder transformer block is applied once, in order. The block-causal
encoder is provided for inference and fitting new probes; no pretrained
block-causal probe heads are included. The oracle clip-causal family includes
its saved linear head. Temporal heads can be fitted with the supplied code.

Linear/LTContext/LT-Causal checkpoint selection follows the configured original
validation metric. The combined segmentation score sums frame accuracy,
foreground accuracy, Edit/100, and F1 at 0.10/0.25/0.50 divided by 100, preserving
the existing convention. Do not compare this scalar across unrelated datasets
as though it were an accuracy percentage.

## LT-Causal normalization and padding

The original LT-Causal implementation uses `InstanceNorm1d` over the complete
padded temporal dimension. Its historical `use_instance_norm` flag is inverted:
`false` constructs instance normalization, while `true` constructs an identity.
The retained LT-Causal recipe uses `false`. We preserve this behavior and the flag's
meaning for weight compatibility; changing either is an algorithm change.

Whole-sequence normalization can expose future statistics despite causal
attention and convolutions. These heads therefore do **not** establish strict
streaming causality. They also produce different predictions when the same
video is padded to different lengths. The release defaults to one complete
video per validation batch for consistent padding. Probe fitting retains the
historical training batch size. Reproducing old multi-video validation requires
both the original batch size and the original grouping/order, which was shuffled
across epochs. A checkpoint filename's historical accuracy is not a promise of
that score under the release's batch-size-one protocol.

The original and released causal heads were compared on identical padded input:
their logits agree exactly. New padding-invariant or streaming-causal models
should use a separately named implementation and be retrained and validated.
This head limitation does not change the released block-causal P-JEPA encoder.

## EgoProceL

Both feature families are retained with their original oracle encoders:
FACT-provided 2048-D features (epoch 400) and pooled V-JEPA 1408-D features
(epoch 540). Both output 1408 dimensions. Foreground/activity linear heads and
background ID 0 preserve the existing label convention.

The pooled inputs come from a frozen V-JEPA 2.1 ViT-g backbone with a supervised
framewise pooler trained on FACT step labels. They are not raw backbone patch
averages. The two archived feature families also cover different video subsets
(914 FACT-feature videos versus 423 pooled-feature videos). See the
[dataset card](huggingface_dataset_card.md) for extraction and timing details.

Historical configurations use the split named `test` as `val_split` during
probe/checkpoint selection. This is retained for reproducing those runs and
must not be presented as a never-seen final test set. A new evaluation protocol
requires separately specified training/validation/test manifests and separately
identified checkpoint selection.

## Artifacts

The manifest identifies selected historical checkpoints within the agreed
families; it does not rank unlike protocols against each other. The release
preserves the original parameter tensors. Release bundles contain cleaned
metadata; the manifest also records source checksums when serialization changes.
Embedded old paths are provenance, not runtime requirements.
Head/encoder compatibility is checked by content identity when
files move. Archive caches are regenerated when source identity changes.

Feature training snapshots store model architecture alongside
weights; the CLI currently supports training from scratch and encoder warm
starts. Exact training resumption is not a supported claim for old checkpoints
that lack complete optimizer, teacher, scheduler, and RNG state.
