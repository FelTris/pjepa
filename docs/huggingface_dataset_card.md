---
pretty_name: P-JEPA feature archives
license: other
license_name: source-dataset-terms
license_link: LICENSE.md
tags:
  - video
  - feature-extraction
  - temporal-action-segmentation
  - surgical-phase-recognition
  - pytorch
  - pjepa
---

# P-JEPA feature archives

Precomputed **input features** and aligned annotations for the
[P-JEPA encoders](https://huggingface.co/FelTris/pjepa). These are backbone/pooler
outputs, before P-JEPA temporal encoding. No raw videos or P-JEPA output caches
are included. All feature tensors are float32.

The files use the directory layout expected by the
[P-JEPA code](https://github.com/FelTris/pjepa). That GitHub repository is currently
private pending the code release. The plain PyTorch example below works without
it; the full training/evaluation recipes require access to the code.

## Contents and encoder inputs

| Dataset / feature family | Feature extractor | Width | Stored sampling | Videos / splits | Size |
|---|---|---:|---|---|---:|
| LEMON | PL-Stitch ViT-B/16, final CLS | 768 | 4 fps; released P-JEPA recipe takes every fourth token (1 fps) | 4,194 pretrain | 40.24 GB |
| Cholec80 | Same PL-Stitch | 768 | 1 fps | 80; train 01–40, test 41–80 | 0.57 GB |
| M2CAI16 | Same PL-Stitch | 768 | 1 fps | 27 train / 14 test | 0.29 GB |
| Assembly101 | Released TSM features, 8-frame input | 2048 | Stride 8 on a 30-Hz feature timeline: 3.75 fps | 204 train / 61 val; camera C10119 | 3.37 GB |
| EgoProceL / FACT features | Upstream FACT-provided feature vectors | 2048 | 10-Hz source resampled to nominal 4 fps | 731 train / 183 test | 13.92 GB |
| EgoProceL / pooled V-JEPA | V-JEPA 2.1 ViT-g + supervised framewise pooler | 1408 | Nominal 4 fps | 337 train / 86 test | 5.73 GB |

Sizes are decimal GB; total features are approximately **64.13 GB**, plus three
small CSV segment manifests. Coverage is the archived experimental subset, not
a claim to contain every video/view of each original dataset. The two EgoProceL
families have different coverage and must use their respective manifests.

### Feature provenance and timing

- **PL-Stitch:** `pl_lemon.pth`, SHA-256
  `8c4166f24c02a7cb5d10a89ebe179dfb50de25319031f4128af22d57a992d8a3`;
  upstream code commit `f4a3b19de71c041d77e091817dbe93a7da168498`.
  Final CLS features precede the projection head. Images are RGB, resized
  directly to 224×224 using bilinear antialiased interpolation, with ImageNet
  channel mean/std. Feature computation and storage use float32. This image
  preprocessing is already applied; pass the stored vectors directly to P-JEPA.
- **Assembly101:** the official TSM feature release is a supervised feature
  representation. This archive samples every eighth feature on its 30-Hz
  annotation timeline; `times` contains seconds. Do not substitute the video's
  60-fps container rate when interpreting annotation frame numbers. The exact
  upstream extractor checkpoint hash is not recorded in the archive.
- **FACT features:** repacked from the upstream EgoProceL `.npy` distribution.
  FACT names the provider here, not an additional temporal model to run.
  The exact backbone/checkpoint identity is not recorded in the archive, so
  2048 dimensions alone do not identify a compatible replacement encoder.
  Features are trimmed to available ground-truth length before resampling.
  Indices follow `floor(arange(0, duration, 0.25) * 10)`: stored timestamps can
  alternate between 0.2- and 0.3-second gaps. Use `times`, not `arange(T)/4`.
- **Pooled V-JEPA:** extraction used `vjepa2_1_vit_giant_384_framewise`, a frozen
  pretrained V-JEPA 2.1 ViT-g backbone with a supervised framewise pooler trained
  using FACT step labels (`model-epoch=21.ckpt`). The extraction configuration
  uses a 256-pixel test crop, 64-frame non-overlapping windows sampled at 4 fps,
  and 64 learned frame queries (pooler depth 4, 16 heads). Saved features are
  frame-pooler outputs before classifier pooling, not raw V-JEPA patch averages.
  Timestamps follow sampled video frames and can deviate from an exact 0.25-s
  grid. The final window is padded during extraction; padded outputs are omitted.
  This representation has supervised provenance even when subsequent P-JEPA
  fitting uses an SSL loss. Its extractor weights are not part of this dataset.

## Download

Download everything into your feature-data root (allow at least 65 GB):

```bash
python -m pip install huggingface_hub
export PJEPA_DATA_ROOT=/path/to/pjepa-data
hf download FelTris/pjepa_features --repo-type dataset --local-dir "$PJEPA_DATA_ROOT"
cd "$PJEPA_DATA_ROOT"
sha256sum -c SHA256SUMS
```

For only Assembly101, for example:

```bash
hf download FelTris/pjepa_features --repo-type dataset \
  --include 'assembly101_small/*' 'assembly101_annotations/*' \
  --local-dir "$PJEPA_DATA_ROOT"
```

The corresponding pairs for EgoProceL are `egoprocel_small/fact_npy_features_4fps_full.pt`
with `egoprocel_annotations/egoprocel_fact_features_segments.csv`, and
`egoprocel_small/pooled_features_fact.pt` with
`egoprocel_annotations/egoprocel_segments.csv`. Download every LEMON shard along
with its index. Partial downloads will not pass the full-repository checksum check.

## Format and loading

Use PyTorch 2.6+; these are packed torch archives rather than a Hugging Face
`datasets.load_dataset` table. `manifest.json` records file hashes, byte sizes,
dimensions, token/video counts, split counts, and extraction metadata.

```python
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "FelTris/pjepa_features",
    "Cholec80/cholec80_pl_stitch_vitb16_1fps.pt",
    repo_type="dataset",
)
archive = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
video_index = 0
start, end = map(int, archive["offsets"][video_index:video_index + 2])
features = archive["tokens"][start:end]  # [time, 768], float32
times = archive["times"][start:end]      # seconds
labels = archive["phase_labels"][start:end]
video_id = archive["video_ids"][video_index]
```

For a packed Assembly101/EgoProceL file, use the same `tokens`, `times`, and
`offsets` slicing; `paths[video_index]` identifies the sequence. Labels and
intervals come from its matching CSV, joined by `video_path` (replace a video
extension with `.pt` to match archive `paths`). CSV `official_split` supplies
the split. Preserve class IDs and background handling from the supplied recipes;
EgoProceL uses `fact_step_id` with background 0, Assembly101 uses `action_id`.
The release loaders handle interval alignment and take grouping.

LEMON's index maps each video through `shard_ids` and `local_indices` to a
shard's `offsets`. Resolve `shards[shard_id]` relative to the downloaded index
directory. For 1-fps P-JEPA inputs, slice each video's tokens/times with `[::4]`.
Historical absolute paths in archive metadata document extraction provenance;
they are not required files on the user's machine. No raw-video decoder,
PL-Stitch installation, FACT installation, or V-JEPA installation is needed.

The released LEMON loader memory-maps shards with a bounded shard cache. The
Assembly101/EgoProceL training loader loads its packed archive into host memory;
budget RAM for the selected archive plus training buffers. The plain loading
example uses memory mapping to avoid eagerly reading the entire tensor.

## Protocols and matching P-JEPA checkpoints

| Input | Encoder in `FelTris/pjepa` |
|---|---|
| PL-Stitch at 1 fps | `lemon_plstitch_block32.pt` |
| Assembly101 TSM at 3.75 fps | `assembly101_tsm_block64.pt` or oracle `assembly101_tsm_clip.pt` |
| EgoProceL FACT features | Oracle `egoprocel_fact_features_clip.pt` |
| EgoProceL pooled V-JEPA | Oracle `egoprocel_pooled_clip.pt` |

**Clip-causal models require ground-truth segment boundaries and are oracle
comparisons. Use block-causal models going forward.** Block-causal P-JEPA does
not require true boundaries for attention; historical training sequence
construction and upstream feature learning can still use annotations.

LEMON SSL uses only its `pretrain` split. Cholec80 development selection fits
probes on videos 01–32 and selects on 33–40; 41–80 remain held out. Full Cholec80
probing uses 01–40 / 41–80. M2CAI16 uses all 27 / 14 official train/test videos;
inline `source_splits` records membership. Surgical phase names are included
in `phase_class_names` (7 Cholec80 classes; 8 M2CAI16 classes).

Historical EgoProceL configurations use the split named `test` for model/probe
selection. It must not be described as an untouched final test set. The pooled
feature extractor's configuration also uses `test` for validation. Different
feature families, supervised extractors, and dataset coverage must be controlled
when interpreting downstream comparisons.

## Sources and terms

Credit the original datasets and feature methods when using these archives:
[LEMON](https://github.com/visurg-ai/LEMON),
[PL-Stitch](https://github.com/visurg-ai/PL-Stitch),
[Cholec80 and M2CAI16](https://camma.unistra.fr/datasets/),
[Assembly101 and its TSM features](https://github.com/assembly-101/assembly101-temporal-action-segmentation),
[FACT's EgoProceL distribution](https://github.com/ZijiaLewisLu/CVPR2024-FACT),
and [V-JEPA 2.1](https://github.com/facebookresearch/vjepa2).
See [LICENSE.md](LICENSE.md) for source-specific terms. The MIT license on
P-JEPA code/encoder weights does not relicense the source datasets or annotations.
