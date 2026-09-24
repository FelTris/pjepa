---
license: mit
tags:
  - pytorch
  - video
  - self-supervised-learning
  - feature-extraction
---

# P-JEPA

Self-supervised temporal encoders for **precomputed video features**. These
checkpoints learn temporal representations over feature sequences; they do not
extract features from raw video. All encoder weights are released under MIT.

| Checkpoint | Training features | Output width | Attention |
|---|---|---|---|
| `lemon_plstitch_block32.pt` | LEMON, PL-Stitch, 768-D, 1 fps | 768 | 32-token blocks |
| `assembly101_tsm_block64.pt` | Assembly101, TSM, 2048-D, 3.75 fps | 1408 | 64-token blocks |
| `assembly101_tsm_clip.pt` | Assembly101, TSM, 2048-D | 1408 | Clip-causal (oracle) |
| `egoprocel_fact_features_clip.pt` | EgoProceL, FACT I3D features, 2048-D, 4 fps | 1408 | Clip-causal (oracle) |
| `egoprocel_pooled_clip.pt` | EgoProceL, pooled V-JEPA features, 1408-D, 4 fps | 1408 | Clip-causal (oracle) |

**Use block-causal models going forward.** Clip-causal models require true
segment boundaries, which are generally unavailable in real-world inference.
They are historical oracle comparisons. Block-causal attention permits access
to the whole current block; it is not zero-latency frame-by-frame streaming.

![PL-Stitch and LEMON-trained P-JEPA features on Cholec80, colored by relative time](assets/cholec80_temporal_features.png)

**Frozen transfer / zero-shot encoder application to Cholec80.**
The LEMON-trained P-JEPA encoder is applied without fine-tuning on Cholec80.
The t-SNE panels compare PL-Stitch features (left) with frozen P-JEPA
representations (right): 100 frames sampled uniformly from each of the 40
held-out videos, colored by relative time from start (purple) to end (yellow).
Cholec80 development labels were used for checkpoint selection; supervised
linear probes are separate from this representation visualization.

## Use

Install the [P-JEPA code](https://github.com/FelTris/pjepa) (Python 3.11+, PyTorch 2.6+; currently private pending release) and
`huggingface_hub`, then:

```python
import torch
from huggingface_hub import hf_hub_download
from pjepa.checkpoints import load_model
from pjepa.inference.encoder import encode

path = hf_hub_download("FelTris/pjepa", "lemon_plstitch_block32.pt")
model, metadata = load_model(path, device="cpu")
features = torch.randn(1, 128, 768)  # replace with PL-Stitch features
representations = encode(model, features)
```

Oracle encoders additionally require `segment_lengths`. Use the matching feature
backbone and timing for each checkpoint. Downstream classification/segmentation
requires a separately fitted head; small linear heads accompany the code release.

Precomputed inputs and matching annotations are available at
[FelTris/pjepa_features](https://huggingface.co/datasets/FelTris/pjepa_features).
The dataset card documents sampling rates, feature extractors, and download instructions.

## Protocol and validation

The FACT-input P-JEPA encoder was trained on the original mixed-view
EgoProceL split (731 train / 183 test). The public feature archive now matches
the pooled archive's 337 train / 86 test video IDs, excluding three CMU static
camera views. Its saved encoder weights are unchanged; the current archive does
not reproduce its original training-data coverage.

LEMON encoder selection uses Cholec80 development videos; the test videos are
held out. Historical EgoProceL runs use a split named `test` for selection, so
it must not be described as untouched test evaluation. These are research
representations, not validated clinical decision systems.

Strict loading and numerical parity were checked against the source models,
including complete-video GPU inference. No new benchmark scores are claimed
here. See [the manifest](manifest.json) for architectures and artifact hashes;
[SHA256SUMS](SHA256SUMS) verifies the downloads. Dataset and input-feature terms
remain separate from the encoder weight license.
