# Third-party source notices

This release extracts code from the local research implementations listed in
`docs/source_provenance.json`. Source hashes record the audited versions; it
must not be assumed that a current upstream checkout has identical behavior.

- Feature-JEPA transformer/model utilities derive from the modified V-JEPA
  codebase. Preserve the Meta MIT notice in `licenses/V-JEPA-MIT.txt`.
- LTContext model code is retained under `src/pjepa/_vendor/ltcontext` with its
  license in `licenses/LTContext-CC-BY-NC-4.0.txt`. That component is licensed
  under CC BY-NC 4.0; the root MIT file does not relicense it.
- The source repository inherited MS-TCN code and attribution. Its original
  notice is retained in `licenses/MS-TCN-MIT.txt` and `LICENSE`. MS-TCN model
  implementations and training scripts are excluded from this release.
- `timm`, PyTorch, NumPy, loguru, einops, LMDB, and tqdm are installed dependencies
  under their respective upstream licenses; their packages are not copied here.

Dataset and feature-backbone weights are supplied separately. Their terms are
not changed by this code release. FACT temporal models, LV-MAE, and raw-video
backbone implementations are not bundled.

The released P-JEPA encoder weights use the MIT license in
`licenses/ENCODER-MIT.txt`.
