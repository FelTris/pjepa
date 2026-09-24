# Release validation

Validated on 2026-09-23 and 2026-09-24 with Python 3.11.9, PyTorch 2.6.0+cu118, NumPy 1.26.4,
timm 0.6.13, einops 0.6.1, and an NVIDIA RTX 3090. This is a tested environment,
not a guarantee for every future dependency version.

## Automated checks

```bash
python -m pip install -e '.[temporal,builders,dev]'
pytest -q
ruff check --select F src tests
```

All 25 tests pass from both the source tree and the installed wheel.

The suite covers architecture metadata conflicts that tensor shapes cannot
detect, missing oracle boundaries, padding, finite SSL gradients and EMA updates,
portable paths, relocated head/encoder identity, saved-head scoring, and
full-video context/order invariants. Tiny end-to-end fixtures exercise LEMON
training and both block/clip-causal feature training, checkpoint reloading,
fitting a linear head, and evaluating the saved head. Linear foreground/activity
heads and both retained temporal heads also have fitting/scoring checks.

## Existing checkpoint checks

- All five encoder checkpoints and all five retained registered linear heads load strictly.
- Original and release encoder, predictor, and teacher tensors agree exactly
  on fixed CPU inputs, including padded sequences crossing block boundaries
  and irregular oracle segment lengths. Reference model runs use separate
  Python processes.
- GPU evaluation covers every Cholec80 test video (40 videos, 98,234 frames)
  and every M2CAI16 test video (14 videos, 26,961 frames). Original and release
  student features and saved linear-head predictions agree exactly.
- Complete-video GPU encoder comparisons agree exactly for both Assembly101
  families and both EgoProceL feature families (one take per family).
- Before excluding temporal weights from distribution, the oracle Assembly101
  temporal configuration was checked on the full
  120-take validation split. Its associated linear head reproduces the
  historical aggregate segmentation metrics.
- Both original and released LT-Causal implementations produce identical logits
  for identical padded inputs. Their legacy whole-sequence normalization makes
  outputs depend on padding. Full-split evaluation with batch sizes one and
  sixteen confirms this dependence. Their historical checkpoint-name scores
  are **not** reproduced under the batch-size-one release protocol; see
  [normalization and padding](protocols.md#lt-causal-normalization-and-padding).

## Isolation and scope

A wheel was built without dependency downloads, installed into a temporary
package directory, and imported from outside the workspace. All package modules
imported with research/sibling namespaces blocked. Every registered encoder and
head also loaded from that installation. Core imports were checked with optional
packages unavailable, and the CLI/archive-builder help commands were exercised.

No full SSL training run was repeated for this refactor. The tests establish
short-run training behavior and existing-weight compatibility, not fresh-run
convergence or bitwise training resumption. EgoProceL full-dataset probe fitting
and all cross-dataset/ablation sweeps were not rerun. Archive builders were
ported with their input contracts; feature datasets were not rebuilt.

Verification logs and numerical results are kept outside the release. No test
requires the original research checkout, a real dataset, or downloaded weights.

On 2026-09-24, the remaining configurations, checkpoint registry, and README
inventory were checked together. The registry contains five encoders and five
linear probe heads. A synthetic cross-dataset CLI run also verified raw/student linear
evaluation, feature-cache creation and reuse, and masking of nonshared phases.
