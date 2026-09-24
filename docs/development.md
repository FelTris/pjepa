# Development guide

## Reading the implementation

Start with `inference/encoder.py` and `models/feature_vjepa_rope.py` for the model's
public behavior. `models/feature_base.py` contains the preserved feature-JEPA
base implementation and tensor-packing helpers. `models/modules_2d.py` contains
only the transformer components used by the retained encoder and predictor.

`training/experiment.py` constructs Assembly101/EgoProceL experiments from a
portable recipe. `cli/feature_experiment.py` selects train, fit-probe, or saved-head
evaluation. Neither contains the numerical implementation of every task.

The feature training state is initialized in the small
`training/feature_ssl.py::Trainer` class. Its method groups are explicit:

| Module | Responsibility |
|---|---|
| `data/feature_batches.py` | Convert padded loader batches, apply masks, encode frozen features |
| `probes/operations.py` | Construct heads/optimizers, handle foreground labels, aggregate frame/segment predictions |
| `training/ssl_loop.py` | SSL optimizer/EMA steps and selection schedule |
| `training/probe_fit.py` | Fit frozen-feature heads and restore the selected head |
| `evaluation/probe_epoch.py` | Score an epoch without fitting or checkpoint selection |
| `evaluation/saved_probes.py` | Load and score retained temporal/linear heads |
| `training/tracking.py` | Optional logging only |

These groups share the state initialized by `Trainer`; they do not have hidden
constructors or import a second research trainer. Core model loading and direct
inference do not use this class. The old 4,663-line trainer, unrelated temporal
heads, EPIC branches, plotting logic, and cluster-specific paths are absent.

LEMON has a separate `training/lemon.py` loop because its video sampling and
Cholec80 development selection protocol differ. Its masking/validation and
checkpoint helpers live in `training/lemon_support.py`; its CLI just loads a
recipe and calls the loop.

## Preserving behavior

Changing Python module paths must not change `state_dict` parameter names.
Registered architecture fields are part of the checkpoint contract even when
they do not appear in tensor shapes. Verify head count, RoPE, attention mode,
and block size when changing construction code.

Use synthetic tests for padding, oracle-boundary requirements, metadata
conflicts, one-step SSL updates, end-to-end tiny training, saved-head evaluation,
and temporal diagnostic invariants. Tests write only to pytest temporary paths.
The tests do not embed pretrained weights, datasets, figures, or old results.

Real-checkpoint parity should compare original and released encoder/predictor/
teacher tensors in separate processes, followed by representative complete-video
and saved-head checks. Do not validate temporal context by replacing a full
sequence with arbitrary independent chunks. For a new algorithmic change,
create a new named experiment rather than silently changing a historical recipe.

## Packaging and outputs

`pyproject.toml` builds a wheel containing the Python package, artifact registry,
and vendored license notices. Configs and docs are part of the source release.
Encoder checkpoints are distributed separately through Hugging Face. The five
small linear heads are explicitly allowlisted in `.gitignore` for the GitHub
source release; other weight files remain ignored. Checkpoints are not included
in the wheel. `checkpoints/manifest.json` mirrors the package's
`artifacts.json`; update both together when registering an artifact.

The source provenance manifest records hashes of input research files used for
extraction. Old artifact paths in metadata are provenance only. Runtime configs
must use portable roots, and new CLI code must not mutate `sys.path` or search
for sibling repositories.

Generated outputs belong below `PJEPA_OUTPUT_ROOT`. Never add datasets, feature
caches, figures, tracking logs, or evaluation predictions to the code release.
