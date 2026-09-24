"""Load registered probe heads without constructing a trainer."""

import os
from pathlib import Path
import torch
from pjepa.checkpoints import artifact_hashes, registry, sha256_file
from pjepa.probes.linear import LinearProbe
from pjepa.probes.surgical_phase_linear_probe import SurgicalPhaseLinearHead


def load_probe(checkpoint, *, checkpoint_root=None, device="cpu"):
    """Return a head and its architecture/encoder provenance from the registry."""
    entries = registry()["heads"]
    entry = entries.get(str(checkpoint))
    root = Path(checkpoint_root or os.environ.get("PJEPA_CHECKPOINT_ROOT", "checkpoints/weights"))
    path = root / entry["filename"] if entry else Path(checkpoint)
    digest = sha256_file(path)
    if entry is None:
        entry = next((e for e in entries.values() if digest in artifact_hashes(e)), None)
    if entry is None or digest not in artifact_hashes(entry):
        raise ValueError(
            "Unknown probe artifact or checksum mismatch; provide its registered checkpoint."
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    arch = entry["architecture"]
    if entry["kind"] == "linear":
        cls = (
            SurgicalPhaseLinearHead if entry["dataset"] in {"cholec80", "m2cai16"} else LinearProbe
        )
        head = cls(arch["in_dim"], arch["num_classes"])
    elif entry["kind"] == "ltcontext":
        from pjepa.probes.ltcontext_probe import LTContextProbe

        head = LTContextProbe(**arch)
    elif entry["kind"] == "causal_ltcontext":
        from pjepa.probes.causal_ltcontext import CausalLTContextProbe

        head = CausalLTContextProbe(**arch)
    else:
        raise ValueError(f"Unknown probe kind {entry['kind']}")
    head.load_state_dict(payload.get("model_state_dict", payload), strict=True)
    return head.to(device).eval(), dict(entry)
