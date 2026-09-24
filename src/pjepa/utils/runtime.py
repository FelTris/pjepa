import json
import os
import random
from pathlib import Path

import numpy as np
import torch


DEFAULT_SEED = 1538574472


def resolve_device(device_name="auto"):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def seed_everything(seed=DEFAULT_SEED, deterministic_cudnn=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
    return seed


def load_json(path):
    """Load a recipe with portable data/checkpoint/output roots.

    Roots may be set in the JSON or through PJEPA_DATA_ROOT,
    PJEPA_CHECKPOINT_ROOT, and PJEPA_OUTPUT_ROOT. Relative roots are
    interpreted from the current working directory, never a research repo.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    defaults = {
        "data_root": "data",
        "checkpoint_root": "checkpoints/weights",
        "output_root": "outputs",
    }
    env = {
        "data_root": "PJEPA_DATA_ROOT",
        "checkpoint_root": "PJEPA_CHECKPOINT_ROOT",
        "output_root": "PJEPA_OUTPUT_ROOT",
    }
    roots = {
        key: str(
            Path(os.environ.get(env[key], config.get("roots", {}).get(key, value)))
            .expanduser()
            .resolve()
        )
        for key, value in defaults.items()
    }

    def expand(value):
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, str):
            for key, root in roots.items():
                value = value.replace("{" + key + "}", root)
        return value

    return expand(config)


def resolve_path(path_value, root=""):
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((Path(root) / path).resolve()) if root else str(path)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def prepare_generator(name, generator_cls, data_path, reset=True, **generator_kwargs):
    generator = generator_cls(**generator_kwargs)
    generator.read_data(data_path)
    items = getattr(generator, "list_of_examples", getattr(generator, "samples", []))
    print(f"{name} examples found: {len(items)}")
    if reset:
        generator.reset()
    return generator
