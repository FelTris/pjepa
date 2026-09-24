"""Train an encoder, fit frozen probes, or evaluate saved feature-model heads."""

from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
import torch

from pjepa.evaluation.linear import evaluate_linear
from pjepa.training.experiment import (
    build_data,
    build_trainer,
    load_encoder_into,
    probe_arguments,
    probe_training_data,
    train_features,
)
from pjepa.utils.runtime import load_json, resolve_device, seed_everything


def run(config, *, action, checkpoint=None, linear_head=None, temporal_head=None, output=None):
    runtime = config.get("runtime", {})
    seed_everything(runtime.get("seed", 1538574472), runtime.get("deterministic_cudnn", True))
    device = resolve_device(runtime.get("device", "auto"))
    generators, num_classes, data, manifests = build_data(
        config, include_train=action != "evaluate"
    )
    trainer = build_trainer(config, num_classes)
    trainer.model.to(device)
    options = probe_arguments(config)
    source = options["probe_feature_source"]
    checkpoint = checkpoint or config.get("init", {}).get("checkpoint")
    if action != "train" and source == "student" and not checkpoint:
        raise ValueError("Student features require an encoder checkpoint.")
    if checkpoint and (action == "train" or source == "student"):
        load_encoder_into(trainer, config, checkpoint, device)
    output = (
        Path(output or config["paths"].get("output_root", "outputs"))
        / config["experiment"]["run_name"]
    )
    output.mkdir(parents=True, exist_ok=True)
    if action == "train":
        probe_generator, _ = probe_training_data(config, data, manifests, generators)
        train_features(trainer, config, generators, probe_generator, device, output)
        return {"action": action, "output": str(output)}
    if action == "fit-probe":
        probe_generator, subset = probe_training_data(config, data, manifests, generators)
        result = trainer.run_linear_probe(
            probe_generator, device=device, val_batch_gen=generators["val"], **options
        )
        if result is None:
            raise RuntimeError("No validation statistics returned.")
        torch.save(trainer.lin_probe.state_dict(), output / "linear.pt")
        if trainer.ltcontext_probe is not None:
            torch.save(trainer.ltcontext_probe.state_dict(), output / "temporal.pt")
        result.update(train_subset=subset)
    elif action == "evaluate":
        if not linear_head:
            raise ValueError("Saved-head evaluation requires --linear-head.")
        if temporal_head:
            if options["temporal_probe_kind"] not in {"ltcontext", "causal_ltcontext"}:
                raise ValueError("Choose a temporal probe configuration for --temporal-head.")
            allowed = inspect.signature(trainer.evaluate_loaded_ltcontext_probes).parameters
            args = {k: v for k, v in options.items() if k in allowed}
            result = trainer.evaluate_loaded_ltcontext_probes(
                generators["val"],
                device=device,
                temporal_probe_path=temporal_head,
                linear_probe_path=linear_head,
                **args,
            )
        else:
            result = evaluate_linear(
                trainer, generators["val"], linear_head, device=device, **options
            )
    else:
        raise ValueError(f"Unknown action: {action}")
    with (output / "evaluation.json").open("w") as stream:
        json.dump(result, stream, indent=2)
    with (output / "config.json").open("w") as stream:
        json.dump(config, stream, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "fit-probe", "evaluate"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--linear-head")
    parser.add_argument("--temporal-head")
    parser.add_argument("--output")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_json(args.config)
    if args.device:
        config.setdefault("runtime", {})["device"] = args.device
    run(
        config,
        action=args.action,
        checkpoint=args.checkpoint,
        linear_head=args.linear_head,
        temporal_head=args.temporal_head,
        output=args.output,
    )


if __name__ == "__main__":
    main()
