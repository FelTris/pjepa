from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class SurgicalPhaseLinearHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(int(input_dim), int(num_classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class _PhaseTokenDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        if int(features.shape[0]) != int(targets.numel()):
            raise ValueError("Phase feature and target lengths differ.")
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index]


@dataclass
class ExtractedPhaseFeatures:
    features: torch.Tensor
    targets: torch.Tensor
    num_sequences: int


@torch.no_grad()
def extract_frozen_phase_features(
    model,
    loader: DataLoader,
    *,
    device: torch.device,
    feature_source: str,
    use_bfloat16: bool = False,
) -> ExtractedPhaseFeatures:
    feature_source = str(feature_source).lower()
    if feature_source not in {"raw", "student"}:
        raise ValueError("feature_source must be 'raw' or 'student'.")
    if feature_source == "student" and model is None:
        raise ValueError("Student feature extraction requires a P-JEPA model.")

    previous_training = None if model is None else bool(model.training)
    if model is not None:
        model.eval()
    all_features: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    num_sequences = 0
    autocast_enabled = bool(use_bfloat16 and device.type == "cuda")
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True).bool()
        if feature_source == "student":
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                encoded = model.student_enc(
                    features.unsqueeze(1),
                    valid_mask=valid_mask.unsqueeze(1),
                    context_mask=valid_mask.unsqueeze(1),
                    N=1,
                    L=int(features.shape[1]),
                    segment_lengths=None,
                )
            flat_features = encoded.float()
        else:
            flat_features = features.float()
        all_features.append(flat_features[valid_mask].cpu())
        phase = batch["targets"]["phase"]
        all_targets.append(phase[batch["valid_mask"].bool()].long().cpu())
        num_sequences += int(features.shape[0])

    if model is not None and previous_training:
        model.train()
    if not all_features:
        raise RuntimeError("No surgical phase features were extracted.")
    return ExtractedPhaseFeatures(
        features=torch.cat(all_features, dim=0),
        targets=torch.cat(all_targets, dim=0),
        num_sequences=num_sequences,
    )


def frame_accuracy_and_macro_f1(
    predictions: torch.Tensor, targets: torch.Tensor, *, num_classes: int
) -> tuple[float, float]:
    predictions = predictions.reshape(-1).long().cpu()
    targets = targets.reshape(-1).long().cpu()
    if int(predictions.numel()) != int(targets.numel()) or not targets.numel():
        raise ValueError("Predictions and targets must be non-empty and equally sized.")
    accuracy = float((predictions == targets).float().mean().item())
    f1_scores: list[float] = []
    for class_index in range(int(num_classes)):
        predicted_class = predictions == class_index
        target_class = targets == class_index
        true_positive = int((predicted_class & target_class).sum().item())
        false_positive = int((predicted_class & ~target_class).sum().item())
        false_negative = int((~predicted_class & target_class).sum().item())
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return accuracy, float(sum(f1_scores) / len(f1_scores))


@torch.no_grad()
def evaluate_phase_head(
    head: SurgicalPhaseLinearHead,
    dataset: _PhaseTokenDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_classes: int,
    num_workers: int = 0,
) -> dict[str, float]:
    head.eval()
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
    )
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    loss_sum = 0.0
    examples = 0
    for features, phase in loader:
        phase = phase.to(device, non_blocking=True)
        logits = head(features.to(device, non_blocking=True))
        loss_sum += float(F.cross_entropy(logits, phase, reduction="sum").item())
        examples += int(phase.numel())
        predictions.append(logits.argmax(dim=-1).cpu())
        targets.append(phase.cpu())
    accuracy, macro_f1 = frame_accuracy_and_macro_f1(
        torch.cat(predictions), torch.cat(targets), num_classes=num_classes
    )
    return {
        "phase_acc": accuracy,
        "phase_macro_f1": macro_f1,
        "phase_loss": loss_sum / max(1, examples),
    }


def run_frozen_phase_probe(
    model,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    *,
    device: torch.device,
    config: dict[str, Any],
    num_classes: int,
    feature_source: str = "student",
    select_best_on_eval: bool = True,
    log_prefix: str = "Surgical phase",
) -> dict[str, Any]:
    seed = int(config.get("seed", 1538574472))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_data = extract_frozen_phase_features(
        model,
        train_loader,
        device=device,
        feature_source=feature_source,
        use_bfloat16=bool(config.get("use_bfloat16", False)),
    )
    eval_data = extract_frozen_phase_features(
        model,
        eval_loader,
        device=device,
        feature_source=feature_source,
        use_bfloat16=bool(config.get("use_bfloat16", False)),
    )
    train_dataset = _PhaseTokenDataset(train_data.features, train_data.targets)
    eval_dataset = _PhaseTokenDataset(eval_data.features, eval_data.targets)

    input_dim = int(train_data.features.shape[-1])
    head = SurgicalPhaseLinearHead(input_dim, num_classes).to(device)
    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    learning_rate = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            head.parameters(),
            lr=learning_rate,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported probe optimizer: {optimizer_name}")
    milestones = [int(value) for value in config.get("lr_milestones", [])]
    scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=float(config.get("lr_gamma", 0.1)),
        )
        if milestones
        else None
    )
    token_batch_size = int(config.get("token_batch_size", 4096))
    train_token_loader = DataLoader(
        train_dataset,
        batch_size=token_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=int(config.get("token_num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    default_updates_per_epoch = len(train_token_loader)
    updates_per_epoch = int(config.get("updates_per_epoch", default_updates_per_epoch))
    if updates_per_epoch <= 0:
        raise ValueError("Probe updates_per_epoch must be positive.")
    selection_metric = str(config.get("selection_metric", "phase_acc"))
    if selection_metric not in {"phase_acc", "phase_macro_f1"}:
        raise ValueError("Probe selection_metric must be phase_acc or phase_macro_f1.")
    best_value = float("-inf")
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    num_epochs = int(config.get("epochs", 30))
    if num_epochs <= 0:
        raise ValueError("Probe epochs must be positive.")

    for epoch in range(num_epochs):
        head.train()
        loss_sum = 0.0
        examples = 0
        train_iterator = iter(train_token_loader)
        for _ in range(updates_per_epoch):
            try:
                features, phase = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_token_loader)
                features, phase = next(train_iterator)
            features = features.to(device, non_blocking=True)
            phase = phase.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = F.cross_entropy(logits, phase)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().item()) * int(phase.numel())
            examples += int(phase.numel())
        if scheduler is not None:
            scheduler.step()
        train_loss = loss_sum / max(1, examples)

        if select_best_on_eval:
            metrics = evaluate_phase_head(
                head,
                eval_dataset,
                device=device,
                batch_size=token_batch_size,
                num_classes=num_classes,
                num_workers=int(config.get("token_num_workers", 0)),
            )
            metrics["probe_epoch"] = float(epoch + 1)
            metrics["train_loss"] = train_loss
            value = float(metrics[selection_metric])
            if value > best_value:
                best_value = value
                best_metrics = dict(metrics)
                best_state = copy.deepcopy(head.state_dict())
            print(
                f"[{log_prefix} probe {epoch + 1:03d}] loss={train_loss:.6f} "
                f"acc={metrics['phase_acc']:.6f} macro_f1={metrics['phase_macro_f1']:.6f}"
            )
        else:
            print(f"[{log_prefix} probe {epoch + 1:03d}] loss={train_loss:.6f}")

    if select_best_on_eval:
        if best_state is None or best_metrics is None:
            raise RuntimeError("Phase probe did not produce a selected epoch.")
        head.load_state_dict(best_state, strict=True)
    else:
        best_state = copy.deepcopy(head.state_dict())
        best_metrics = evaluate_phase_head(
            head,
            eval_dataset,
            device=device,
            batch_size=token_batch_size,
            num_classes=num_classes,
            num_workers=int(config.get("token_num_workers", 0)),
        )
        best_metrics["probe_epoch"] = float(num_epochs)
        best_metrics["train_loss"] = train_loss
        print(
            f"[{log_prefix} final] acc={best_metrics['phase_acc']:.6f} "
            f"macro_f1={best_metrics['phase_macro_f1']:.6f}"
        )

    return {
        "metrics": best_metrics,
        "head_state_dict": best_state,
        "feature_source": str(feature_source),
        "selection_metric": selection_metric if select_best_on_eval else None,
        "train_tokens": int(train_data.features.shape[0]),
        "eval_tokens": int(eval_data.features.shape[0]),
        "train_sequences": int(train_data.num_sequences),
        "eval_sequences": int(eval_data.num_sequences),
        "updates_per_epoch": int(updates_per_epoch),
        "default_updates_per_epoch": int(default_updates_per_epoch),
    }
