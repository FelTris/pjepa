from collections import Counter
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


def levenshtein(pred_labels: Sequence[int], gt_labels: Sequence[int]) -> int:
    pred_len = len(pred_labels)
    gt_len = len(gt_labels)
    distance = np.zeros((pred_len + 1, gt_len + 1), dtype=np.int64)
    distance[:, 0] = np.arange(pred_len + 1, dtype=np.int64)
    distance[0, :] = np.arange(gt_len + 1, dtype=np.int64)

    for pred_idx in range(1, pred_len + 1):
        for gt_idx in range(1, gt_len + 1):
            if gt_labels[gt_idx - 1] == pred_labels[pred_idx - 1]:
                distance[pred_idx, gt_idx] = distance[pred_idx - 1, gt_idx - 1]
            else:
                distance[pred_idx, gt_idx] = min(
                    int(distance[pred_idx - 1, gt_idx]) + 1,
                    int(distance[pred_idx, gt_idx - 1]) + 1,
                    int(distance[pred_idx - 1, gt_idx - 1]) + 1,
                )
    return int(distance[pred_len, gt_len])


def get_segments(frame_labels: np.ndarray, bg_class: int) -> Tuple[List[int], List[int], List[int]]:
    labels = np.asarray(frame_labels, dtype=np.int64)
    if labels.size == 0:
        return [], [], []

    seg_labels: List[int] = []
    seg_starts: List[int] = []
    seg_ends: List[int] = []

    last_label = int(labels[0])
    if last_label != bg_class:
        seg_labels.append(last_label)
        seg_starts.append(0)

    for idx in range(1, int(labels.shape[0])):
        current = int(labels[idx])
        if current == last_label:
            continue
        if current != bg_class:
            seg_labels.append(current)
            seg_starts.append(idx)
        if last_label != bg_class:
            seg_ends.append(idx)
        last_label = current

    if last_label != bg_class:
        seg_ends.append(int(labels.shape[0]))

    return seg_labels, seg_starts, seg_ends


def edit_score_ms_tcn(pred_frame: np.ndarray, gt_frame: np.ndarray, bg_class: int) -> float:
    pred_labels, _, _ = get_segments(pred_frame, bg_class)
    gt_labels, _, _ = get_segments(gt_frame, bg_class)
    if len(pred_labels) == 0 and len(gt_labels) == 0:
        return 100.0
    distance = levenshtein(pred_labels, gt_labels)
    denom = max(len(pred_labels), len(gt_labels), 1)
    return (1.0 - float(distance) / float(denom)) * 100.0


def f_score_ms_tcn(
    pred_frame: np.ndarray, gt_frame: np.ndarray, overlap: float, bg_class: int
) -> Tuple[float, float, float]:
    pred_label, pred_start, pred_end = get_segments(pred_frame, bg_class)
    gt_label, gt_start, gt_end = get_segments(gt_frame, bg_class)

    true_pos = 0.0
    false_pos = 0.0
    hits = np.zeros(len(gt_label), dtype=np.int64)

    for idx in range(len(pred_label)):
        if len(gt_label) == 0:
            false_pos += 1.0
            continue
        intersection = np.minimum(pred_end[idx], gt_end) - np.maximum(pred_start[idx], gt_start)
        union = np.maximum(pred_end[idx], gt_end) - np.minimum(pred_start[idx], gt_start)
        iou = (intersection / np.maximum(union, 1.0e-12)) * np.array(
            [pred_label[idx] == gt_label[j] for j in range(len(gt_label))]
        )
        matched = int(np.argmax(iou))
        if iou[matched] >= overlap and not hits[matched]:
            true_pos += 1.0
            hits[matched] = 1
        else:
            false_pos += 1.0

    false_neg = float(len(gt_label) - int(np.sum(hits)))
    return true_pos, false_pos, false_neg


def evaluate_sequences(
    predictions: Mapping[str, np.ndarray],
    ground_truth: Mapping[str, np.ndarray],
    *,
    label_names: Mapping[int, str],
    bg_label_id: int,
    overlaps: Sequence[float],
) -> Dict[str, object]:
    common_ids = sorted(set(predictions.keys()) & set(ground_truth.keys()))
    pred_only = sorted(set(predictions.keys()) - set(ground_truth.keys()))
    gt_only = sorted(set(ground_truth.keys()) - set(predictions.keys()))

    total_frames = 0
    correct_frames = 0
    total_frames_non_idle = 0
    correct_frames_non_idle = 0

    gt_support = Counter()
    pred_support = Counter()
    true_pos = Counter()
    idle_fp_by_label = Counter()
    idle_frames = 0
    idle_fp_frames = 0
    edit_scores: List[float] = []
    overlap_counts = {float(ov): {"tp": 0.0, "fp": 0.0, "fn": 0.0} for ov in overlaps}

    for video_id in common_ids:
        pred = np.asarray(predictions[video_id], dtype=np.int64)
        gt = np.asarray(ground_truth[video_id], dtype=np.int64)
        if pred.shape != gt.shape:
            raise ValueError(
                f"Prediction and GT shapes differ for {video_id}: {pred.shape} vs {gt.shape}"
            )

        total_frames += int(gt.shape[0])
        correct_frames += int(np.sum(pred == gt))
        non_idle_mask = gt != bg_label_id
        total_frames_non_idle += int(np.sum(non_idle_mask))
        if np.any(non_idle_mask):
            correct_frames_non_idle += int(np.sum((pred == gt) & non_idle_mask))

        gt_labels, gt_counts = np.unique(gt, return_counts=True)
        pred_labels, pred_counts = np.unique(pred, return_counts=True)
        tp_labels, tp_counts = np.unique(gt[pred == gt], return_counts=True)
        for label_id, count in zip(gt_labels.tolist(), gt_counts.tolist()):
            gt_support[int(label_id)] += int(count)
        for label_id, count in zip(pred_labels.tolist(), pred_counts.tolist()):
            pred_support[int(label_id)] += int(count)
        for label_id, count in zip(tp_labels.tolist(), tp_counts.tolist()):
            true_pos[int(label_id)] += int(count)

        idle_mask = gt == bg_label_id
        idle_frames += int(np.sum(idle_mask))
        idle_fp_mask = idle_mask & (pred != bg_label_id)
        idle_fp_frames += int(np.sum(idle_fp_mask))
        if np.any(idle_fp_mask):
            labels, counts = np.unique(pred[idle_fp_mask], return_counts=True)
            for label_id, count in zip(labels.tolist(), counts.tolist()):
                idle_fp_by_label[int(label_id)] += int(count)

        edit_scores.append(edit_score_ms_tcn(pred, gt, bg_label_id))
        for ov in overlaps:
            tp, fp, fn = f_score_ms_tcn(pred, gt, float(ov), bg_label_id)
            overlap_counts[float(ov)]["tp"] += tp
            overlap_counts[float(ov)]["fp"] += fp
            overlap_counts[float(ov)]["fn"] += fn

    class_rows = []
    for class_id in sorted(set(gt_support.keys()) | set(pred_support.keys())):
        support = int(gt_support[class_id])
        pred_count = int(pred_support[class_id])
        tp = int(true_pos[class_id])
        acc = float(tp) / float(support) if support > 0 else 0.0
        prec = float(tp) / float(pred_count) if pred_count > 0 else 0.0
        denom_iou = support + pred_count - tp
        iou = float(tp) / float(denom_iou) if denom_iou > 0 else 0.0
        class_rows.append(
            {
                "label_id": int(class_id),
                "label_name": str(label_names.get(int(class_id), f"label_{class_id}")),
                "support_frames": support,
                "pred_frames": pred_count,
                "tp_frames": tp,
                "accuracy": acc,
                "precision": prec,
                "iou": iou,
                "fp_on_idle_frames": int(idle_fp_by_label.get(class_id, 0)),
            }
        )

    non_idle_rows = [
        row for row in class_rows if row["label_id"] != bg_label_id and row["support_frames"] > 0
    ]
    mean_class_acc = (
        float(np.mean([row["accuracy"] for row in non_idle_rows])) if non_idle_rows else 0.0
    )
    mean_iou_non_idle = (
        float(np.mean([row["iou"] for row in non_idle_rows])) if non_idle_rows else 0.0
    )

    f1_by_overlap: Dict[str, float] = {}
    for ov in overlaps:
        tp = overlap_counts[float(ov)]["tp"]
        fp = overlap_counts[float(ov)]["fp"]
        fn = overlap_counts[float(ov)]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_by_overlap[str(ov)] = 100.0 * f1

    return {
        "videos_common": len(common_ids),
        "videos_pred_only": len(pred_only),
        "videos_gt_only": len(gt_only),
        "frames_evaluated": int(total_frames),
        "frames_evaluated_non_idle": int(total_frames_non_idle),
        "frame_accuracy": float(correct_frames) / float(max(total_frames, 1)),
        "frame_accuracy_excluding_idle": float(correct_frames_non_idle)
        / float(max(total_frames_non_idle, 1)),
        "mean_class_accuracy_excluding_idle": mean_class_acc,
        "mean_iou_excluding_idle": mean_iou_non_idle,
        "edit_score": float(np.mean(edit_scores)) if edit_scores else 0.0,
        "f1_overlap": f1_by_overlap,
        "idle_frames": int(idle_frames),
        "idle_fp_frames": int(idle_fp_frames),
        "idle_fp_rate": float(idle_fp_frames) / float(max(idle_frames, 1)),
        "idle_fp_by_label": {str(k): int(v) for k, v in idle_fp_by_label.items()},
        "per_class": class_rows,
    }
