"""Historical combined metric used for checkpoint selection."""

import math


def _ltcontext_combined_metric(accuracy, edit_score, f1_10, f1_25, f1_50, accuracy_no_bg=None):
    score = float(accuracy)
    if accuracy_no_bg is not None:
        accuracy_no_bg = float(accuracy_no_bg)
        if math.isfinite(accuracy_no_bg):
            score += accuracy_no_bg
    for value in (edit_score, f1_10, f1_25, f1_50):
        value = float(value)
        if math.isfinite(value):
            score += value / 100.0
    return score
