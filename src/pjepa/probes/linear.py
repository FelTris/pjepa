"""Framewise heads with legacy checkpoint parameter names."""

import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class ForegroundActivityLinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_foreground_classes: int):
        super().__init__()
        self.foreground_fc = nn.Linear(in_dim, num_foreground_classes)
        self.activity_fc = nn.Linear(in_dim, 2)

    def forward(self, x):
        return {"foreground": self.foreground_fc(x), "activity": self.activity_fc(x)}
