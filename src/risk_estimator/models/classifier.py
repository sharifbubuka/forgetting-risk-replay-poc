from __future__ import annotations

import torch.nn as nn

from ..config import EMBEDDING_DIM, HIDDEN_DIM, TOTAL_CLASSES


class ContinualClassifier(nn.Module):
    """Small MLP classifier for continual learning over CLIP embeddings."""

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        num_classes: int = TOTAL_CLASSES,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
