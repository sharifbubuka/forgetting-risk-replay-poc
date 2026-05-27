from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import EVAL_BATCH_SIZE
from ..data.datasets import embedding_collate_fn


@torch.no_grad()
def compute_margin_risk(
    model,
    dataset,
    device: str,
    batch_size: int = EVAL_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Compute margin-based forgetting risk over the full output space.

    Lower prediction margin is treated as higher risk.
    """
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=embedding_collate_fn,
    )

    rows = []

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        logits = model(features)
        probs = F.softmax(logits, dim=-1)

        correct_probs = probs[
            torch.arange(labels.size(0), device=device),
            labels,
        ]

        probs_without_correct = probs.clone()
        probs_without_correct[
            torch.arange(labels.size(0), device=device),
            labels,
        ] = -1.0

        best_wrong_probs = probs_without_correct.max(dim=-1).values
        margin = correct_probs - best_wrong_probs

        risk = 1.0 - margin

        for i in range(labels.size(0)):
            rows.append(
                {
                    "index": int(batch["indices"][i]),
                    "task_id": int(batch["task_ids"][i]),
                    "label": int(labels[i].detach().cpu()),
                    "original_label": int(batch["original_labels"][i]),
                    "margin": float(margin[i].detach().cpu()),
                    "risk_margin": float(risk[i].detach().cpu()),
                }
            )

    return pd.DataFrame(rows)
