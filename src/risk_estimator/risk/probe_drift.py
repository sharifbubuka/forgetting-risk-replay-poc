from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import EVAL_BATCH_SIZE
from ..data.datasets import embedding_collate_fn


@torch.no_grad()
def compute_probe_drift_risk(
    model_before,
    probe_model,
    dataset,
    device: str,
    batch_size: int = EVAL_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Compute logit and probability drift between model_before and probe_model.

    Higher drift is treated as higher forgetting risk.
    """
    model_before.eval()
    probe_model.eval()

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

        logits_before = model_before(features)
        logits_probe = probe_model(features)

        logits_before_norm = F.normalize(logits_before, dim=-1)
        logits_probe_norm = F.normalize(logits_probe, dim=-1)

        cosine_sim = (logits_before_norm * logits_probe_norm).sum(dim=-1)
        logit_drift = 1.0 - cosine_sim

        probs_before = F.softmax(logits_before, dim=-1)
        probs_probe = F.softmax(logits_probe, dim=-1)

        prob_l1_drift = torch.abs(probs_before - probs_probe).sum(dim=-1)

        for i in range(labels.size(0)):
            rows.append(
                {
                    "index": int(batch["indices"][i]),
                    "task_id": int(batch["task_ids"][i]),
                    "label": int(labels[i].detach().cpu()),
                    "original_label": int(batch["original_labels"][i]),
                    "risk_probe_logit_drift": float(logit_drift[i].detach().cpu()),
                    "risk_probe_prob_drift": float(prob_l1_drift[i].detach().cpu()),
                }
            )

    return pd.DataFrame(rows)
