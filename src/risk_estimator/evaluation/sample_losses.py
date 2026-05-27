from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import EVAL_BATCH_SIZE
from ..data.datasets import embedding_collate_fn


@torch.no_grad()
def compute_sample_losses(
    model,
    dataset,
    device: str,
    batch_size: int = EVAL_BATCH_SIZE,
    restrict_to_task_classes: bool = False,
    task_id: int | None = None,
    task_classes: dict[int, list[int]] | None = None,
    class_to_local: dict[int, int] | None = None,
) -> pd.DataFrame:
    """
    Compute per-sample loss, confidence, predictions, and correctness.

    If restrict_to_task_classes=True, task_id, task_classes, and class_to_local
    must be provided.
    """
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=embedding_collate_fn,
    )

    rows = []

    if restrict_to_task_classes:
        assert task_id is not None
        assert task_classes is not None
        assert class_to_local is not None

        original_task_classes = task_classes[task_id]
        local_task_classes = [class_to_local[c] for c in original_task_classes]
        local_task_classes_tensor = torch.tensor(local_task_classes, device=device)
        label_to_pos = {cls: pos for pos, cls in enumerate(local_task_classes)}

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        logits = model(features)

        if restrict_to_task_classes:
            logits_eval = logits[:, local_task_classes_tensor]

            labels_eval = torch.tensor(
                [label_to_pos[int(y)] for y in labels.detach().cpu()],
                dtype=torch.long,
                device=device,
            )

            losses = F.cross_entropy(logits_eval, labels_eval, reduction="none")
            probs = F.softmax(logits_eval, dim=-1)

            pred_pos = probs.argmax(dim=-1)
            preds = local_task_classes_tensor[pred_pos]

            correct_probs = probs[
                torch.arange(labels_eval.size(0), device=device),
                labels_eval,
            ]

        else:
            losses = F.cross_entropy(logits, labels, reduction="none")
            probs = F.softmax(logits, dim=-1)

            preds = probs.argmax(dim=-1)

            correct_probs = probs[
                torch.arange(labels.size(0), device=device),
                labels,
            ]

        for i in range(labels.size(0)):
            pred_i = int(preds[i].detach().cpu())
            label_i = int(labels[i].detach().cpu())

            rows.append(
                {
                    "index": int(batch["indices"][i]),
                    "task_id": int(batch["task_ids"][i]),
                    "label": label_i,
                    "original_label": int(batch["original_labels"][i]),
                    "loss": float(losses[i].detach().cpu()),
                    "correct_prob": float(correct_probs[i].detach().cpu()),
                    "pred": pred_i,
                    "is_correct": pred_i == label_i,
                }
            )

    return pd.DataFrame(rows)


def compute_actual_sample_forgetting(
    losses_before: pd.DataFrame,
    losses_after: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge before/after losses and compute actual sample-level forgetting.
    """
    forgetting_df = losses_before.merge(
        losses_after,
        on=["index", "task_id", "label", "original_label"],
        suffixes=("_before", "_after"),
    )

    forgetting_df["loss_increase"] = (
        forgetting_df["loss_after"] - forgetting_df["loss_before"]
    )

    forgetting_df["prob_drop"] = (
        forgetting_df["correct_prob_before"] - forgetting_df["correct_prob_after"]
    )

    forgetting_df["forgotten_binary"] = (
        forgetting_df["loss_increase"] > 0
    ).astype(int)

    return forgetting_df


def compute_actual_forgetting_for_transition(
    model_before,
    model_after,
    old_dataset,
    old_task_ids: list[int],
    device: str,
    batch_size: int = EVAL_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Compute sample-level forgetting for all old samples in a transition.

    Uses full classifier outputs, which is closer to CIL-style analysis.
    """
    losses_before = compute_sample_losses(
        model=model_before,
        dataset=old_dataset,
        device=device,
        batch_size=batch_size,
        restrict_to_task_classes=False,
    )

    losses_after = compute_sample_losses(
        model=model_after,
        dataset=old_dataset,
        device=device,
        batch_size=batch_size,
        restrict_to_task_classes=False,
    )

    forgetting_df = compute_actual_sample_forgetting(
        losses_before=losses_before,
        losses_after=losses_after,
    )

    forgetting_df["temporal_distance"] = forgetting_df["task_id"].apply(
        lambda old_task_id: max(old_task_ids) + 1 - old_task_id
    )

    return forgetting_df
