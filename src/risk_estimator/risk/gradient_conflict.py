from __future__ import annotations

import copy

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from ..config import BATCH_SIZE
from ..data.datasets import embedding_collate_fn


def get_last_layer_grad_vector(model) -> torch.Tensor | None:
    """Extract gradients from the final Linear layer."""
    grads = []
    final_layer = model.net[-1]

    if final_layer.weight.grad is not None:
        grads.append(final_layer.weight.grad.detach().flatten())

    if final_layer.bias.grad is not None:
        grads.append(final_layer.bias.grad.detach().flatten())

    if len(grads) == 0:
        return None

    return torch.cat(grads)


def compute_batch_gradient(model, features, labels) -> torch.Tensor:
    """Compute gradient vector for one batch."""
    model.zero_grad()

    logits = model(features)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    grad_vec = get_last_layer_grad_vector(model)

    if grad_vec is None:
        raise RuntimeError("No gradients found.")

    return grad_vec.detach().clone()


def compute_average_new_task_gradient(
    model,
    new_task_dataset,
    device: str,
    num_batches: int = 5,
    batch_size: int = BATCH_SIZE,
) -> torch.Tensor:
    """Compute average classifier-head gradient for the incoming task."""
    model_for_grad = copy.deepcopy(model).to(device)
    model_for_grad.train()

    loader = DataLoader(
        new_task_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=embedding_collate_fn,
    )

    grad_vectors = []

    for step, batch in enumerate(loader):
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        grad_vec = compute_batch_gradient(model_for_grad, features, labels)
        grad_vectors.append(grad_vec)

        if step + 1 >= num_batches:
            break

    avg_grad = torch.stack(grad_vectors, dim=0).mean(dim=0)
    avg_grad = F.normalize(avg_grad, dim=0)

    return avg_grad


def compute_gradient_conflict_risk(
    model,
    old_task_dataset,
    avg_new_task_grad: torch.Tensor,
    device: str,
    max_samples: int = 500,
    batch_size: int = 1,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Compute gradient-conflict risk for a subset of old samples.

    risk = -cos(old_sample_gradient, new_task_gradient)
    """
    model_for_grad = copy.deepcopy(model).to(device)
    model_for_grad.train()

    subset_indices = list(range(min(max_samples, len(old_task_dataset))))
    subset = Subset(old_task_dataset, subset_indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=embedding_collate_fn,
    )

    rows = []

    iterator = tqdm(
        loader,
        desc="Gradient conflict",
        leave=False,
        disable=not show_progress,
    )

    for batch in iterator:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        old_grad = compute_batch_gradient(model_for_grad, features, labels)
        old_grad = F.normalize(old_grad, dim=0)

        cosine = torch.dot(old_grad, avg_new_task_grad).item()
        risk = -cosine

        rows.append(
            {
                "index": int(batch["indices"][0]),
                "task_id": int(batch["task_ids"][0]),
                "label": int(batch["labels"][0]),
                "original_label": int(batch["original_labels"][0]),
                "grad_cosine_with_new_task": cosine,
                "risk_gradient_conflict": risk,
            }
        )

    return pd.DataFrame(rows)
