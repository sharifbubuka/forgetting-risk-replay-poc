from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    EPOCHS_PER_TASK,
    HIDDEN_DIM,
    LEARNING_RATE,
    TOTAL_CLASSES,
    WEIGHT_DECAY,
)
from ..data.datasets import embedding_collate_fn
from ..models import ContinualClassifier
from ..utils.seed import set_seed


def train_on_task(
    model,
    train_dataset,
    device: str,
    epochs: int = EPOCHS_PER_TASK,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    verbose: bool = True,
):
    """Train model on one task or combined replay dataset."""
    model.train()

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=embedding_collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        correct = 0

        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()

        if verbose:
            avg_loss = total_loss / total
            acc = correct / total
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"loss={avg_loss:.4f} | train_acc={acc:.4f}"
            )

    return model


def create_fresh_classifier(
    device: str,
    input_dim: int = EMBEDDING_DIM,
    num_classes: int = TOTAL_CLASSES,
    hidden_dim: int = HIDDEN_DIM,
):
    """Create a fresh continual classifier."""
    return ContinualClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
    ).to(device)


def train_until_before_task(
    target_task_id: int,
    train_emb_datasets: dict[int, object],
    device: str,
    seed: int = 42,
    epochs_per_task: int = EPOCHS_PER_TASK,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    verbose: bool = True,
):
    """
    Train sequentially on tasks before target_task_id.

    Example:
    target_task_id=2 means train on Task 1 and Task 2,
    then return model before Task 3.
    """
    assert target_task_id > 0, "target_task_id must be > 0."

    set_seed(seed)

    model = create_fresh_classifier(device=device)

    for task_id in range(target_task_id):
        if verbose:
            print("\n" + "=" * 60)
            print(f"Training base model on Task {task_id + 1}")
            print("=" * 60)

        model = train_on_task(
            model=model,
            train_dataset=train_emb_datasets[task_id],
            device=device,
            epochs=epochs_per_task,
            batch_size=batch_size,
            lr=lr,
            verbose=verbose,
        )

    model.eval()
    return model


def create_probe_model(
    model,
    new_task_dataset,
    device: str,
    probe_steps: int = 10,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
):
    """
    Create a temporary probe model after a few updates on the incoming task.

    Used for proactive forgetting-risk estimation.
    """
    probe_model = copy.deepcopy(model).to(device)
    probe_model.train()

    loader = DataLoader(
        new_task_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=embedding_collate_fn,
    )

    optimizer = torch.optim.AdamW(
        probe_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    step = 0

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        logits = probe_model(features)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        if step >= probe_steps:
            break

    probe_model.eval()
    return probe_model
