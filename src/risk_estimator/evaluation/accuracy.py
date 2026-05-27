from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..config import EVAL_BATCH_SIZE
from ..data.datasets import embedding_collate_fn


@torch.no_grad()
def evaluate_on_task(
    model,
    test_dataset,
    task_id: int,
    task_classes: dict[int, list[int]],
    class_to_local: dict[int, int],
    device: str,
    batch_size: int = EVAL_BATCH_SIZE,
    restrict_to_task_classes: bool = True,
) -> float:
    """
    Evaluate one task.

    If restrict_to_task_classes=True, predictions are restricted to that task's classes.
    This is TIL-style evaluation.
    """
    model.eval()

    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=embedding_collate_fn,
    )

    correct = 0
    total = 0

    original_task_classes = task_classes[task_id]
    local_task_classes = [class_to_local[c] for c in original_task_classes]
    local_task_classes_tensor = torch.tensor(local_task_classes, device=device)

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        logits = model(features)

        if restrict_to_task_classes:
            task_logits = logits[:, local_task_classes_tensor]
            preds_local = task_logits.argmax(dim=-1)
            preds = local_task_classes_tensor[preds_local]
        else:
            preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


@torch.no_grad()
def evaluate_on_task_seen_classes(
    model,
    test_dataset,
    seen_task_ids: list[int],
    task_classes: dict[int, list[int]],
    class_to_local: dict[int, int],
    device: str,
    batch_size: int = EVAL_BATCH_SIZE,
) -> float:
    """
    Evaluate one task in class-incremental mode.

    Predictions are restricted to all classes from seen_task_ids.
    """
    model.eval()

    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=embedding_collate_fn,
    )

    seen_original_classes = []

    for seen_task_id in seen_task_ids:
        seen_original_classes.extend(task_classes[seen_task_id])

    seen_local_classes = [class_to_local[c] for c in seen_original_classes]
    seen_local_classes_tensor = torch.tensor(seen_local_classes, device=device)

    correct = 0
    total = 0

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        logits = model(features)

        seen_logits = logits[:, seen_local_classes_tensor]
        preds_seen_pos = seen_logits.argmax(dim=-1)
        preds = seen_local_classes_tensor[preds_seen_pos]

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


@torch.no_grad()
def evaluate_seen_tasks(
    model,
    test_emb_datasets: dict[int, object],
    max_task_id: int,
    task_classes: dict[int, list[int]],
    class_to_local: dict[int, int],
    device: str,
    restrict_to_task_classes: bool = True,
) -> dict[int, float]:
    """
    Evaluate tasks 0..max_task_id.

    If restrict_to_task_classes=True, uses TIL-style evaluation.
    Otherwise, uses full classifier outputs.
    """
    results = {}

    for task_id in range(max_task_id + 1):
        acc = evaluate_on_task(
            model=model,
            test_dataset=test_emb_datasets[task_id],
            task_id=task_id,
            task_classes=task_classes,
            class_to_local=class_to_local,
            device=device,
            restrict_to_task_classes=restrict_to_task_classes,
        )

        results[task_id] = acc

    return results


@torch.no_grad()
def evaluate_seen_tasks_cil(
    model,
    test_emb_datasets: dict[int, object],
    max_task_id: int,
    task_classes: dict[int, list[int]],
    class_to_local: dict[int, int],
    device: str,
) -> dict[int, float]:
    """
    Evaluate tasks 0..max_task_id in CIL mode.

    Each task is evaluated against all seen classes.
    """
    seen_task_ids = list(range(max_task_id + 1))
    results = {}

    for task_id in range(max_task_id + 1):
        acc = evaluate_on_task_seen_classes(
            model=model,
            test_dataset=test_emb_datasets[task_id],
            seen_task_ids=seen_task_ids,
            task_classes=task_classes,
            class_to_local=class_to_local,
            device=device,
        )

        results[task_id] = acc

    return results


def evaluate_task_pair_cil(
    model,
    test_emb_datasets: dict[int, object],
    task_a: int,
    task_b: int,
    task_classes: dict[int, list[int]],
    class_to_local: dict[int, int],
    device: str,
) -> dict:
    """
    Evaluate two tasks in CIL mode.

    Used for Task 1 -> Task 2 replay comparison.
    """
    seen_task_ids = [task_a, task_b]

    task_a_acc = evaluate_on_task_seen_classes(
        model=model,
        test_dataset=test_emb_datasets[task_a],
        seen_task_ids=seen_task_ids,
        task_classes=task_classes,
        class_to_local=class_to_local,
        device=device,
    )

    task_b_acc = evaluate_on_task_seen_classes(
        model=model,
        test_dataset=test_emb_datasets[task_b],
        seen_task_ids=seen_task_ids,
        task_classes=task_classes,
        class_to_local=class_to_local,
        device=device,
    )

    avg_acc = (task_a_acc + task_b_acc) / 2

    return {
        "task_a_acc": task_a_acc,
        "task_b_acc": task_b_acc,
        "avg_acc": avg_acc,
    }
