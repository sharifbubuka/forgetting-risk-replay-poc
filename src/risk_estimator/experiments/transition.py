"""
Transition-level forgetting-risk analysis.

A transition means:
previous tasks -> current task
for example:
Task 1 + Task 2 -> Task 3
"""

from __future__ import annotations

import copy

import pandas as pd

from ..config import (
    BATCH_SIZE,
    EPOCHS_PER_TASK,
    LEARNING_RATE,
    PROBE_STEPS,
    SEED,
)
from ..data import CombinedEmbeddingDataset
from ..evaluation import (
    compute_actual_forgetting_for_transition,
    evaluate_seen_tasks,
    evaluate_seen_tasks_cil,
)
from ..risk import (
    add_corrected_probe_risk,
    add_old_combined_risk,
    compute_average_new_task_gradient,
    compute_gradient_conflict_risk,
    compute_margin_risk,
    compute_probe_drift_risk,
    correlation_report,
)
from ..training import (
    create_probe_model,
    train_on_task,
    train_until_before_task,
)


def get_previous_tasks_dataset(
    current_task_id: int,
    train_emb_datasets: dict[int, object],
):
    """
    Return combined dataset containing all tasks before current_task_id.
    """
    assert current_task_id > 0, "There are no previous tasks for Task 1."

    previous_datasets = [
        train_emb_datasets[task_id]
        for task_id in range(current_task_id)
    ]

    return CombinedEmbeddingDataset(previous_datasets)


def compute_transition_risks(
    model_before,
    old_dataset,
    current_dataset,
    device: str,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    probe_steps: int = PROBE_STEPS,
    compute_gradient: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Compute proactive forgetting-risk scores before training on current task.
    """
    margin_df = compute_margin_risk(
        model=model_before,
        dataset=old_dataset,
        device=device,
    )

    probe_model = create_probe_model(
        model=model_before,
        new_task_dataset=current_dataset,
        device=device,
        probe_steps=probe_steps,
        batch_size=batch_size,
        lr=lr,
    )

    probe_df = compute_probe_drift_risk(
        model_before=model_before,
        probe_model=probe_model,
        dataset=old_dataset,
        device=device,
    )

    risk_df = margin_df.merge(
        probe_df,
        on=["index", "task_id", "label", "original_label"],
        how="left",
    )

    if compute_gradient:
        avg_new_grad = compute_average_new_task_gradient(
            model=model_before,
            new_task_dataset=current_dataset,
            device=device,
            num_batches=5,
            batch_size=batch_size,
        )

        grad_df = compute_gradient_conflict_risk(
            model=model_before,
            old_task_dataset=old_dataset,
            avg_new_task_grad=avg_new_grad,
            device=device,
            max_samples=500,
            batch_size=1,
            show_progress=show_progress,
        )

        risk_df = risk_df.merge(
            grad_df,
            on=["index", "task_id", "label", "original_label"],
            how="left",
        )

    risk_df = add_corrected_probe_risk(risk_df)
    risk_df = add_old_combined_risk(risk_df)

    return risk_df


def analyze_transition_forgetting_risk(
    current_task_id: int,
    train_emb_datasets: dict[int, object],
    test_emb_datasets: dict[int, object],
    task_classes: dict[int, list[int]],
    class_to_local: dict[int, int],
    device: str,
    epochs_before: int = EPOCHS_PER_TASK,
    epochs_current: int = EPOCHS_PER_TASK,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    probe_steps: int = PROBE_STEPS,
    compute_gradient: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Full proactive forgetting-risk analysis for one transition.

    Example:
    current_task_id=2 means:
    Task 1 + Task 2 -> Task 3
    """
    assert current_task_id > 0, "Need at least one previous task."

    if verbose:
        print("\n" + "#" * 70)
        print(f"Analyzing transition: previous tasks -> Task {current_task_id + 1}")
        print("#" * 70)

    model_before = train_until_before_task(
        target_task_id=current_task_id,
        train_emb_datasets=train_emb_datasets,
        device=device,
        seed=SEED,
        epochs_per_task=epochs_before,
        batch_size=batch_size,
        lr=lr,
        verbose=verbose,
    )

    old_dataset = get_previous_tasks_dataset(
        current_task_id=current_task_id,
        train_emb_datasets=train_emb_datasets,
    )

    current_dataset = train_emb_datasets[current_task_id]
    old_task_ids = list(range(current_task_id))

    if verbose:
        print("\nComputing proactive risk scores...")

    risk_df = compute_transition_risks(
        model_before=model_before,
        old_dataset=old_dataset,
        current_dataset=current_dataset,
        device=device,
        batch_size=batch_size,
        lr=lr,
        probe_steps=probe_steps,
        compute_gradient=compute_gradient,
    )

    if verbose:
        print(f"\nTraining on Task {current_task_id + 1} without replay...")

    model_after = copy.deepcopy(model_before).to(device)

    model_after = train_on_task(
        model=model_after,
        train_dataset=current_dataset,
        device=device,
        epochs=epochs_current,
        batch_size=batch_size,
        lr=lr,
        verbose=verbose,
    )

    if verbose:
        print("\nComputing actual forgetting...")

    actual_forgetting_df = compute_actual_forgetting_for_transition(
        model_before=model_before,
        model_after=model_after,
        old_dataset=old_dataset,
        old_task_ids=old_task_ids,
        device=device,
    )

    eval_df = actual_forgetting_df.merge(
        risk_df,
        on=["index", "task_id", "label", "original_label"],
        how="left",
    )

    score_cols = [
        "risk_margin",
        "risk_probe_logit_drift",
        "risk_probe_prob_drift",
        "combined_risk",
        "corrected_probe_risk",
    ]

    if compute_gradient:
        score_cols.append("risk_gradient_conflict")

    corr_df = correlation_report(
        df=eval_df,
        target_col="loss_increase",
        score_cols=score_cols,
    )

    acc_before_til = evaluate_seen_tasks(
        model=model_before,
        test_emb_datasets=test_emb_datasets,
        max_task_id=current_task_id - 1,
        task_classes=task_classes,
        class_to_local=class_to_local,
        device=device,
        restrict_to_task_classes=True,
    )

    acc_after_til = evaluate_seen_tasks(
        model=model_after,
        test_emb_datasets=test_emb_datasets,
        max_task_id=current_task_id,
        task_classes=task_classes,
        class_to_local=class_to_local,
        device=device,
        restrict_to_task_classes=True,
    )

    acc_before_cil = evaluate_seen_tasks_cil(
        model=model_before,
        test_emb_datasets=test_emb_datasets,
        max_task_id=current_task_id - 1,
        task_classes=task_classes,
        class_to_local=class_to_local,
        device=device,
    )

    acc_after_cil = evaluate_seen_tasks_cil(
        model=model_after,
        test_emb_datasets=test_emb_datasets,
        max_task_id=current_task_id,
        task_classes=task_classes,
        class_to_local=class_to_local,
        device=device,
    )

    return {
        "current_task_id": current_task_id,
        "model_before": model_before,
        "model_after": model_after,
        "old_dataset": old_dataset,
        "current_dataset": current_dataset,
        "risk_df": risk_df,
        "actual_forgetting_df": actual_forgetting_df,
        "eval_df": eval_df,
        "corr_df": corr_df,
        "acc_before_til": acc_before_til,
        "acc_after_til": acc_after_til,
        "acc_before_cil": acc_before_cil,
        "acc_after_cil": acc_after_cil,
    }
