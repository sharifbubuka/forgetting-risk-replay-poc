from __future__ import annotations

import numpy as np
import pandas as pd


def compute_forgetting(accuracy_matrix: np.ndarray) -> tuple[dict[int, dict], float]:
    """
    Compute task-level forgetting from an accuracy matrix.

    accuracy_matrix[t, i] = accuracy on task i after training task t.
    """
    num_tasks = accuracy_matrix.shape[0]
    final_row = accuracy_matrix[-1]

    forgetting_per_task = {}

    for task_id in range(num_tasks - 1):
        task_scores = accuracy_matrix[:, task_id]
        task_scores = task_scores[~np.isnan(task_scores)]

        max_score = np.max(task_scores)
        final_score = final_row[task_id]
        forgetting = max_score - final_score

        forgetting_per_task[task_id] = {
            "max_score": float(max_score),
            "final_score": float(final_score),
            "forgetting": float(forgetting),
        }

    avg_forgetting = float(
        np.mean([item["forgetting"] for item in forgetting_per_task.values()])
    )

    return forgetting_per_task, avg_forgetting


def accuracy_matrix_to_dataframe(
    accuracy_matrix: np.ndarray,
    prefix_after: str = "After T",
    prefix_test: str = "Test T",
) -> pd.DataFrame:
    """Convert accuracy matrix to a readable dataframe."""
    num_tasks = accuracy_matrix.shape[0]

    return pd.DataFrame(
        accuracy_matrix,
        index=[f"{prefix_after}{i + 1}" for i in range(num_tasks)],
        columns=[f"{prefix_test}{i + 1}" for i in range(num_tasks)],
    )


def summarize_temporal_forgetting(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize actual forgetting and predicted risk by old task.
    """
    summary = (
        eval_df.groupby("task_id")
        .agg(
            avg_loss_increase=("loss_increase", "mean"),
            avg_prob_drop=("prob_drop", "mean"),
            forgotten_rate=("forgotten_binary", "mean"),
            avg_combined_risk=("combined_risk", "mean"),
            n=("index", "count"),
        )
        .reset_index()
    )

    summary["task_name"] = summary["task_id"].apply(lambda x: f"Task {x + 1}")

    return summary