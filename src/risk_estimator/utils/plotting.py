"""
Plotting helpers for experiment outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_current_figure(path: str | Path, dpi: int = 200) -> None:
    """Save the current matplotlib figure and create parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_memory_distribution(
    dist_df: pd.DataFrame,
    title: str,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Plot replay-memory class distribution."""
    plt.figure(figsize=(10, 4))
    plt.bar(dist_df["class_name"], dist_df["count"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of memory samples")
    plt.title(title)
    plt.grid(axis="y")

    if save_path is not None:
        save_current_figure(save_path)

    if show:
        plt.show()


def plot_bar_metric(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    save_path: str | Path | None = None,
    ylim: tuple[float, float] | None = None,
    rotation: int = 35,
    figsize: tuple[int, int] = (12, 5),
    show: bool = True,
):
    """Generic bar plot for experiment result tables."""
    plt.figure(figsize=figsize)
    plt.bar(df[x_col], df[y_col])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=rotation, ha="right")
    plt.grid(axis="y")

    if ylim is not None:
        plt.ylim(*ylim)

    if save_path is not None:
        save_current_figure(save_path)

    if show:
        plt.show()


def plot_accuracy_matrix_over_time(
    accuracy_matrix: np.ndarray,
    num_tasks: int,
    title: str,
    ylabel: str = "Accuracy",
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Plot task accuracy curves over sequential training."""
    plt.figure(figsize=(8, 5))

    for task_id in range(num_tasks):
        y = accuracy_matrix[:, task_id]
        x = np.arange(1, num_tasks + 1)
        plt.plot(x, y, marker="o", label=f"Task {task_id + 1}")

    plt.xlabel("After training task")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(np.arange(1, num_tasks + 1))
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        save_current_figure(save_path)

    if show:
        plt.show()


def plot_risk_vs_forgetting(
    df: pd.DataFrame,
    risk_col: str,
    target_col: str = "loss_increase",
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Scatter plot of predicted risk against actual forgetting."""
    temp = df[[risk_col, target_col]].dropna()

    plt.figure(figsize=(6, 4))
    plt.scatter(temp[risk_col], temp[target_col], alpha=0.4)
    plt.xlabel(risk_col)
    plt.ylabel(target_col)
    plt.title(title or f"{risk_col} vs {target_col}")
    plt.grid(True)

    if save_path is not None:
        save_current_figure(save_path)

    if show:
        plt.show()


def plot_temporal_summary(
    summary_df: pd.DataFrame,
    y_col: str,
    title: str,
    ylabel: str,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Plot temporal forgetting/risk summary by previous task."""
    plt.figure(figsize=(7, 4))
    plt.bar(summary_df["task_name"], summary_df[y_col])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y")

    if save_path is not None:
        save_current_figure(save_path)

    if show:
        plt.show()