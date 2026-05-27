from __future__ import annotations

import random

import pandas as pd

from .risk_based import create_memory_subset_from_indices


def select_hybrid_risk_random_indices(
    risk_df: pd.DataFrame,
    memory_size: int,
    risk_col: str,
    risk_fraction: float = 0.5,
    seed: int = 42,
) -> list[int]:
    """
    Select part of memory from high-risk samples and the rest randomly.
    """
    random.seed(seed)

    risk_budget = int(memory_size * risk_fraction)
    random_budget = memory_size - risk_budget

    risk_indices = (
        risk_df.sort_values(risk_col, ascending=False)
        .head(risk_budget)["index"]
        .tolist()
    )

    risk_indices_set = set(risk_indices)

    remaining_indices = (
        risk_df[~risk_df["index"].isin(risk_indices_set)]["index"].tolist()
    )

    random_indices = random.sample(
        remaining_indices,
        min(random_budget, len(remaining_indices)),
    )

    return risk_indices + random_indices


def create_hybrid_risk_random_memory_dataset(
    risk_df: pd.DataFrame,
    base_dataset,
    memory_size: int,
    risk_col: str,
    risk_fraction: float = 0.5,
    seed: int = 42,
):
    """Create hybrid risk + random replay memory."""
    selected_indices = select_hybrid_risk_random_indices(
        risk_df=risk_df,
        memory_size=memory_size,
        risk_col=risk_col,
        risk_fraction=risk_fraction,
        seed=seed,
    )

    memory_dataset = create_memory_subset_from_indices(
        base_dataset,
        selected_indices,
    )

    return memory_dataset, selected_indices
