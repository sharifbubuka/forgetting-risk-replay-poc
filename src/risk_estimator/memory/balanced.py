from __future__ import annotations

import pandas as pd

from .risk_based import create_memory_subset_from_indices

def select_class_balanced_high_risk_indices(
    risk_df: pd.DataFrame,
    memory_size: int,
    risk_col: str,
    class_col: str = "original_label",
) -> list[int]:
    """
    Select high-risk samples with an equal budget per class.
    """
    selected_indices = []

    classes = sorted(risk_df[class_col].unique())
    per_class_budget = memory_size // len(classes)

    for cls in classes:
        cls_df = risk_df[risk_df[class_col] == cls]

        cls_selected = (
            cls_df.sort_values(risk_col, ascending=False)
            .head(per_class_budget)["index"]
            .tolist()
        )

        selected_indices.extend(cls_selected)

    remaining = memory_size - len(selected_indices)

    if remaining > 0:
        already_selected = set(selected_indices)

        extra = (
            risk_df[~risk_df["index"].isin(already_selected)]
            .sort_values(risk_col, ascending=False)
            .head(remaining)["index"]
            .tolist()
        )

        selected_indices.extend(extra)

    return selected_indices


def create_class_balanced_high_risk_memory_dataset(
    risk_df: pd.DataFrame,
    base_dataset,
    memory_size: int,
    risk_col: str,
):
    """Create class-balanced high-risk replay memory."""
    selected_indices = select_class_balanced_high_risk_indices(
        risk_df=risk_df,
        memory_size=memory_size,
        risk_col=risk_col,
    )

    memory_dataset = create_memory_subset_from_indices(
        base_dataset,
        selected_indices,
    )

    return memory_dataset, selected_indices
