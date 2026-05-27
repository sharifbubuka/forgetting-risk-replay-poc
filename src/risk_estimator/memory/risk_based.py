from __future__ import annotations

import pandas as pd
from torch.utils.data import Subset


def create_memory_subset_from_indices(dataset, selected_original_indices: list[int]):
    """
    Create a dataset subset using original sample indices.

    The dataset is expected to return items with an 'index' field.
    """
    selected_original_indices = set(selected_original_indices)
    local_indices = []

    for local_idx in range(len(dataset)):
        item = dataset[local_idx]
        original_index = int(item["index"])

        if original_index in selected_original_indices:
            local_indices.append(local_idx)

    return Subset(dataset, local_indices)


def select_top_risk_indices(
    risk_df: pd.DataFrame,
    risk_col: str,
    memory_size: int,
    ascending: bool = False,
) -> list[int]:
    """
    Select top-k sample indices by risk score.

    ascending=False selects highest risk.
    ascending=True selects lowest risk.
    """
    return (
        risk_df.sort_values(risk_col, ascending=ascending)
        .head(memory_size)["index"]
        .tolist()
    )


def create_top_risk_memory_dataset(
    risk_df: pd.DataFrame,
    base_dataset,
    risk_col: str,
    memory_size: int,
    ascending: bool = False,
):
    """Create a replay memory dataset from a risk score."""
    selected_indices = select_top_risk_indices(
        risk_df=risk_df,
        risk_col=risk_col,
        memory_size=memory_size,
        ascending=ascending,
    )

    memory_dataset = create_memory_subset_from_indices(
        base_dataset,
        selected_indices,
    )

    return memory_dataset, selected_indices