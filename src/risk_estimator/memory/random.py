from __future__ import annotations

import random

from .risk_based import create_memory_subset_from_indices


def select_random_indices(
    all_indices: list[int],
    memory_size: int,
    seed: int = 42,
) -> list[int]:
    """Select random memory indices."""
    random.seed(seed)
    return random.sample(all_indices, min(memory_size, len(all_indices)))


def create_random_memory_dataset(
    risk_df,
    base_dataset,
    memory_size: int,
    seed: int = 42,
):
    """Create random replay memory using indices from a risk dataframe."""
    all_indices = risk_df["index"].tolist()

    selected_indices = select_random_indices(
        all_indices=all_indices,
        memory_size=memory_size,
        seed=seed,
    )

    memory_dataset = create_memory_subset_from_indices(
        base_dataset,
        selected_indices,
    )

    return memory_dataset, selected_indices
