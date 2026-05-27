from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .risk_based import create_memory_subset_from_indices


def get_features_for_original_indices(dataset, selected_original_indices: list[int]):
    """
    Return feature matrix and metadata for original sample indices.
    """
    selected_original_indices = set(selected_original_indices)

    features = []
    indices = []
    labels = []

    for local_idx in range(len(dataset)):
        item = dataset[local_idx]
        original_index = int(item["index"])

        if original_index in selected_original_indices:
            features.append(item["features"])
            indices.append(original_index)
            labels.append(int(item["original_label"]))

    features = torch.stack(features).float()

    return features, indices, labels


def greedy_farthest_first(
    features: torch.Tensor,
    k: int,
    seed_index: int = 0,
    show_progress: bool = True,
) -> list[int]:
    """
    Select k diverse points using greedy farthest-first in cosine space.
    """
    features = F.normalize(features, dim=-1)
    n = features.size(0)

    if k >= n:
        return list(range(n))

    selected = [seed_index]

    sim_to_selected = features @ features[seed_index].unsqueeze(1)
    min_dist = 1.0 - sim_to_selected.squeeze(1)

    iterator = tqdm(
        range(k - 1),
        desc="Selecting diverse samples",
        leave=False,
        disable=not show_progress,
    )

    for _ in iterator:
        next_idx = torch.argmax(min_dist).item()
        selected.append(next_idx)

        sim = features @ features[next_idx].unsqueeze(1)
        dist = 1.0 - sim.squeeze(1)

        min_dist = torch.minimum(min_dist, dist)

    return selected


def select_embedding_diverse_high_risk_indices(
    risk_df,
    base_dataset,
    risk_col: str,
    memory_size: int,
    candidate_multiplier: int = 4,
    show_progress: bool = True,
) -> list[int]:
    """
    Select diverse samples from a high-risk candidate pool.
    """
    candidate_size = memory_size * candidate_multiplier

    candidate_indices = (
        risk_df.sort_values(risk_col, ascending=False)
        .head(candidate_size)["index"]
        .tolist()
    )

    candidate_features, candidate_original_indices, _ = get_features_for_original_indices(
        base_dataset,
        candidate_indices,
    )

    selected_local_indices = greedy_farthest_first(
        candidate_features,
        k=memory_size,
        seed_index=0,
        show_progress=show_progress,
    )

    selected_indices = [
        candidate_original_indices[i] for i in selected_local_indices
    ]

    return selected_indices


def create_embedding_diverse_high_risk_memory_dataset(
    risk_df,
    base_dataset,
    risk_col: str,
    memory_size: int,
    candidate_multiplier: int = 4,
    show_progress: bool = True,
):
    """Create embedding-diverse high-risk replay memory."""
    selected_indices = select_embedding_diverse_high_risk_indices(
        risk_df=risk_df,
        base_dataset=base_dataset,
        risk_col=risk_col,
        memory_size=memory_size,
        candidate_multiplier=candidate_multiplier,
        show_progress=show_progress,
    )

    memory_dataset = create_memory_subset_from_indices(
        base_dataset,
        selected_indices,
    )

    return memory_dataset, selected_indices
