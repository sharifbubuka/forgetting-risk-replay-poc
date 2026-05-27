from __future__ import annotations

import pandas as pd

def memory_class_distribution(dataset, class_names: list[str]) -> pd.DataFrame:
    """Return class distribution of a memory dataset."""
    labels = []

    for i in range(len(dataset)):
        item = dataset[i]
        labels.append(int(item["original_label"]))

    dist = pd.Series(labels).value_counts().sort_index()
    dist_df = dist.reset_index()
    dist_df.columns = ["original_label", "count"]
    dist_df["class_name"] = dist_df["original_label"].apply(
        lambda x: class_names[x]
    )

    return dist_df