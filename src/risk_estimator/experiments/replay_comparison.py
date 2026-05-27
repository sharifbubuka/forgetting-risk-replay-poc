from __future__ import annotations

import copy

import pandas as pd

from ..config import (
    BATCH_SIZE,
    EPOCHS_PER_TASK,
    LEARNING_RATE,
    MEMORY_SIZE,
    SEED,
)
from ..data import CombinedEmbeddingDataset
from ..evaluation import evaluate_on_task_seen_classes
from ..memory import (
    create_class_balanced_high_risk_memory_dataset,
    create_embedding_diverse_high_risk_memory_dataset,
    create_hybrid_risk_random_memory_dataset,
    create_random_memory_dataset,
    create_top_risk_memory_dataset,
)
from ..training import train_on_task


def train_with_optional_replay(
    model_before,
    current_dataset,
    replay_dataset,
    device: str,
    epochs: int = EPOCHS_PER_TASK,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    verbose: bool = True,
):
    """
    Train on current task, optionally with replay memory.
    """
    model = copy.deepcopy(model_before).to(device)

    if replay_dataset is None:
        train_dataset = current_dataset
    else:
        train_dataset = CombinedEmbeddingDataset([current_dataset, replay_dataset])

    model = train_on_task(
        model=model,
        train_dataset=train_dataset,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=verbose,
    )

    return model


def evaluate_two_task_cil(
    model,
    test_emb_datasets: dict[int, object],
    task_a: int,
    task_b: int,
    task_classes: dict[int, list[int]],
    class_to_local: dict[int, int],
    device: str,
    method_name: str,
) -> dict:
    """
    Evaluate two tasks in CIL mode.
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

    return {
        "method": method_name,
        "task1_acc_after_task2_cil": task_a_acc,
        "task2_acc_after_task2_cil": task_b_acc,
        "avg_acc_cil": (task_a_acc + task_b_acc) / 2,
    }


def build_main_replay_memories(
    risk_df: pd.DataFrame,
    base_old_dataset,
    memory_size: int = MEMORY_SIZE,
    risk_col: str = "corrected_probe_risk",
    seed: int = SEED,
) -> dict[str, object]:
    """
    Build the main replay memories used in the PoC.
    """
    memories = {}

    random_memory, _ = create_random_memory_dataset(
        risk_df=risk_df,
        base_dataset=base_old_dataset,
        memory_size=memory_size,
        seed=seed,
    )
    memories["Random replay"] = random_memory

    corrected_probe_memory, _ = create_top_risk_memory_dataset(
        risk_df=risk_df,
        base_dataset=base_old_dataset,
        risk_col=risk_col,
        memory_size=memory_size,
        ascending=False,
    )
    memories["Corrected-probe replay"] = corrected_probe_memory

    class_balanced_memory, _ = create_class_balanced_high_risk_memory_dataset(
        risk_df=risk_df,
        base_dataset=base_old_dataset,
        memory_size=memory_size,
        risk_col=risk_col,
    )
    memories["Class-balanced corrected-probe replay"] = class_balanced_memory

    embedding_diverse_memory, _ = create_embedding_diverse_high_risk_memory_dataset(
        risk_df=risk_df,
        base_dataset=base_old_dataset,
        risk_col=risk_col,
        memory_size=memory_size,
        candidate_multiplier=4,
    )
    memories["Embedding-diverse corrected-probe replay"] = embedding_diverse_memory

    for risk_fraction in [0.25, 0.50, 0.75]:
        hybrid_memory, _ = create_hybrid_risk_random_memory_dataset(
            risk_df=risk_df,
            base_dataset=base_old_dataset,
            memory_size=memory_size,
            risk_col=risk_col,
            risk_fraction=risk_fraction,
            seed=seed,
        )

        name = f"Hybrid corrected-probe {int(risk_fraction * 100)}% risk"
        memories[name] = hybrid_memory

    return memories


def run_replay_comparison_for_two_tasks(
    model_before_task_b,
    current_dataset,
    risk_df: pd.DataFrame,
    base_old_dataset,
    test_emb_datasets: dict[int, object],
    task_classes: dict[int, list[int]],
    class_to_local: dict[int, int],
    device: str,
    task_a: int = 0,
    task_b: int = 1,
    memory_size: int = MEMORY_SIZE,
    risk_col: str = "corrected_probe_risk",
    epochs: int = EPOCHS_PER_TASK,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    """
    Build memories, train replay models, and evaluate them in CIL mode.
    """
    replay_memories = build_main_replay_memories(
        risk_df=risk_df,
        base_old_dataset=base_old_dataset,
        memory_size=memory_size,
        risk_col=risk_col,
    )

    trained_models = {}

    rows = []

    no_replay_model = train_with_optional_replay(
        model_before=model_before_task_b,
        current_dataset=current_dataset,
        replay_dataset=None,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=verbose,
    )

    trained_models["No replay"] = no_replay_model

    rows.append(
        evaluate_two_task_cil(
            model=no_replay_model,
            test_emb_datasets=test_emb_datasets,
            task_a=task_a,
            task_b=task_b,
            task_classes=task_classes,
            class_to_local=class_to_local,
            device=device,
            method_name="No replay",
        )
    )

    for method_name, replay_dataset in replay_memories.items():
        if verbose:
            print("\n" + "=" * 70)
            print(method_name)
            print("=" * 70)

        model = train_with_optional_replay(
            model_before=model_before_task_b,
            current_dataset=current_dataset,
            replay_dataset=replay_dataset,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            verbose=verbose,
        )

        trained_models[method_name] = model

        rows.append(
            evaluate_two_task_cil(
                model=model,
                test_emb_datasets=test_emb_datasets,
                task_a=task_a,
                task_b=task_b,
                task_classes=task_classes,
                class_to_local=class_to_local,
                device=device,
                method_name=method_name,
            )
        )

    results_df = pd.DataFrame(rows)

    task_a_acc_before = evaluate_on_task_seen_classes(
        model=model_before_task_b,
        test_dataset=test_emb_datasets[task_a],
        seen_task_ids=[task_a],
        task_classes=task_classes,
        class_to_local=class_to_local,
        device=device,
    )

    results_df["task1_acc_before_task2_cil"] = task_a_acc_before
    results_df["task1_forgetting_cil"] = (
        results_df["task1_acc_before_task2_cil"]
        - results_df["task1_acc_after_task2_cil"]
    )

    return results_df, trained_models, replay_memories
