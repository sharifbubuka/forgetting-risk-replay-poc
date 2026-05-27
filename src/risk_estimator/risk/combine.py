from __future__ import annotations

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ..config import (
    CORRECTED_PROBE_LOGIT_WEIGHT,
    CORRECTED_PROBE_PROB_WEIGHT,
)


def minmax_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a pandas series."""
    s = series.copy()
    min_val = s.min()
    max_val = s.max()

    if max_val - min_val < 1e-8:
        return s * 0.0

    return (s - min_val) / (max_val - min_val)


def add_corrected_probe_risk(
    df: pd.DataFrame,
    prob_col: str = "risk_probe_prob_drift",
    logit_col: str = "risk_probe_logit_drift",
    output_col: str = "corrected_probe_risk",
    prob_weight: float = CORRECTED_PROBE_PROB_WEIGHT,
    logit_weight: float = CORRECTED_PROBE_LOGIT_WEIGHT,
) -> pd.DataFrame:
    """
    Add corrected probe risk.

    This intentionally excludes margin risk because the PoC found it unreliable
    in CIL evaluation.
    """
    out = df.copy()

    out[f"{prob_col}_norm"] = minmax_normalize(out[prob_col])
    out[f"{logit_col}_norm"] = minmax_normalize(out[logit_col])

    out[output_col] = (
        prob_weight * out[f"{prob_col}_norm"]
        + logit_weight * out[f"{logit_col}_norm"]
    )

    return out


def add_old_combined_risk(
    df: pd.DataFrame,
    output_col: str = "combined_risk",
) -> pd.DataFrame:
    """
    Add the original margin-heavy combined risk.

    Kept for ablation and comparison only.
    """
    out = df.copy()

    out["risk_margin_norm"] = minmax_normalize(out["risk_margin"])
    out["risk_probe_prob_drift_norm"] = minmax_normalize(
        out["risk_probe_prob_drift"]
    )

    if "risk_gradient_conflict" in out.columns:
        out["risk_gradient_conflict_norm"] = minmax_normalize(
            out["risk_gradient_conflict"]
        ).fillna(0.0)
    else:
        out["risk_gradient_conflict_norm"] = 0.0

    out[output_col] = (
        0.4 * out["risk_margin_norm"]
        + 0.4 * out["risk_probe_prob_drift_norm"]
        + 0.2 * out["risk_gradient_conflict_norm"]
    )

    return out


def correlation_report(
    df: pd.DataFrame,
    target_col: str,
    score_cols: list[str],
) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations against a target column.
    """
    rows = []

    for score_col in score_cols:
        if score_col not in df.columns:
            continue

        temp = df[[target_col, score_col]].dropna()

        if len(temp) < 5:
            continue

        pearson = pearsonr(temp[score_col], temp[target_col])
        spearman = spearmanr(temp[score_col], temp[target_col])

        rows.append(
            {
                "score": score_col,
                "n": len(temp),
                "pearson_r": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman_r": spearman.statistic,
                "spearman_p": spearman.pvalue,
            }
        )

    return pd.DataFrame(rows).sort_values("spearman_r", ascending=False)


def topk_overlap(
    df: pd.DataFrame,
    risk_col: str,
    target_col: str = "loss_increase",
    k: int = 100,
) -> dict:
    """
    Compute overlap between top-k predicted-risk samples
    and top-k actually forgotten samples.
    """
    temp = df[["index", risk_col, target_col]].dropna()

    top_risk = set(
        temp.sort_values(risk_col, ascending=False).head(k)["index"].tolist()
    )

    top_forgotten = set(
        temp.sort_values(target_col, ascending=False).head(k)["index"].tolist()
    )

    overlap = len(top_risk.intersection(top_forgotten))

    return {
        "risk_col": risk_col,
        "k": k,
        "overlap": overlap,
        "precision_at_k": overlap / k,
    }
