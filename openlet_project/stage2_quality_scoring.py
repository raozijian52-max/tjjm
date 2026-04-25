import numpy as np
import pandas as pd

from stage2_quality_config import DIMENSION_MAP, INDICATOR_DIRECTIONS


def minmax_normalize_indicators(indicator_df):
    """把 18 个原始指标统一 Min-Max 归一化到 0 到 1。"""
    norm_df = indicator_df[["trajectory_id", "scene_id", "task_name"]].copy()
    details = {}

    for col, direction in INDICATOR_DIRECTIONS.items():
        values = pd.to_numeric(indicator_df[col], errors="coerce").astype(float)
        col_min = float(np.nanmin(values)) if np.any(np.isfinite(values)) else np.nan
        col_max = float(np.nanmax(values)) if np.any(np.isfinite(values)) else np.nan

        if not np.isfinite(col_min) or not np.isfinite(col_max):
            norm_values = pd.Series(0.5, index=indicator_df.index, dtype=float)
        elif abs(col_max - col_min) <= 1e-12:
            norm_values = pd.Series(1.0, index=indicator_df.index, dtype=float)
        else:
            norm_values = (values - col_min) / (col_max - col_min)
            if direction < 0:
                norm_values = 1.0 - norm_values
            norm_values = norm_values.fillna(norm_values.mean()).fillna(0.5)

        norm_df[col] = norm_values.clip(0.0, 1.0)
        details[col] = {
            "min": col_min,
            "max": col_max,
            "direction": direction,
            "missing_ratio": float(values.isna().mean()),
        }

    return norm_df, details

def compute_entropy_weights(norm_df):
    """用熵权法根据指标区分度自动计算 18 个指标的权重。"""
    cols = list(INDICATOR_DIRECTIONS.keys())
    X = norm_df[cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    if X.shape[0] == 0:
        return pd.DataFrame(columns=["indicator", "weight", "entropy", "divergence"])

    column_sums = X.sum(axis=0)
    P = np.divide(
        X,
        column_sums.reshape(1, -1),
        out=np.full_like(X, 1.0 / max(X.shape[0], 1)),
        where=column_sums.reshape(1, -1) > 1e-12,
    )

    k = 1.0 / np.log(max(X.shape[0], 2))
    entropy = -k * np.sum(P * np.log(P + 1e-12), axis=0)
    divergence = 1.0 - entropy

    if np.sum(divergence) <= 1e-12:
        weights = np.full(len(cols), 1.0 / len(cols))
    else:
        weights = divergence / np.sum(divergence)

    return pd.DataFrame({
        "indicator": cols,
        "weight": weights,
        "entropy": entropy,
        "divergence": divergence,
    })

def compute_quality_scores(norm_df, weight_df):
    """根据熵权合成综合 Q_score，并计算五个维度分量分数。"""
    cols = list(INDICATOR_DIRECTIONS.keys())
    weight_map = weight_df.set_index("indicator")["weight"].to_dict()

    score_df = norm_df[["trajectory_id", "scene_id", "task_name"]].copy()
    score_df["Q_score"] = 0.0

    for col in cols:
        score_df["Q_score"] += norm_df[col] * weight_map.get(col, 0.0)

    for dimension, prefixes in DIMENSION_MAP.items():
        dim_cols = [
            col for col in cols
            if any(col.startswith(prefix + "_") for prefix in prefixes)
        ]
        dim_weight_sum = sum(weight_map.get(col, 0.0) for col in dim_cols)
        if dim_weight_sum <= 1e-12:
            score_df[f"Q_{dimension}"] = norm_df[dim_cols].mean(axis=1)
        else:
            score_df[f"Q_{dimension}"] = sum(
                norm_df[col] * weight_map.get(col, 0.0) for col in dim_cols
            ) / dim_weight_sum

    return score_df

def compute_pca_robustness(norm_df, score_df):
    """用 PCA 第一主成分和 Q_score 做相关性检查，作为稳健性参考。"""
    cols = list(INDICATOR_DIRECTIONS.keys())
    X = norm_df[cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.5, posinf=1.0, neginf=0.0)

    if len(X) < 2:
        return {
            "status": "not_enough_samples",
            "spearman_corr_with_entropy_q": None,
            "explained_variance_ratio": None,
        }

    centered = X - np.mean(X, axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = centered @ vt[0]
    pc1 = (pc1 - np.min(pc1)) / (np.max(pc1) - np.min(pc1) + 1e-12)

    corr = pd.Series(pc1).corr(score_df["Q_score"], method="spearman")
    total_variance = float(np.sum(singular_values ** 2))
    explained = float((singular_values[0] ** 2) / total_variance) if total_variance > 1e-12 else 0.0
    return {
        "status": "ok",
        "spearman_corr_with_entropy_q": float(corr) if pd.notna(corr) else None,
        "explained_variance_ratio": explained,
    }
