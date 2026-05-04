import math

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

def compute_spearman_corr_pvalue(x, y):
    """?? Spearman ????? p ????? scipy??? scipy ???? p ??"""
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    valid_mask = x.notna() & y.notna()
    x = x[valid_mask]
    y = y[valid_mask]
    n = len(x)

    if n < 3:
        return None, None

    try:
        from scipy.stats import spearmanr

        result = spearmanr(x.to_numpy(), y.to_numpy())
        corr = result.statistic
        pvalue = result.pvalue
    except Exception:
        corr = x.rank(method="average").corr(y.rank(method="average"), method="pearson")

        if pd.isna(corr):
            return None, None

        corr_for_t = float(np.clip(corr, -0.999999, 0.999999))
        t_value = corr_for_t * math.sqrt((n - 2) / (1.0 - corr_for_t ** 2))

        # Fallback only: approximate two-sided p-value with normal tail.
        pvalue = math.erfc(abs(t_value) / math.sqrt(2.0))

    if pd.isna(corr) or pd.isna(pvalue):
        return None, None

    return float(corr), float(pvalue)


def compute_pca_robustness(norm_df, score_df, target_cumulative_variance=0.90):
    """??? PCA ???? Q_score ? Spearman ???????????????"""
    cols = list(INDICATOR_DIRECTIONS.keys())
    X = norm_df[cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.5, posinf=1.0, neginf=0.0)

    if len(X) < 2:
        return {
            "status": "not_enough_samples",
            "spearman_corr_with_entropy_q": None,
            "spearman_pvalue_with_entropy_q": None,
            "explained_variance_ratio": None,
            "components": [],
        }

    centered = X - np.mean(X, axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    total_variance = float(np.sum(singular_values ** 2))
    if total_variance <= 1e-12:
        return {
            "status": "zero_variance",
            "spearman_corr_with_entropy_q": None,
            "spearman_pvalue_with_entropy_q": None,
            "explained_variance_ratio": 0.0,
            "components": [],
        }

    explained_ratios = (singular_values ** 2) / total_variance
    cumulative_ratios = np.cumsum(explained_ratios)
    selected_n_components = int(np.searchsorted(cumulative_ratios, target_cumulative_variance) + 1)
    selected_n_components = min(selected_n_components, len(explained_ratios))

    component_rows = []
    component_scores = []

    for i in range(selected_n_components):
        pc_score = centered @ vt[i]
        pc_score = (pc_score - np.min(pc_score)) / (np.max(pc_score) - np.min(pc_score) + 1e-12)

        corr, pvalue = compute_spearman_corr_pvalue(pc_score, score_df["Q_score"])
        sign_flipped = False

        # PCA ?????????????????? Q_score ?????????
        if corr is not None and corr < 0:
            pc_score = 1.0 - pc_score
            corr, pvalue = compute_spearman_corr_pvalue(pc_score, score_df["Q_score"])
            sign_flipped = True

        component_scores.append(pc_score)
        component_rows.append({
            "component": int(i + 1),
            "explained_variance_ratio": float(explained_ratios[i]),
            "cumulative_explained_variance_ratio": float(cumulative_ratios[i]),
            "spearman_corr_with_entropy_q": corr,
            "spearman_pvalue_with_entropy_q": pvalue,
            "sign_flipped_for_positive_corr": sign_flipped,
        })

    selected_weights = explained_ratios[:selected_n_components]
    selected_weights = selected_weights / np.sum(selected_weights)
    combined_score = np.sum(
        np.vstack(component_scores) * selected_weights.reshape(-1, 1),
        axis=0,
    )
    combined_corr, combined_pvalue = compute_spearman_corr_pvalue(
        combined_score,
        score_df["Q_score"],
    )

    return {
        "status": "ok",
        "target_cumulative_variance": float(target_cumulative_variance),
        "selected_n_components": selected_n_components,
        "selected_cumulative_explained_variance_ratio": float(cumulative_ratios[selected_n_components - 1]),
        "spearman_corr_with_entropy_q": component_rows[0]["spearman_corr_with_entropy_q"],
        "spearman_pvalue_with_entropy_q": component_rows[0]["spearman_pvalue_with_entropy_q"],
        "explained_variance_ratio": component_rows[0]["explained_variance_ratio"],
        "combined_selected_components_spearman_corr_with_entropy_q": combined_corr,
        "combined_selected_components_spearman_pvalue_with_entropy_q": combined_pvalue,
        "components": component_rows,
    }
