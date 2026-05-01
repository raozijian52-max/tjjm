import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CONFIG
from utils import save_csv


REQUIRED_INPUT_COLS = [
    "global_id",
    "trajectory_id",
    "scene_id",
    "trajectory_normalized_mse",
    "trajectory_imitation_score",
    "Q_score",
    "delta_score_emp",
]

OUTCOME_COLS = ["outcome_mse", "outcome_score"]


# 读取阶段四主表
# 输入：无
# 输出：master_df

def load_stage5_input():
    path = os.path.join(CONFIG["interim_dir"], "stage4_modeling_master_table.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到阶段五输入文件：{path}")

    master_df = pd.read_csv(path)

    missing_cols = [c for c in REQUIRED_INPUT_COLS if c not in master_df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"阶段五输入缺少字段：{missing_cols}")

    return master_df


# 构建阶段五统一分析数据
# 输入：master_df
# 输出：causal_df, treatment_info_df

def build_causal_dataset(master_df: pd.DataFrame):
    df = master_df.copy()

    df["outcome_mse"] = df["trajectory_normalized_mse"]
    df["outcome_score"] = df["trajectory_imitation_score"]

    q_threshold = float(df["Q_score"].median())
    df["treat_high_quality"] = (df["Q_score"] >= q_threshold).astype(int)

    scene_delta = (
        df[["scene_id", "delta_score_emp"]]
        .drop_duplicates(subset=["scene_id"])
        .reset_index(drop=True)
    )
    value_threshold = float(scene_delta["delta_score_emp"].median())
    high_value_scene_ids = set(
        scene_delta.loc[scene_delta["delta_score_emp"] >= value_threshold, "scene_id"].astype(str)
    )
    df["treat_high_value"] = df["scene_id"].astype(str).isin(high_value_scene_ids).astype(int)

    dim_cols = [
        "Q_completeness",
        "Q_accuracy",
        "Q_diversity",
        "Q_consistency",
    ]
    dim_map = {
        "Q_completeness": "treat_high_completeness",
        "Q_accuracy": "treat_high_accuracy",
        "Q_diversity": "treat_high_diversity",
        "Q_consistency": "treat_high_consistency",
    }
    for q_col in dim_cols:
        if q_col in df.columns:
            thr = float(df[q_col].median())
            df[dim_map[q_col]] = (df[q_col] >= thr).astype(int)

    stage1_cols = [
        c for c in df.columns
        if c.startswith("stage1_") and pd.api.types.is_numeric_dtype(df[c])
    ]

    keep_cols = [
        "global_id",
        "trajectory_id",
        "scene_id",
        "outcome_mse",
        "outcome_score",
        "treat_high_quality",
        "treat_high_value",
        "Q_score",
        "delta_score_emp",
    ] + stage1_cols

    for t_col in dim_map.values():
        if t_col in df.columns:
            keep_cols.append(t_col)

    causal_df = df[keep_cols].copy()

    treatment_rows = []
    treatments = [
        ("treat_high_quality", q_threshold, "trajectory_median_Q_score"),
        ("treat_high_value", value_threshold, "scene_median_delta_score_emp"),
    ]
    for q_col, t_col in [("Q_completeness", "treat_high_completeness"), ("Q_accuracy", "treat_high_accuracy"),
                         ("Q_diversity", "treat_high_diversity"), ("Q_consistency", "treat_high_consistency")]:
        if t_col in causal_df.columns:
            treatments.append((t_col, float(df[q_col].median()), f"trajectory_median_{q_col}"))

    for t_col, thr, thr_type in treatments:
        n_total = int(len(causal_df))
        n_treated = int(causal_df[t_col].sum())
        n_control = int(n_total - n_treated)
        treated_rate = float(n_treated / n_total) if n_total > 0 else np.nan
        treatment_rows.append(
            {
                "treatment": t_col,
                "n_total": n_total,
                "n_treated": n_treated,
                "n_control": n_control,
                "treated_rate": treated_rate,
                "threshold": thr,
                "threshold_type": thr_type,
            }
        )

    treatment_info_df = pd.DataFrame(treatment_rows)

    return causal_df, treatment_info_df


# 获取控制变量列
# 输入：causal_df, analysis_variant
# 输出：confounder_cols, model_df

def get_confounder_cols(causal_df: pd.DataFrame, analysis_variant: str):
    if analysis_variant not in {"no_scene", "with_scene"}:
        raise ValueError(f"analysis_variant 不合法：{analysis_variant}")

    model_df = causal_df.copy()
    stage1_cols = [
        c for c in model_df.columns
        if c.startswith("stage1_") and pd.api.types.is_numeric_dtype(model_df[c])
    ]

    confounder_cols = list(stage1_cols)

    if analysis_variant == "with_scene":
        scene_dummies = pd.get_dummies(
            model_df["scene_id"].astype(str),
            prefix="scene",
            dtype=float,
        )
        model_df = pd.concat([model_df, scene_dummies], axis=1)
        confounder_cols += list(scene_dummies.columns)

    return confounder_cols, model_df


# 拟合倾向得分模型
# 输入：df, treatment_col, confounder_cols
# 输出：propensity, propensity_model

def fit_propensity_model(df: pd.DataFrame, treatment_col: str, confounder_cols: List[str]):
    treat = df[treatment_col].astype(int).values
    if treat.min() == treat.max():
        raise ValueError(f"{treatment_col} 全为同一类，无法拟合倾向得分模型。")

    x = df[confounder_cols]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(x, treat)

    propensity = model.predict_proba(x)[:, 1]
    propensity = np.clip(propensity, 0.01, 0.99)

    return propensity, model


def _weighted_mean(values: np.ndarray, weights: np.ndarray):
    denom = np.sum(weights)
    if denom <= 0:
        return np.nan
    return float(np.sum(values * weights) / denom)


def _weighted_var(values: np.ndarray, weights: np.ndarray):
    mu = _weighted_mean(values, weights)
    if np.isnan(mu):
        return np.nan
    denom = np.sum(weights)
    if denom <= 0:
        return np.nan
    return float(np.sum(weights * (values - mu) ** 2) / denom)


def _direction_and_interpretation(outcome_col: str, estimate: float):
    if outcome_col == "outcome_mse":
        if estimate < 0:
            return "beneficial", "estimate < 0，表示 treatment 降低 BC 模仿误差（有益）"
        if estimate > 0:
            return "harmful", "estimate > 0，表示 treatment 提高 BC 模仿误差（不利）"
        return "neutral", "estimate = 0，表示 treatment 对 BC 模仿误差无差异"

    if outcome_col == "outcome_score":
        if estimate > 0:
            return "beneficial", "estimate > 0，表示 treatment 提高 imitation_score（有益）"
        if estimate < 0:
            return "harmful", "estimate < 0，表示 treatment 降低 imitation_score（不利）"
        return "neutral", "estimate = 0，表示 treatment 对 imitation_score 无差异"

    return "unknown", "outcome 未定义解释规则"


# 计算未调整均值差
# 输入：df, treatment_col, outcome_col
# 输出：effect_row dict

def estimate_raw_difference(df: pd.DataFrame, treatment_col: str, outcome_col: str):
    treated = df[df[treatment_col] == 1][outcome_col].values
    control = df[df[treatment_col] == 0][outcome_col].values

    estimate = float(np.mean(treated) - np.mean(control))
    direction, interpretation = _direction_and_interpretation(outcome_col, estimate)

    return {
        "method": "raw_difference",
        "outcome": outcome_col,
        "effect_type": "difference_in_means",
        "estimate": estimate,
        "n_treated": int(len(treated)),
        "n_control": int(len(control)),
        "n_used": int(len(treated) + len(control)),
        "direction": direction,
        "interpretation": interpretation,
    }


# 最近邻匹配 ATT
# 输入：df, treatment_col, outcome_cols, propensity
# 输出：effect_rows, matched_pairs_df

def run_psm_att(df: pd.DataFrame, treatment_col: str, outcome_cols: List[str], propensity: np.ndarray):
    work_df = df.copy().reset_index(drop=True)
    work_df["_propensity"] = propensity

    treated_df = work_df[work_df[treatment_col] == 1].copy()
    control_df = work_df[work_df[treatment_col] == 0].copy()

    if len(treated_df) == 0 or len(control_df) == 0:
        return [], pd.DataFrame()

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control_df[["_propensity"]].values)

    distances, indices = nn.kneighbors(treated_df[["_propensity"]].values)
    matched_control = control_df.iloc[indices.flatten()].copy().reset_index(drop=True)
    matched_treated = treated_df.reset_index(drop=True)

    pair_df = pd.DataFrame(
        {
            "treated_global_id": matched_treated["global_id"].values,
            "control_global_id": matched_control["global_id"].values,
            "treated_propensity": matched_treated["_propensity"].values,
            "control_propensity": matched_control["_propensity"].values,
            "propensity_distance": distances.flatten(),
            "treated_outcome_mse": matched_treated["outcome_mse"].values,
            "control_outcome_mse": matched_control["outcome_mse"].values,
            "treated_outcome_score": matched_treated["outcome_score"].values,
            "control_outcome_score": matched_control["outcome_score"].values,
        }
    )

    effect_rows = []
    for outcome_col in outcome_cols:
        est = float(
            np.mean(
                matched_treated[outcome_col].values - matched_control[outcome_col].values
            )
        )
        direction, interpretation = _direction_and_interpretation(outcome_col, est)
        effect_rows.append(
            {
                "method": "psm_att",
                "outcome": outcome_col,
                "effect_type": "ATT",
                "estimate": est,
                "n_treated": int(len(matched_treated)),
                "n_control": int(len(control_df)),
                "n_used": int(len(matched_treated)),
                "direction": direction,
                "interpretation": interpretation,
            }
        )

    return effect_rows, pair_df


# IPW ATE
# 输入：df, treatment_col, outcome_cols, propensity
# 输出：effect_rows, weights_df

def run_ipw_ate(df: pd.DataFrame, treatment_col: str, outcome_cols: List[str], propensity: np.ndarray):
    work_df = df.copy().reset_index(drop=True)
    treat = work_df[treatment_col].astype(int).values
    p = propensity

    ipw = treat / p + (1 - treat) / (1 - p)

    p_t = float(np.mean(treat))
    p_c = 1.0 - p_t
    sw = treat * p_t / p + (1 - treat) * p_c / (1 - p)

    weights_df = pd.DataFrame(
        {
            "global_id": work_df["global_id"].values,
            "scene_id": work_df["scene_id"].values,
            "treat": treat,
            "propensity": p,
            "ipw_weight": ipw,
            "stabilized_ipw_weight": sw,
        }
    )

    effect_rows = []
    for outcome_col in outcome_cols:
        y = work_df[outcome_col].values

        y_t = y[treat == 1]
        y_c = y[treat == 0]
        w_t = sw[treat == 1]
        w_c = sw[treat == 0]

        mu_t = _weighted_mean(y_t, w_t)
        mu_c = _weighted_mean(y_c, w_c)
        est = float(mu_t - mu_c)

        direction, interpretation = _direction_and_interpretation(outcome_col, est)
        effect_rows.append(
            {
                "method": "ipw_ate",
                "outcome": outcome_col,
                "effect_type": "ATE",
                "estimate": est,
                "n_treated": int(np.sum(treat == 1)),
                "n_control": int(np.sum(treat == 0)),
                "n_used": int(len(work_df)),
                "direction": direction,
                "interpretation": interpretation,
            }
        )

    return effect_rows, weights_df


def _smd_unweighted(df: pd.DataFrame, treatment_col: str, covariate: str):
    t = df[df[treatment_col] == 1][covariate].dropna().values
    c = df[df[treatment_col] == 0][covariate].dropna().values

    if len(t) < 2 or len(c) < 2:
        return np.nan

    mt, mc = np.mean(t), np.mean(c)
    vt = np.var(t, ddof=1)
    vc = np.var(c, ddof=1)
    pooled_sd = np.sqrt((vt + vc) / 2.0)

    if pooled_sd <= 0 or np.isnan(pooled_sd):
        return 0.0

    return float((mt - mc) / pooled_sd)


def _smd_weighted(df: pd.DataFrame, treatment_col: str, covariate: str, weight_col: str):
    t_df = df[df[treatment_col] == 1][[covariate, weight_col]].dropna()
    c_df = df[df[treatment_col] == 0][[covariate, weight_col]].dropna()

    if len(t_df) < 2 or len(c_df) < 2:
        return np.nan

    mt = _weighted_mean(t_df[covariate].values, t_df[weight_col].values)
    mc = _weighted_mean(c_df[covariate].values, c_df[weight_col].values)

    vt = _weighted_var(t_df[covariate].values, t_df[weight_col].values)
    vc = _weighted_var(c_df[covariate].values, c_df[weight_col].values)
    pooled_sd = np.sqrt((vt + vc) / 2.0)

    if pooled_sd <= 0 or np.isnan(pooled_sd):
        return 0.0

    return float((mt - mc) / pooled_sd)


# 协变量平衡 SMD
# 输入：df, treatment_col, confounder_cols, method, propensity, matched_pairs_df
# 输出：balance_df

def compute_smd_table(
    df: pd.DataFrame,
    treatment_col: str,
    confounder_cols: List[str],
    method: str,
    propensity: np.ndarray = None,
    matched_pairs_df: pd.DataFrame = None,
):
    if method not in {"psm_att", "ipw_ate"}:
        raise ValueError(f"不支持的 method：{method}")

    rows = []

    for cov in confounder_cols:
        smd_before = _smd_unweighted(df, treatment_col, cov)

        if method == "psm_att":
            if matched_pairs_df is None or len(matched_pairs_df) == 0:
                smd_after = np.nan
            else:
                treated_ids = set(matched_pairs_df["treated_global_id"].astype(str))
                control_ids = list(matched_pairs_df["control_global_id"].astype(str))

                treated_df = df[df["global_id"].astype(str).isin(treated_ids)].copy()
                control_df = df[df["global_id"].astype(str).isin(control_ids)].copy()
                control_df = control_df.set_index("global_id").loc[control_ids].reset_index()

                after_df = pd.concat([treated_df, control_df], axis=0, ignore_index=True)
                after_df[treatment_col] = [1] * len(treated_df) + [0] * len(control_df)
                smd_after = _smd_unweighted(after_df, treatment_col, cov)

        else:
            if propensity is None:
                smd_after = np.nan
            else:
                t = df[treatment_col].astype(int).values
                p = np.clip(propensity, 0.01, 0.99)
                p_t = float(np.mean(t))
                p_c = 1.0 - p_t
                sw = t * p_t / p + (1 - t) * p_c / (1 - p)

                w_df = df[[treatment_col, cov]].copy()
                w_df["_sw"] = sw
                smd_after = _smd_weighted(w_df, treatment_col, cov, "_sw")

        abs_before = np.abs(smd_before) if pd.notna(smd_before) else np.nan
        abs_after = np.abs(smd_after) if pd.notna(smd_after) else np.nan
        improved = bool(abs_after < abs_before) if pd.notna(abs_before) and pd.notna(abs_after) else False

        rows.append(
            {
                "covariate": cov,
                "smd_before": smd_before,
                "smd_after": smd_after,
                "abs_smd_before": abs_before,
                "abs_smd_after": abs_after,
                "balance_improved": improved,
            }
        )

    return pd.DataFrame(rows)


# 单个 treatment + variant 完整分析
# 输入：causal_df, treatment_col, analysis_variant
# 输出：effect_df, matched_pairs_df, weights_df, balance_df, overlap_df

def run_one_treatment_analysis(causal_df: pd.DataFrame, treatment_col: str, analysis_variant: str):
    if treatment_col == "treat_high_value" and analysis_variant == "with_scene":
        raise ValueError("treat_high_value 不允许 with_scene 分析。")

    confounder_cols, model_df = get_confounder_cols(causal_df, analysis_variant)

    work_cols = [
        "global_id", "trajectory_id", "scene_id", treatment_col
    ] + OUTCOME_COLS + confounder_cols
    df = model_df[work_cols].copy()

    propensity, _ = fit_propensity_model(df, treatment_col, confounder_cols)
    df["_propensity"] = propensity

    treated_ps = df.loc[df[treatment_col] == 1, "_propensity"]
    control_ps = df.loc[df[treatment_col] == 0, "_propensity"]

    overlap_row = {
        "treatment": treatment_col,
        "analysis_variant": analysis_variant,
        "n_total": int(len(df)),
        "treated_ps_min": float(treated_ps.min()),
        "treated_ps_p25": float(treated_ps.quantile(0.25)),
        "treated_ps_median": float(treated_ps.median()),
        "treated_ps_p75": float(treated_ps.quantile(0.75)),
        "treated_ps_max": float(treated_ps.max()),
        "control_ps_min": float(control_ps.min()),
        "control_ps_p25": float(control_ps.quantile(0.25)),
        "control_ps_median": float(control_ps.median()),
        "control_ps_p75": float(control_ps.quantile(0.75)),
        "control_ps_max": float(control_ps.max()),
    }
    overlap_df = pd.DataFrame([overlap_row])

    effect_rows = []
    for outcome_col in OUTCOME_COLS:
        effect_rows.append(estimate_raw_difference(df, treatment_col, outcome_col))

    psm_effect_rows, pair_df = run_psm_att(df, treatment_col, OUTCOME_COLS, propensity)
    effect_rows.extend(psm_effect_rows)

    ipw_effect_rows, w_df = run_ipw_ate(df, treatment_col, OUTCOME_COLS, propensity)
    effect_rows.extend(ipw_effect_rows)

    effect_df = pd.DataFrame(effect_rows)
    effect_df.insert(0, "analysis_variant", analysis_variant)
    effect_df.insert(0, "treatment", treatment_col)

    if len(pair_df) > 0:
        pair_df.insert(0, "analysis_variant", analysis_variant)
        pair_df.insert(0, "treatment", treatment_col)

    if len(w_df) > 0:
        w_df.insert(0, "analysis_variant", analysis_variant)
        w_df.insert(0, "treatment", treatment_col)

    psm_balance = compute_smd_table(
        df=df,
        treatment_col=treatment_col,
        confounder_cols=confounder_cols,
        method="psm_att",
        matched_pairs_df=pair_df,
    )
    psm_balance.insert(0, "method", "psm_att")

    ipw_balance = compute_smd_table(
        df=df,
        treatment_col=treatment_col,
        confounder_cols=confounder_cols,
        method="ipw_ate",
        propensity=propensity,
    )
    ipw_balance.insert(0, "method", "ipw_ate")

    balance_df = pd.concat([psm_balance, ipw_balance], axis=0, ignore_index=True)
    balance_df.insert(0, "analysis_variant", analysis_variant)
    balance_df.insert(0, "treatment", treatment_col)

    return effect_df, pair_df, w_df, balance_df, overlap_df


# 阶段五总入口
# 输入：无
# 输出：causal_df, effect_df, balance_df, matched_pairs_df, weights_df, overlap_df

def run_stage5_causal_analysis():
    master_df = load_stage5_input()
    causal_df, treatment_info_df = build_causal_dataset(master_df)

    analyses = [
        ("treat_high_quality", "no_scene"),
        ("treat_high_quality", "with_scene"),
        ("treat_high_value", "no_scene"),
        ("treat_high_completeness", "no_scene"),
        ("treat_high_accuracy", "no_scene"),
        ("treat_high_diversity", "no_scene"),
        ("treat_high_consistency", "no_scene"),
    ]

    effects = []
    pairs = []
    weights = []
    balances = []
    overlaps = []

    for treatment_col, analysis_variant in analyses:
        if treatment_col not in causal_df.columns:
            continue

        try:
            e_df, p_df, w_df, b_df, o_df = run_one_treatment_analysis(
                causal_df=causal_df,
                treatment_col=treatment_col,
                analysis_variant=analysis_variant,
            )
            effects.append(e_df)
            if len(p_df) > 0:
                pairs.append(p_df)
            if len(w_df) > 0:
                weights.append(w_df)
            balances.append(b_df)
            overlaps.append(o_df)
        except ValueError as ex:
            print(f"[Stage5][Skip] {treatment_col}-{analysis_variant}: {ex}")

    effect_df = pd.concat(effects, axis=0, ignore_index=True) if len(effects) > 0 else pd.DataFrame()
    matched_pairs_df = pd.concat(pairs, axis=0, ignore_index=True) if len(pairs) > 0 else pd.DataFrame()
    weights_df = pd.concat(weights, axis=0, ignore_index=True) if len(weights) > 0 else pd.DataFrame()
    balance_df = pd.concat(balances, axis=0, ignore_index=True) if len(balances) > 0 else pd.DataFrame()
    overlap_df = pd.concat(overlaps, axis=0, ignore_index=True) if len(overlaps) > 0 else pd.DataFrame()

    save_csv(causal_df, os.path.join(CONFIG["interim_dir"], "stage5_causal_dataset.csv"))
    save_csv(treatment_info_df, os.path.join(CONFIG["interim_dir"], "stage5_treatment_summary.csv"))
    save_csv(overlap_df, os.path.join(CONFIG["interim_dir"], "stage5_propensity_overlap.csv"))
    save_csv(balance_df, os.path.join(CONFIG["interim_dir"], "stage5_balance_summary.csv"))
    save_csv(effect_df, os.path.join(CONFIG["interim_dir"], "stage5_effect_estimates.csv"))
    save_csv(matched_pairs_df, os.path.join(CONFIG["interim_dir"], "stage5_matched_pairs.csv"))
    save_csv(weights_df, os.path.join(CONFIG["interim_dir"], "stage5_ipw_weights.csv"))

    return causal_df, effect_df, balance_df, matched_pairs_df, weights_df, overlap_df, treatment_info_df
