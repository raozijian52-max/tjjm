import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CONFIG
from utils import save_csv


REQUIRED_SCORE_COLS = [
    "global_id",
    "trajectory_id",
    "scene_id",
    "Q_score",
    "Q_accuracy",
    "Q_consistency",
]

OUTCOME_COLS = ["outcome_mse", "outcome_score"]


def _safe_ratio_to_int_ratio(ratio_value: float) -> int:
    return int(round(float(ratio_value) * 100))


def _load_stage4_scores():
    path = os.path.join(CONFIG["interim_dir"], "stage4_curation_scores.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到阶段五输入文件：{path}")

    df = pd.read_csv(path)
    miss = [c for c in REQUIRED_SCORE_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"stage4_curation_scores 缺少字段：{miss}")

    return df


def _load_stage4_selection():
    path = os.path.join(CONFIG["interim_dir"], "stage4_curation_selection_table.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到阶段五输入文件：{path}")

    df = pd.read_csv(path)
    need = ["strategy", "ratio", "run_seed", "global_id", "scene_id"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"stage4_curation_selection_table 缺少字段：{miss}")

    return df


def _load_stage4_metrics():
    path = os.path.join(CONFIG["interim_dir"], "stage4_curation_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到阶段五输入文件：{path}")

    df = pd.read_csv(path)
    need = ["strategy", "ratio", "run_seed", "test_nmse", "test_imitation_score"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"stage4_curation_metrics 缺少字段：{miss}")

    return df


def load_stage5_input():
    scores_df = _load_stage4_scores()
    selection_df = _load_stage4_selection()
    metrics_df = _load_stage4_metrics()
    return scores_df, selection_df, metrics_df


def build_causal_dataset(scores_df: pd.DataFrame, selection_df: pd.DataFrame, metrics_df: pd.DataFrame):
    # pool 轨迹为候选总体；test 不参与 treatment 定义
    base_df = scores_df[scores_df["split"] == "pool"].copy()

    # 用 full@ratio=1.0 的 run_seed 对应测试集效能作为轨迹级 outcome（同 run_seed 下常量）
    full_metrics = metrics_df[metrics_df["strategy"] == "full"].copy()
    if full_metrics.empty:
        raise ValueError("stage4_curation_metrics 中不存在 strategy=full，无法构建 outcome")

    full_metrics["ratio_int"] = full_metrics["ratio"].apply(_safe_ratio_to_int_ratio)
    full_metrics = full_metrics[full_metrics["ratio_int"] == 100].copy()
    if full_metrics.empty:
        raise ValueError("stage4_curation_metrics 中不存在 full 且 ratio=1.0 的记录")

    full_metrics = full_metrics[["run_seed", "test_nmse", "test_imitation_score"]].drop_duplicates()

    # 处理变量：双轨策展精选 vs 被剔除（按 hybrid_Q_delta，默认 ratio=0.5）
    sel = selection_df.copy()
    sel["ratio_int"] = sel["ratio"].apply(_safe_ratio_to_int_ratio)
    target_ratio = 50
    if not (sel["strategy"] == "hybrid_Q_delta").any():
        raise ValueError("stage4_curation_selection_table 中不存在 strategy=hybrid_Q_delta")

    # 若无 0.5，则退化到最接近 0.5 的比例
    candidate_ratios = sorted(sel.loc[sel["strategy"] == "hybrid_Q_delta", "ratio_int"].dropna().unique().tolist())
    if len(candidate_ratios) == 0:
        raise ValueError("hybrid_Q_delta 下无可用 ratio")
    if target_ratio not in candidate_ratios:
        target_ratio = min(candidate_ratios, key=lambda x: abs(x - 50))

    sel = sel[(sel["strategy"] == "hybrid_Q_delta") & (sel["ratio_int"] == target_ratio)].copy()

    # 每个 run_seed 下，构造轨迹层样本：是否被选中 + 该 run_seed 下 BC 效能
    rows = []
    for run_seed in sorted(sel["run_seed"].dropna().unique().tolist()):
        sel_seed = sel[sel["run_seed"] == run_seed]
        selected_ids = set(sel_seed["global_id"].astype(str).tolist())

        met = full_metrics[full_metrics["run_seed"] == run_seed]
        if met.empty:
            continue
        outcome_mse = float(met["test_nmse"].iloc[0])
        outcome_score = float(met["test_imitation_score"].iloc[0])

        seed_df = base_df.copy()
        seed_df["run_seed"] = int(run_seed)
        seed_df["treat_curation_selected"] = seed_df["global_id"].astype(str).isin(selected_ids).astype(int)
        seed_df["outcome_mse"] = outcome_mse
        seed_df["outcome_score"] = outcome_score
        rows.append(seed_df)

    if len(rows) == 0:
        raise ValueError("无法对齐 selection 与 metrics 的 run_seed，无法构建阶段五数据")

    df = pd.concat(rows, axis=0, ignore_index=True)

    # 兼容旧定义：Q_score 中位数分组，作为稳健性/补充
    q_threshold = float(df["Q_score"].median())
    df["treat_high_quality"] = (df["Q_score"] >= q_threshold).astype(int)

    # 协变量：场景 + 轨迹长度 + 物体类别 + 质量维度（若存在）
    if "object_id" not in df.columns:
        df["object_id"] = "unknown"

    if "trajectory_length" not in df.columns:
        if "selected_rank" in df.columns and pd.api.types.is_numeric_dtype(df["selected_rank"]):
            df["trajectory_length"] = df["selected_rank"].astype(float)
        else:
            df["trajectory_length"] = np.nan

    keep_cols = [
        "global_id", "trajectory_id", "scene_id", "run_seed",
        "outcome_mse", "outcome_score",
        "treat_curation_selected", "treat_high_quality",
        "Q_score", "Q_accuracy", "Q_consistency",
        "trajectory_length", "object_id",
    ]

    for c in ["Q_completeness", "Q_diversity"]:
        if c in df.columns:
            keep_cols.append(c)

    causal_df = df[keep_cols].copy()

    treatment_info_df = pd.DataFrame(
        [
            {
                "treatment": "treat_curation_selected",
                "n_total": int(len(causal_df)),
                "n_treated": int(causal_df["treat_curation_selected"].sum()),
                "n_control": int(len(causal_df) - causal_df["treat_curation_selected"].sum()),
                "treated_rate": float(causal_df["treat_curation_selected"].mean()),
                "threshold": target_ratio / 100.0,
                "threshold_type": "curation_ratio",
                "source_strategy": "hybrid_Q_delta",
            },
            {
                "treatment": "treat_high_quality",
                "n_total": int(len(causal_df)),
                "n_treated": int(causal_df["treat_high_quality"].sum()),
                "n_control": int(len(causal_df) - causal_df["treat_high_quality"].sum()),
                "treated_rate": float(causal_df["treat_high_quality"].mean()),
                "threshold": q_threshold,
                "threshold_type": "trajectory_median_Q_score",
                "source_strategy": "stage2_quality",
            },
        ]
    )

    return causal_df, treatment_info_df


def get_confounder_cols(causal_df: pd.DataFrame, analysis_variant: str):
    if analysis_variant not in {"no_scene", "with_scene"}:
        raise ValueError(f"analysis_variant 不合法：{analysis_variant}")

    model_df = causal_df.copy()
    confounder_cols = []

    numeric_covs = ["trajectory_length", "Q_score", "Q_accuracy", "Q_consistency"]
    for c in numeric_covs:
        if c in model_df.columns and pd.api.types.is_numeric_dtype(model_df[c]):
            confounder_cols.append(c)

    if "object_id" in model_df.columns:
        obj_dummies = pd.get_dummies(model_df["object_id"].astype(str), prefix="obj", dtype=float)
        model_df = pd.concat([model_df, obj_dummies], axis=1)
        confounder_cols += list(obj_dummies.columns)

    # run_seed 固定效应，避免不同 seed 的系统差异
    if "run_seed" in model_df.columns:
        seed_dummies = pd.get_dummies(model_df["run_seed"].astype(str), prefix="seed", dtype=float)
        model_df = pd.concat([model_df, seed_dummies], axis=1)
        confounder_cols += list(seed_dummies.columns)

    if analysis_variant == "with_scene":
        scene_dummies = pd.get_dummies(model_df["scene_id"].astype(str), prefix="scene", dtype=float)
        model_df = pd.concat([model_df, scene_dummies], axis=1)
        confounder_cols += list(scene_dummies.columns)

    return confounder_cols, model_df


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
        est = float(np.mean(matched_treated[outcome_col].values - matched_control[outcome_col].values))
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
        y_t, y_c = y[treat == 1], y[treat == 0]
        w_t, w_c = sw[treat == 1], sw[treat == 0]
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
    vt, vc = np.var(t, ddof=1), np.var(c, ddof=1)
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


def compute_smd_table(df: pd.DataFrame, treatment_col: str, confounder_cols: List[str], method: str, propensity: np.ndarray = None, matched_pairs_df: pd.DataFrame = None):
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
        rows.append({
            "covariate": cov,
            "smd_before": smd_before,
            "smd_after": smd_after,
            "abs_smd_before": abs_before,
            "abs_smd_after": abs_after,
            "balance_improved": improved,
        })

    return pd.DataFrame(rows)


def run_one_treatment_analysis(causal_df: pd.DataFrame, treatment_col: str, analysis_variant: str):
    confounder_cols, model_df = get_confounder_cols(causal_df, analysis_variant)

    work_cols = ["global_id", "trajectory_id", "scene_id", treatment_col] + OUTCOME_COLS + confounder_cols
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

    effect_rows = [estimate_raw_difference(df, treatment_col, oc) for oc in OUTCOME_COLS]
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

    psm_balance = compute_smd_table(df, treatment_col, confounder_cols, method="psm_att", matched_pairs_df=pair_df)
    psm_balance.insert(0, "method", "psm_att")
    ipw_balance = compute_smd_table(df, treatment_col, confounder_cols, method="ipw_ate", propensity=propensity)
    ipw_balance.insert(0, "method", "ipw_ate")

    balance_df = pd.concat([psm_balance, ipw_balance], axis=0, ignore_index=True)
    balance_df.insert(0, "analysis_variant", analysis_variant)
    balance_df.insert(0, "treatment", treatment_col)

    return effect_df, pair_df, w_df, balance_df, overlap_df


def run_stage5_causal_analysis():
    scores_df, selection_df, metrics_df = load_stage5_input()
    causal_df, treatment_info_df = build_causal_dataset(scores_df, selection_df, metrics_df)

    analyses = [
        ("treat_curation_selected", "no_scene"),
        ("treat_curation_selected", "with_scene"),
        ("treat_high_quality", "no_scene"),
        ("treat_high_quality", "with_scene"),
    ]

    effects, pairs, weights, balances, overlaps = [], [], [], [], []

    for treatment_col, analysis_variant in analyses:
        if treatment_col not in causal_df.columns:
            continue
        try:
            e_df, p_df, w_df, b_df, o_df = run_one_treatment_analysis(causal_df, treatment_col, analysis_variant)
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
