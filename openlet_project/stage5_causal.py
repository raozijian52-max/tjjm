import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CONFIG
from utils import save_csv


OUTCOME_PRIMARY = "trajectory_imitation_score"  # 越大越好
OUTCOME_SECONDARY = "trajectory_normalized_mse"  # 越小越好

# 反馈建议：处理变量使用质量维度本身，不使用“是否被策展选中”
TREATMENT_SPECS = {
    "T_high_Q_score": ("Q_score", [0.5, 0.6]),
    "T_high_Q_accuracy": ("Q_accuracy", [0.5, 0.6]),
    "T_high_Q_consistency": ("Q_consistency", [0.5, 0.6]),
}

# 异质性分组（可按业务再调整）
HETERO_GROUPS = {
    "high_precision_tool": {"S4", "S5", "S7", "S10"},
    "regular_grasp_sort": {"S1", "S2", "S3", "S6", "S8", "S9"},
}


def _load_stage4_scores() -> pd.DataFrame:
    path = os.path.join(CONFIG["interim_dir"], "stage4_curation_scores.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到阶段五输入文件：{path}")
    return pd.read_csv(path)


def _load_stage4_oof_labels() -> pd.DataFrame:
    path = os.path.join(CONFIG["interim_dir"], "stage4_bc_trajectory_labels.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到 OOF 轨迹级标签文件：{path}。请先运行 stage4_labels 以生成 trajectory_imitation_score。"
        )
    return pd.read_csv(path)


def _load_stage2_quality(scene_ids: List[str]) -> pd.DataFrame:
    dfs = []
    for sid in scene_ids:
        p = os.path.join(CONFIG["processed_dir"], f"{str(sid).lower()}_stage2_quality_dataset.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if "trajectory_id" not in df.columns:
            continue
        df["scene_id"] = str(sid)
        df["global_id"] = df["scene_id"].astype(str) + "_" + df["trajectory_id"].astype(str)
        dfs.append(df)
    if len(dfs) == 0:
        return pd.DataFrame(columns=["global_id", "scene_id", "trajectory_id"])
    return pd.concat(dfs, axis=0, ignore_index=True)


def _build_causal_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    score_df = _load_stage4_scores()
    label_df = _load_stage4_oof_labels()

    if "split" in score_df.columns:
        score_df = score_df[score_df["split"] == "pool"].copy()

    required_score_cols = ["global_id", "trajectory_id", "scene_id", "Q_score", "Q_accuracy", "Q_consistency"]
    miss = [c for c in required_score_cols if c not in score_df.columns]
    if miss:
        raise ValueError(f"stage4_curation_scores 缺少字段: {miss}")

    required_label_cols = ["global_id", "trajectory_normalized_mse", "trajectory_imitation_score", "n_bc_windows"]
    miss = [c for c in required_label_cols if c not in label_df.columns]
    if miss:
        raise ValueError(f"stage4_bc_trajectory_labels 缺少字段: {miss}")

    merge_cols = ["global_id", "trajectory_normalized_mse", "trajectory_imitation_score", "n_bc_windows"]
    df = score_df.merge(label_df[merge_cols], on="global_id", how="inner")

    # 尽量补充阶段2中的可观测混杂协变量
    stage2_df = _load_stage2_quality(CONFIG.get("scene_ids", []))
    if len(stage2_df) > 0:
        cov_candidates = [
            "global_id",
            "trajectory_duration",
            "global_duration_s",
            "global_arm_path_l1",
            "global_effector_path_l1",
            "semantic_entropy",
            "semantic_transition_rate",
            "arm_action_range",
            "arm_state_jerk_energy",
        ]
        use_cols = [c for c in cov_candidates if c in stage2_df.columns]
        if "global_id" in use_cols and len(use_cols) > 1:
            df = df.merge(stage2_df[use_cols].drop_duplicates("global_id"), on="global_id", how="left")

    # 如果没有 task_name，用 scene_id 占位
    if "task_name" not in df.columns:
        df["task_name"] = df["scene_id"].astype(str)

    treat_rows = []
    for tname, (src_col, quantiles) in TREATMENT_SPECS.items():
        if src_col not in df.columns:
            continue
        for q in quantiles:
            thr = float(df[src_col].quantile(q))
            col = f"{tname}_q{int(q * 100)}"
            df[col] = (df[src_col] >= thr).astype(int)
            treat_rows.append(
                {
                    "treatment": col,
                    "source": src_col,
                    "quantile": float(q),
                    "threshold": thr,
                }
            )

    keep = [
        "global_id",
        "trajectory_id",
        "scene_id",
        "task_name",
        "trajectory_imitation_score",
        "trajectory_normalized_mse",
        "n_bc_windows",
        "Q_score",
        "Q_accuracy",
        "Q_consistency",
        "trajectory_duration",
        "global_duration_s",
        "global_arm_path_l1",
        "global_effector_path_l1",
        "semantic_entropy",
        "semantic_transition_rate",
        "arm_action_range",
        "arm_state_jerk_energy",
    ]
    treat_cols = [c for c in df.columns if c.startswith("T_high_")]
    keep = [c for c in keep if c in df.columns]

    out = df[keep + treat_cols].copy()
    return out, pd.DataFrame(treat_rows)


def _choose_confounders(df: pd.DataFrame, treatment_col: str) -> List[str]:
    covs = []
    candidates = [
        "n_bc_windows",
        "trajectory_duration",
        "global_duration_s",
        "global_arm_path_l1",
        "global_effector_path_l1",
        "semantic_entropy",
        "semantic_transition_rate",
        "arm_action_range",
        "arm_state_jerk_energy",
    ]
    for c in candidates:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > 0:
            covs.append(c)

    # 关键：不要把 treatment 源变量本身纳入协变量
    if "Q_score" in treatment_col and "Q_score" in covs:
        covs.remove("Q_score")
    if "Q_accuracy" in treatment_col and "Q_accuracy" in covs:
        covs.remove("Q_accuracy")
    if "Q_consistency" in treatment_col and "Q_consistency" in covs:
        covs.remove("Q_consistency")

    return covs


def _fit_propensity(X: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )
    model.fit(X, t)
    p = model.predict_proba(X)[:, 1]
    return np.clip(p, 0.01, 0.99)


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def _smd(x_t: np.ndarray, x_c: np.ndarray) -> float:
    mt, mc = float(np.mean(x_t)), float(np.mean(x_c))
    vt, vc = float(np.var(x_t, ddof=1)), float(np.var(x_c, ddof=1))
    denom = np.sqrt(max((vt + vc) / 2.0, 1e-12))
    return (mt - mc) / denom


def _balance_table(before_df: pd.DataFrame, after_df: pd.DataFrame, covs: List[str], tcol: str) -> pd.DataFrame:
    rows = []
    for c in covs:
        bt = before_df.loc[before_df[tcol] == 1, c].dropna().values
        bc = before_df.loc[before_df[tcol] == 0, c].dropna().values
        at = after_df.loc[after_df[tcol] == 1, c].dropna().values
        ac = after_df.loc[after_df[tcol] == 0, c].dropna().values
        if min(len(bt), len(bc), len(at), len(ac)) == 0:
            continue
        rows.append(
            {
                "covariate": c,
                "smd_before": _smd(bt, bc),
                "abs_smd_before": abs(_smd(bt, bc)),
                "smd_after": _smd(at, ac),
                "abs_smd_after": abs(_smd(at, ac)),
            }
        )
    return pd.DataFrame(rows)


def _match_within_scene(df: pd.DataFrame, tcol: str, ycol: str, p: np.ndarray, caliper_ratio: float = 0.2):
    x = df.copy().reset_index(drop=True)
    x["_p"] = p
    x["_logitp"] = _logit(x["_p"].values)
    logit_sd = float(np.std(x["_logitp"].values, ddof=1))
    caliper = caliper_ratio * logit_sd

    pairs = []
    for scene, sdf in x.groupby("scene_id"):
        tdf = sdf[sdf[tcol] == 1].copy()
        cdf = sdf[sdf[tcol] == 0].copy()
        if len(tdf) == 0 or len(cdf) == 0:
            continue

        cdf = cdf.copy()
        cdf["_used"] = 0
        for i, tr in tdf.iterrows():
            avail = cdf[cdf["_used"] == 0].copy()
            if len(avail) == 0:
                break
            dist = np.abs(avail["_logitp"].values - tr["_logitp"])
            j = int(np.argmin(dist))
            d = float(dist[j])
            if d <= caliper:
                ctrl_idx = avail.index[j]
                cdf.loc[ctrl_idx, "_used"] = 1
                pairs.append(
                    {
                        "scene_id": scene,
                        "treated_idx": int(i),
                        "control_idx": int(ctrl_idx),
                        "distance_logit": d,
                        "treated_global_id": tr["global_id"],
                        "control_global_id": cdf.loc[ctrl_idx, "global_id"],
                        "treated_y": float(tr[ycol]),
                        "control_y": float(cdf.loc[ctrl_idx, ycol]),
                    }
                )

    pair_df = pd.DataFrame(pairs)
    if len(pair_df) == 0:
        return np.nan, pair_df, pd.DataFrame(), caliper

    att = float((pair_df["treated_y"] - pair_df["control_y"]).mean())

    mt = x.loc[pair_df["treated_idx"].values].copy()
    mc = x.loc[pair_df["control_idx"].values].copy()
    mt["_matched_role"] = 1
    mc["_matched_role"] = 0
    matched_df = pd.concat([mt, mc], axis=0, ignore_index=True)

    return att, pair_df, matched_df, caliper


def _bootstrap_ci(pair_df: pd.DataFrame, y_sign: float, n_boot: int = 500) -> Tuple[float, float]:
    if len(pair_df) == 0:
        return np.nan, np.nan
    rng = np.random.RandomState(42)
    vals = []
    dif = (pair_df["treated_y"].values - pair_df["control_y"].values) * y_sign
    n = len(dif)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        vals.append(float(np.mean(dif[idx])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _hetero_label(scene_id: str) -> str:
    s = str(scene_id)
    for k, v in HETERO_GROUPS.items():
        if s in v:
            return k
    return "other"


def run_stage5_causal_analysis():
    causal_df, treat_info_df = _build_causal_dataset()

    if len(causal_df) == 0:
        raise ValueError("stage5 因果分析数据为空。")

    causal_df["hetero_group"] = causal_df["scene_id"].map(_hetero_label)

    treat_cols = [c for c in causal_df.columns if c.startswith("T_high_")]
    outcomes = [c for c in [OUTCOME_PRIMARY, OUTCOME_SECONDARY] if c in causal_df.columns]

    effect_rows = []
    balance_rows = []
    pair_rows = []

    for tcol in treat_cols:
        for ycol in outcomes:
            # imitation_score: ATT>0更好；nmse: ATT<0更好
            y_sign = 1.0 if ycol == OUTCOME_PRIMARY else -1.0

            for group_name in ["all", "high_precision_tool", "regular_grasp_sort"]:
                if group_name == "all":
                    gdf = causal_df.copy()
                else:
                    gdf = causal_df[causal_df["hetero_group"] == group_name].copy()

                if len(gdf) < 30:
                    continue
                if gdf[tcol].nunique() < 2:
                    continue

                covs = _choose_confounders(gdf, tcol)
                if len(covs) == 0:
                    continue

                t = gdf[tcol].astype(int).values
                p = _fit_propensity(gdf[covs], t)
                att, pair_df, matched_df, caliper = _match_within_scene(gdf, tcol, ycol, p, caliper_ratio=0.2)
                if np.isnan(att):
                    continue

                bal_df = _balance_table(gdf[[tcol] + covs], matched_df[[tcol] + covs], covs, tcol)
                if len(bal_df) > 0:
                    bal_df.insert(0, "treatment", tcol)
                    bal_df.insert(1, "outcome", ycol)
                    bal_df.insert(2, "heterogeneity_group", group_name)
                    bal_df.insert(3, "match_method", "within_scene_1to1_nn_caliper0.2logit")
                    balance_rows.append(bal_df)

                lo, hi = _bootstrap_ci(pair_df, y_sign=y_sign, n_boot=500)

                effect_rows.append(
                    {
                        "treatment": tcol,
                        "outcome": ycol,
                        "heterogeneity_group": group_name,
                        "match_method": "within_scene_1to1_nn_caliper0.2logit",
                        "n_treated_matched": int(len(pair_df)),
                        "att_raw": float(att),
                        "att_beneficial_direction": float(att * y_sign),
                        "ci95_lo_beneficial": lo,
                        "ci95_hi_beneficial": hi,
                        "caliper_logit": float(caliper),
                        "mean_abs_smd_before": float(bal_df["abs_smd_before"].mean()) if len(bal_df) > 0 else np.nan,
                        "mean_abs_smd_after": float(bal_df["abs_smd_after"].mean()) if len(bal_df) > 0 else np.nan,
                    }
                )

                if len(pair_df) > 0:
                    pair_df.insert(0, "treatment", tcol)
                    pair_df.insert(1, "outcome", ycol)
                    pair_df.insert(2, "heterogeneity_group", group_name)
                    pair_df.insert(3, "match_method", "within_scene_1to1_nn_caliper0.2logit")
                    pair_rows.append(pair_df)

    effect_df = pd.DataFrame(effect_rows)
    balance_df = pd.concat(balance_rows, axis=0, ignore_index=True) if len(balance_rows) > 0 else pd.DataFrame()
    matched_pairs_df = pd.concat(pair_rows, axis=0, ignore_index=True) if len(pair_rows) > 0 else pd.DataFrame()

    # 兼容并新增建议文件名
    save_csv(causal_df, os.path.join(CONFIG["interim_dir"], "stage5_causal_dataset.csv"))
    save_csv(causal_df, os.path.join(CONFIG["interim_dir"], "stage5_psm_dataset.csv"))
    save_csv(treat_info_df, os.path.join(CONFIG["interim_dir"], "stage5_treatment_summary.csv"))
    save_csv(balance_df, os.path.join(CONFIG["interim_dir"], "stage5_balance_summary.csv"))
    save_csv(balance_df, os.path.join(CONFIG["interim_dir"], "stage5_psm_balance.csv"))
    save_csv(effect_df, os.path.join(CONFIG["interim_dir"], "stage5_effect_estimates.csv"))
    save_csv(effect_df, os.path.join(CONFIG["interim_dir"], "stage5_psm_att_summary.csv"))
    save_csv(matched_pairs_df, os.path.join(CONFIG["interim_dir"], "stage5_matched_pairs.csv"))
    save_csv(matched_pairs_df, os.path.join(CONFIG["interim_dir"], "stage5_psm_matched_pairs.csv"))

    # 轻量稳健性汇总：按 treatment/outcome 对比 q50/q60
    robustness_df = pd.DataFrame()
    if len(effect_df) > 0:
        tmp = effect_df.copy()
        tmp["threshold_tag"] = tmp["treatment"].str.extract(r"_q(\d+)$", expand=False)
        robustness_df = (
            tmp.groupby(["heterogeneity_group", "outcome", "threshold_tag"], as_index=False)
            .agg(
                n_estimates=("att_beneficial_direction", "count"),
                att_beneficial_mean=("att_beneficial_direction", "mean"),
                ci95_lo_mean=("ci95_lo_beneficial", "mean"),
                ci95_hi_mean=("ci95_hi_beneficial", "mean"),
            )
            .sort_values(["heterogeneity_group", "outcome", "threshold_tag"])
        )
    save_csv(robustness_df, os.path.join(CONFIG["interim_dir"], "stage5_psm_robustness.csv"))

    hetero_df = effect_df.copy()
    save_csv(hetero_df, os.path.join(CONFIG["interim_dir"], "stage5_heterogeneity_att.csv"))

    # run_stage5.py 兼容返回位
    overlap_df = pd.DataFrame()
    weights_df = pd.DataFrame()
    return causal_df, effect_df, balance_df, matched_pairs_df, weights_df, overlap_df, treat_info_df
