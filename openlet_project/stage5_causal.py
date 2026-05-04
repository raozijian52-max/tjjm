import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CONFIG
from utils import save_csv


# 主结果变量：优先 trajectory_normalized_mse（越低越好）
# 兜底：若不存在，则回退到 Q_score（仅保证基础版可运行）
MAIN_OUTCOME = "trajectory_normalized_mse"

# 基础版阈值稳健性
THRESHOLD_SPECS = {
    "high_quality": [0.5, 0.6],      # 中位数 / top40%
    "high_accuracy": [0.5, 0.6],     # 中位数 / top40%
    "high_delta_scene": [0.5, 0.7],  # 中位数 / top30%
}


def _load_stage4_scores() -> pd.DataFrame:
    path = os.path.join(CONFIG["interim_dir"], "stage4_curation_scores.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到阶段五输入文件：{path}")

    df = pd.read_csv(path)
    required = ["global_id", "trajectory_id", "scene_id", "Q_score", "Q_accuracy", "delta_score_emp"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"stage4_curation_scores 缺少字段：{miss}")
    return df


def _ensure_basic_covariates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "trajectory_length" not in out.columns:
        out["trajectory_length"] = np.nan
    if "path_length" not in out.columns:
        out["path_length"] = np.nan
    if "arm_motion_stat" not in out.columns:
        out["arm_motion_stat"] = np.nan
    if "effector_motion_stat" not in out.columns:
        out["effector_motion_stat"] = np.nan
    if "semantic_entropy" not in out.columns:
        out["semantic_entropy"] = np.nan
    if "task_name" not in out.columns:
        out["task_name"] = out["scene_id"].astype(str)
    return out


def _ensure_main_outcome(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    out = df.copy()
    if MAIN_OUTCOME in out.columns and pd.api.types.is_numeric_dtype(out[MAIN_OUTCOME]):
        return out, MAIN_OUTCOME

    # 兜底：若没有轨迹级 mse，先用 Q_score 的反向代理（越小越好）
    # 该兜底仅为“能跑基础版”；建议后续替换为真实轨迹级 BC 误差。
    out["trajectory_normalized_mse"] = 1.0 - out["Q_score"].astype(float)
    return out, "trajectory_normalized_mse"


def _make_treatment_cols(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    info_rows = []

    # 处理变量一：高质量轨迹（Q_score）
    for q in THRESHOLD_SPECS["high_quality"]:
        thr = float(out["Q_score"].quantile(q))
        col = f"treat_high_quality_q{int(q*100)}"
        out[col] = (out["Q_score"] >= thr).astype(int)
        info_rows.append({"treatment": col, "source": "Q_score", "quantile": q, "threshold": thr})

    # 处理变量二：高准确性轨迹（Q_accuracy）
    for q in THRESHOLD_SPECS["high_accuracy"]:
        thr = float(out["Q_accuracy"].quantile(q))
        col = f"treat_high_accuracy_q{int(q*100)}"
        out[col] = (out["Q_accuracy"] >= thr).astype(int)
        info_rows.append({"treatment": col, "source": "Q_accuracy", "quantile": q, "threshold": thr})

    # 处理变量三：高效能价值场景（delta_score_emp，场景级映射到轨迹）
    scene_delta = out.groupby("scene_id", as_index=False)["delta_score_emp"].mean()
    for q in THRESHOLD_SPECS["high_delta_scene"]:
        thr = float(scene_delta["delta_score_emp"].quantile(q))
        high_scene_set = set(scene_delta.loc[scene_delta["delta_score_emp"] >= thr, "scene_id"].astype(str))
        col = f"treat_high_delta_scene_q{int(q*100)}"
        out[col] = out["scene_id"].astype(str).isin(high_scene_set).astype(int)
        info_rows.append({"treatment": col, "source": "delta_score_emp(scene)", "quantile": q, "threshold": thr})

    return out, pd.DataFrame(info_rows)


def build_causal_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    df = _load_stage4_scores()
    df = df[df["split"] == "pool"].copy() if "split" in df.columns else df.copy()
    df = _ensure_basic_covariates(df)
    df, outcome_col = _ensure_main_outcome(df)
    df, treat_info = _make_treatment_cols(df)

    keep = [
        "global_id", "trajectory_id", "scene_id", "task_name",
        outcome_col,
        "Q_score", "Q_accuracy", "Q_consistency", "delta_score_emp",
        "trajectory_length", "path_length", "arm_motion_stat", "effector_motion_stat", "semantic_entropy",
    ]
    keep = [c for c in keep if c in df.columns]

    treat_cols = [c for c in df.columns if c.startswith("treat_")]
    causal_df = df[keep + treat_cols].copy()

    return causal_df, treat_info, outcome_col


def get_confounder_cols(causal_df: pd.DataFrame, with_scene_fe: bool) -> Tuple[List[str], pd.DataFrame]:
    model_df = causal_df.copy()
    confounder_cols = []

    numeric_covs = [
        "trajectory_length", "path_length", "arm_motion_stat", "effector_motion_stat", "semantic_entropy",
        "Q_score", "Q_accuracy", "Q_consistency", "delta_score_emp",
    ]
    for c in numeric_covs:
        if c in model_df.columns and pd.api.types.is_numeric_dtype(model_df[c]):
            # 避免整列全缺失触发 sklearn imputing 警告刷屏
            if model_df[c].notna().sum() > 0:
                confounder_cols.append(c)

    if "task_name" in model_df.columns:
        task_d = pd.get_dummies(model_df["task_name"].astype(str), prefix="task", dtype=float)
        model_df = pd.concat([model_df, task_d], axis=1)
        confounder_cols += list(task_d.columns)

    if with_scene_fe and "scene_id" in model_df.columns:
        scene_d = pd.get_dummies(model_df["scene_id"].astype(str), prefix="scene", dtype=float)
        model_df = pd.concat([model_df, scene_d], axis=1)
        confounder_cols += list(scene_d.columns)

    return confounder_cols, model_df


def fit_propensity(df: pd.DataFrame, treatment_col: str, confounders: List[str]) -> np.ndarray:
    t = df[treatment_col].astype(int).values
    if t.min() == t.max():
        raise ValueError(f"{treatment_col} 全为同一类，无法拟合倾向得分")

    x = df[confounders]
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    model.fit(x, t)
    p = model.predict_proba(x)[:, 1]
    return np.clip(p, 0.01, 0.99)


def _weighted_mean(v: np.ndarray, w: np.ndarray) -> float:
    d = np.sum(w)
    if d <= 0:
        return np.nan
    return float(np.sum(v * w) / d)


def est_raw(df: pd.DataFrame, tcol: str, ycol: str) -> Dict:
    yt = df.loc[df[tcol] == 1, ycol].values
    yc = df.loc[df[tcol] == 0, ycol].values
    est = float(np.mean(yt) - np.mean(yc))
    return {"method": "raw_difference", "estimate": est, "n_used": int(len(yt) + len(yc))}


def est_psm(df: pd.DataFrame, tcol: str, ycol: str, p: np.ndarray) -> Tuple[Dict, pd.DataFrame]:
    x = df.copy().reset_index(drop=True)
    x["_p"] = p
    tdf = x[x[tcol] == 1].copy()
    cdf = x[x[tcol] == 0].copy()
    if len(tdf) == 0 or len(cdf) == 0:
        return {"method": "psm_att", "estimate": np.nan, "n_used": 0}, pd.DataFrame()

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(cdf[["_p"]].values)
    dist, idx = nn.kneighbors(tdf[["_p"]].values)
    mctrl = cdf.iloc[idx.flatten()].reset_index(drop=True)
    mtrt = tdf.reset_index(drop=True)

    est = float(np.mean(mtrt[ycol].values - mctrl[ycol].values))
    pairs = pd.DataFrame({
        "treated_global_id": mtrt["global_id"].values,
        "control_global_id": mctrl["global_id"].values,
        "distance": dist.flatten(),
    })
    return {"method": "psm_att", "estimate": est, "n_used": int(len(mtrt))}, pairs


def est_ipw(df: pd.DataFrame, tcol: str, ycol: str, p: np.ndarray) -> Tuple[Dict, pd.DataFrame]:
    t = df[tcol].astype(int).values
    y = df[ycol].values
    pt = float(np.mean(t))
    pc = 1.0 - pt
    sw = t * pt / p + (1 - t) * pc / (1 - p)

    mu_t = _weighted_mean(y[t == 1], sw[t == 1])
    mu_c = _weighted_mean(y[t == 0], sw[t == 0])
    est = float(mu_t - mu_c)

    wdf = pd.DataFrame({
        "global_id": df["global_id"].values,
        "scene_id": df["scene_id"].values,
        "treat": t,
        "propensity": p,
        "stabilized_ipw_weight": sw,
    })
    return {"method": "ipw_ate", "estimate": est, "n_used": int(len(df))}, wdf


def est_weighted_regression(df: pd.DataFrame, tcol: str, ycol: str, confs: List[str], p: np.ndarray) -> Dict:
    # 线性加权回归（WLS）近似：在标准化协变量上最小二乘闭式解
    t = df[tcol].astype(float).values.reshape(-1, 1)
    y = df[ycol].astype(float).values.reshape(-1, 1)

    x_cov = df[confs].copy()
    x_cov = x_cov.fillna(x_cov.median(numeric_only=True))
    x_cov = x_cov.fillna(0.0)
    x_cov_arr = x_cov.values.astype(float)

    # 设计矩阵：[1, T, X]
    X = np.concatenate([np.ones((len(df), 1)), t, x_cov_arr], axis=1)

    tr = t.flatten()
    pt = float(np.mean(tr))
    pc = 1.0 - pt
    w = tr * pt / p + (1 - tr) * pc / (1 - p)

    W = np.sqrt(w).reshape(-1, 1)
    Xw = X * W
    yw = y * W

    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    est = float(beta[1, 0])
    return {"method": "weighted_regression", "estimate": est, "n_used": int(len(df))}


def run_one(df: pd.DataFrame, tcol: str, ycol: str, group_label: str, with_scene_fe: bool):
    confs, mdf = get_confounder_cols(df, with_scene_fe=with_scene_fe)
    use_cols = ["global_id", "scene_id", tcol, ycol] + confs
    mdf = mdf[use_cols].copy()

    p = fit_propensity(mdf, tcol, confs)

    out = []
    raw = est_raw(mdf, tcol, ycol)
    psm, pairs = est_psm(mdf, tcol, ycol, p)
    ipw, wdf = est_ipw(mdf, tcol, ycol, p)
    wls = est_weighted_regression(mdf, tcol, ycol, confs, p)

    for r in [raw, psm, ipw, wls]:
        r.update({
            "treatment": tcol,
            "analysis_variant": "with_scene" if with_scene_fe else "no_scene",
            "outcome": ycol,
            "effect_type": r["method"].upper(),
            "heterogeneity_group": group_label,
            "with_scene_fe": int(with_scene_fe),
            "direction": "beneficial" if r["estimate"] < 0 else ("harmful" if r["estimate"] > 0 else "neutral"),
        })
        out.append(r)

    if len(pairs) > 0:
        pairs.insert(0, "treatment", tcol)
        pairs.insert(1, "heterogeneity_group", group_label)
        pairs.insert(2, "with_scene_fe", int(with_scene_fe))
    if len(wdf) > 0:
        wdf.insert(0, "treatment", tcol)
        wdf.insert(1, "heterogeneity_group", group_label)
        wdf.insert(2, "with_scene_fe", int(with_scene_fe))

    overlap = pd.DataFrame([{
        "treatment": tcol,
        "heterogeneity_group": group_label,
        "with_scene_fe": int(with_scene_fe),
        "treated_ps_min": float(np.min(p[mdf[tcol] == 1])),
        "treated_ps_max": float(np.max(p[mdf[tcol] == 1])),
        "control_ps_min": float(np.min(p[mdf[tcol] == 0])),
        "control_ps_max": float(np.max(p[mdf[tcol] == 0])),
    }])

    return pd.DataFrame(out), pairs, wdf, overlap


def run_stage5_causal_analysis():
    causal_df, treat_info_df, outcome_col = build_causal_dataset()

    treat_cols = [c for c in causal_df.columns if c.startswith("treat_")]

    # 异质性：全样本 + 分任务（task_name）
    groups = [("all", causal_df)]
    if "task_name" in causal_df.columns:
        for task, gdf in causal_df.groupby("task_name"):
            groups.append((f"task={task}", gdf.copy()))

    effect_list, pair_list, weight_list, overlap_list = [], [], [], []

    for tcol in treat_cols:
        for gname, gdf in groups:
            if len(gdf) < 20:
                continue
            for with_scene_fe in [False, True]:
                try:
                    e, p, w, o = run_one(gdf, tcol, outcome_col, gname, with_scene_fe)
                    effect_list.append(e)
                    if len(p) > 0:
                        pair_list.append(p)
                    if len(w) > 0:
                        weight_list.append(w)
                    overlap_list.append(o)
                except ValueError as ex:
                    print(f"[Stage5][Skip] {tcol} | {gname} | sceneFE={with_scene_fe}: {ex}")

    effect_df = pd.concat(effect_list, axis=0, ignore_index=True) if effect_list else pd.DataFrame()
    matched_pairs_df = pd.concat(pair_list, axis=0, ignore_index=True) if pair_list else pd.DataFrame()
    weights_df = pd.concat(weight_list, axis=0, ignore_index=True) if weight_list else pd.DataFrame()
    overlap_df = pd.concat(overlap_list, axis=0, ignore_index=True) if overlap_list else pd.DataFrame()

    # 基础版无单独 balance 表，这里留空兼容 run_stage5.py
    balance_df = pd.DataFrame()

    save_csv(causal_df, os.path.join(CONFIG["interim_dir"], "stage5_causal_dataset.csv"))
    save_csv(treat_info_df, os.path.join(CONFIG["interim_dir"], "stage5_treatment_summary.csv"))
    save_csv(overlap_df, os.path.join(CONFIG["interim_dir"], "stage5_propensity_overlap.csv"))
    save_csv(balance_df, os.path.join(CONFIG["interim_dir"], "stage5_balance_summary.csv"))
    save_csv(effect_df, os.path.join(CONFIG["interim_dir"], "stage5_effect_estimates.csv"))
    save_csv(matched_pairs_df, os.path.join(CONFIG["interim_dir"], "stage5_matched_pairs.csv"))
    save_csv(weights_df, os.path.join(CONFIG["interim_dir"], "stage5_ipw_weights.csv"))

    return causal_df, effect_df, balance_df, matched_pairs_df, weights_df, overlap_df, treat_info_df
