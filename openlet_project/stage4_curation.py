import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import CONFIG
from stage3_bc_value import (
    set_seed,
    load_all_scenes_aligned_data,
    build_bc_dataset,
    train_bc_model,
    predict_bc,
)
from utils import save_csv


@dataclass
class CurationConfig:
    test_ratio: float = 0.2
    split_seed: int = 42
    ratio_grid: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0)
    run_seeds: Tuple[int, ...] = (42, 2024, 2025)
    random_repeat_seeds: Tuple[int, ...] = (42, 2024, 2025, 3407, 4096)


def _zscore(s: pd.Series) -> pd.Series:
    std = float(s.std(ddof=0))
    if std < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - float(s.mean())) / std


def _compute_eval_action_std(all_aligned: Dict, pool_ids: List[str]) -> np.ndarray:
    _, y_pool, _ = build_bc_dataset(all_aligned, pool_ids)
    eval_std = np.nanstd(y_pool, axis=0).astype(np.float32)
    eval_std[eval_std < 1e-8] = 1.0
    return eval_std


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, eval_std: np.ndarray) -> Dict:
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    nmse = float(np.mean((err / eval_std.reshape(1, -1)) ** 2))
    score = float(np.exp(-nmse))
    return {"test_nmse": nmse, "test_imitation_score": score, "test_mae": mae, "test_mse": mse}


def _stratified_split_test_pool(traj_info_df: pd.DataFrame, cfg: CurationConfig) -> pd.DataFrame:
    rng = np.random.RandomState(cfg.split_seed)
    rows = []
    for scene_id, sdf in traj_info_df.groupby("scene_id"):
        ids = sdf["global_id"].tolist()
        rng.shuffle(ids)
        n_test = max(1, int(round(len(ids) * cfg.test_ratio)))
        test_ids = set(ids[:n_test])
        for gid in ids:
            rows.append(
                {
                    "global_id": gid,
                    "trajectory_id": gid.split("_", 1)[1],
                    "scene_id": scene_id,
                    "split": "test" if gid in test_ids else "pool",
                }
            )
    return pd.DataFrame(rows)


def _load_stage2_quality(scene_ids: List[str]) -> pd.DataFrame:
    dfs = []
    for sid in scene_ids:
        path = os.path.join(CONFIG["processed_dir"], f"{sid.lower()}_stage2_quality_dataset.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到阶段二质量文件: {path}")
        df = pd.read_csv(path)
        if "trajectory_id" not in df.columns:
            raise ValueError(f"{path} 缺少 trajectory_id")
        df["scene_id"] = sid
        df["global_id"] = df["scene_id"].astype(str) + "_" + df["trajectory_id"].astype(str)
        dfs.append(df)
    qdf = pd.concat(dfs, axis=0, ignore_index=True)
    needed = ["global_id", "trajectory_id", "scene_id", "Q_score", "Q_accuracy", "Q_consistency"]
    miss = [c for c in needed if c not in qdf.columns]
    if miss:
        raise ValueError(f"阶段二文件缺少字段: {miss}")
    return qdf


def _load_stage3_scene_delta(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到阶段三场景价值文件: {path}")
    df = pd.read_csv(path)
    if "delta_score_mean" in df.columns:
        val_col = "delta_score_mean"
    elif "delta_score" in df.columns:
        val_col = "delta_score"
    else:
        raise ValueError("阶段三价值文件需包含 delta_score_mean 或 delta_score")
    if "scene_id" not in df.columns:
        raise ValueError("阶段三价值文件缺少 scene_id")
    return df[["scene_id", val_col]].rename(columns={val_col: "delta_score_emp"})


def _build_score_table(split_df: pd.DataFrame, qdf: pd.DataFrame, sdf: pd.DataFrame) -> pd.DataFrame:
    df = split_df.merge(qdf[["global_id", "trajectory_id", "scene_id", "Q_score", "Q_accuracy", "Q_consistency"]], on=["global_id", "trajectory_id", "scene_id"], how="left")
    df = df.merge(sdf, on="scene_id", how="left")

    for col in ["Q_score", "Q_consistency", "delta_score_emp"]:
        if df[col].isna().any():
            raise ValueError(f"{col} 存在缺失，无法继续策展")

    df["z_Q_score"] = _zscore(df["Q_score"])
    df["z_Q_consistency"] = _zscore(df["Q_consistency"])
    df["z_delta_score_emp"] = _zscore(df["delta_score_emp"])
    df["hybrid_score_Q"] = 0.5 * df["z_Q_score"] + 0.5 * df["z_delta_score_emp"]
    df["hybrid_score_consistency"] = 0.5 * df["z_Q_consistency"] + 0.5 * df["z_delta_score_emp"]

    return df


def _select_ids(pool_df: pd.DataFrame, strategy: str, ratio: float, seed: int) -> pd.DataFrame:
    n_pick = max(1, int(round(len(pool_df) * ratio)))

    if strategy == "full":
        selected = pool_df.copy()
        selected = selected.sort_values("global_id").reset_index(drop=True)
        selected["selection_score"] = 1.0

    elif strategy == "random":
        selected = pool_df.sample(n=n_pick, random_state=seed).copy()
        selected["selection_score"] = np.nan

    elif strategy == "high_Q_global":
        selected = pool_df.nlargest(n_pick, "Q_score").copy()
        selected["selection_score"] = selected["Q_score"]

    elif strategy == "high_Q_stratified":
        parts = []
        for _, sdf in pool_df.groupby("scene_id"):
            n_scene = max(1, int(round(len(sdf) * ratio)))
            parts.append(sdf.nlargest(n_scene, "Q_score"))
        selected = pd.concat(parts, axis=0).drop_duplicates("global_id")
        selected["selection_score"] = selected["Q_score"]

    elif strategy == "high_delta":
        selected = pool_df.nlargest(n_pick, "delta_score_emp").copy()
        selected["selection_score"] = selected["delta_score_emp"]

    elif strategy == "hybrid_Q_delta":
        selected = pool_df.nlargest(n_pick, "hybrid_score_Q").copy()
        selected["selection_score"] = selected["hybrid_score_Q"]

    elif strategy == "hybrid_consistency_delta":
        selected = pool_df.nlargest(n_pick, "hybrid_score_consistency").copy()
        selected["selection_score"] = selected["hybrid_score_consistency"]

    elif strategy == "low_Q_score":
        selected = pool_df.nsmallest(n_pick, "Q_score").copy()
        selected["selection_score"] = selected["Q_score"]

    elif strategy == "low_delta":
        selected = pool_df.nsmallest(n_pick, "delta_score_emp").copy()
        selected["selection_score"] = selected["delta_score_emp"]

    else:
        raise ValueError(f"未知策略: {strategy}")

    selected = selected.reset_index(drop=True)
    selected["selected_rank"] = np.arange(1, len(selected) + 1)
    return selected


def _train_eval_once(all_aligned: Dict, train_ids: List[str], test_ids: List[str], eval_std: np.ndarray):
    X_train, y_train, _ = build_bc_dataset(all_aligned, train_ids)
    X_test, y_test, test_meta = build_bc_dataset(all_aligned, test_ids)

    model, X_scaler, y_scaler, _ = train_bc_model(X_train, y_train, X_test, y_test)
    y_pred, _ = predict_bc(model, X_test, X_scaler, y_scaler)

    global_metrics = _compute_metrics(y_test, y_pred, eval_std)

    per_scene_rows = []
    for scene in sorted(test_meta["scene_id"].unique()):
        idx = test_meta["scene_id"].values == scene
        m = _compute_metrics(y_test[idx], y_pred[idx], eval_std)
        per_scene_rows.append(
            {
                "test_scene": scene,
                "n_test_trajectories": int(len(set(test_meta.loc[idx, "global_id"]))),
                "scene_nmse": m["test_nmse"],
                "scene_imitation_score": m["test_imitation_score"],
                "scene_mae": m["test_mae"],
            }
        )

    return global_metrics, pd.DataFrame(per_scene_rows), int(len(X_train))


def run_stage4_curation():
    cfg = CurationConfig()
    set_seed(CONFIG["random_state"])

    all_aligned, traj_info_df = load_all_scenes_aligned_data(CONFIG["scene_ids"])

    split_df = _stratified_split_test_pool(traj_info_df, cfg)
    save_csv(split_df, os.path.join(CONFIG["interim_dir"], "stage4_curation_split.csv"))

    qdf = _load_stage2_quality(CONFIG["scene_ids"])
    delta_path = CONFIG.get("stage4_stage3_delta_path", os.path.join(CONFIG["interim_dir"], "stage3_repeat_delta_summary_11scene_multiseed.csv"))
    sdf = _load_stage3_scene_delta(delta_path)

    scores_df = _build_score_table(split_df, qdf, sdf)
    save_csv(scores_df, os.path.join(CONFIG["interim_dir"], "stage4_curation_scores.csv"))

    pool_df = scores_df[scores_df["split"] == "pool"].copy()
    test_ids = scores_df[scores_df["split"] == "test"]["global_id"].tolist()
    eval_std = _compute_eval_action_std(all_aligned, pool_df["global_id"].tolist())

    strategies = [
        "full", "random", "high_Q_global", "high_Q_stratified", "high_delta",
        "hybrid_Q_delta", "hybrid_consistency_delta", "low_Q_score", "low_delta"
    ]

    selection_rows = []
    metric_rows = []
    per_scene_rows = []

    full_nmse = None
    random_ref = {}

    for strategy in strategies:
        ratios = (1.0,) if strategy == "full" else cfg.ratio_grid
        seeds = cfg.random_repeat_seeds if strategy == "random" else cfg.run_seeds

        for ratio in ratios:
            for run_seed in seeds:
                selected = _select_ids(pool_df, strategy, ratio, run_seed)
                train_ids = selected["global_id"].tolist()

                g_m, s_df, n_windows = _train_eval_once(all_aligned, train_ids, test_ids, eval_std)

                if strategy == "full":
                    full_nmse = g_m["test_nmse"]
                if strategy == "random":
                    random_ref.setdefault(ratio, []).append(g_m["test_nmse"])

                for _, r in selected.iterrows():
                    selection_rows.append(
                        {
                            "strategy": strategy,
                            "ratio": ratio,
                            "run_seed": run_seed,
                            "global_id": r["global_id"],
                            "trajectory_id": r["trajectory_id"],
                            "scene_id": r["scene_id"],
                            "selected_rank": int(r["selected_rank"]),
                            "selection_score": r["selection_score"],
                        }
                    )

                metric_rows.append(
                    {
                        "strategy": strategy,
                        "ratio": ratio,
                        "run_seed": run_seed,
                        "n_train_trajectories": int(len(train_ids)),
                        "n_train_windows": int(n_windows),
                        "test_nmse": g_m["test_nmse"],
                        "test_imitation_score": g_m["test_imitation_score"],
                        "test_mae": g_m["test_mae"],
                    }
                )

                for _, srow in s_df.iterrows():
                    per_scene_rows.append(
                        {
                            "strategy": strategy,
                            "ratio": ratio,
                            "run_seed": run_seed,
                            **srow.to_dict(),
                        }
                    )

    selection_df = pd.DataFrame(selection_rows)
    metrics_df = pd.DataFrame(metric_rows)
    per_scene_df = pd.DataFrame(per_scene_rows)

    rand_mean_by_ratio = {k: float(np.mean(v)) for k, v in random_ref.items()}
    metrics_df["relative_to_random_nmse"] = metrics_df.apply(
        lambda x: (rand_mean_by_ratio.get(x["ratio"], np.nan) - x["test_nmse"]) / rand_mean_by_ratio.get(x["ratio"], np.nan)
        if pd.notna(rand_mean_by_ratio.get(x["ratio"], np.nan)) else np.nan,
        axis=1,
    )
    metrics_df["relative_to_full_nmse"] = metrics_df["test_nmse"].apply(
        lambda v: (full_nmse - v) / full_nmse if (full_nmse is not None and full_nmse > 0) else np.nan
    )

    summary_df = (
        metrics_df.groupby(["strategy", "ratio"], as_index=False)
        .agg(
            n_runs=("run_seed", "count"),
            test_nmse_mean=("test_nmse", "mean"),
            test_nmse_std=("test_nmse", "std"),
            test_imitation_score_mean=("test_imitation_score", "mean"),
            test_imitation_score_std=("test_imitation_score", "std"),
            relative_to_random_nmse_mean=("relative_to_random_nmse", "mean"),
            relative_to_full_nmse_mean=("relative_to_full_nmse", "mean"),
        )
    )

    save_csv(selection_df, os.path.join(CONFIG["interim_dir"], "stage4_curation_selection_table.csv"))
    save_csv(metrics_df, os.path.join(CONFIG["interim_dir"], "stage4_curation_metrics.csv"))
    save_csv(per_scene_df, os.path.join(CONFIG["interim_dir"], "stage4_curation_per_scene_metrics.csv"))
    save_csv(summary_df, os.path.join(CONFIG["interim_dir"], "stage4_curation_summary.csv"))

    return split_df, scores_df, selection_df, metrics_df, per_scene_df, summary_df
