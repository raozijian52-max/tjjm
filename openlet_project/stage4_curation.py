import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

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
from stage3_repeat import run_stage3_repeated_on_pool
from utils import save_csv


@dataclass
class CurationConfig:
    test_ratio: float = 0.2
    split_seed: int = 42
    ratio_grid: Tuple[float, ...] = (0.25, 0.5, 0.75)
    run_seeds: Tuple[int, ...] = (42, 2024, 2025)
    random_repeat_seeds: Tuple[int, ...] = (42, 2024, 2025, 3407, 4096)


def _build_curation_config(
    smoke_test: bool = False,
    ratio_grid: Tuple[float, ...] = None,
    run_seeds: Tuple[int, ...] = None,
    random_repeat_seeds: Tuple[int, ...] = None,
) -> CurationConfig:
    cfg = CurationConfig()
    cfg.test_ratio = float(CONFIG.get("stage4_test_ratio", cfg.test_ratio))
    cfg.ratio_grid = tuple(CONFIG.get("stage4_ratio_grid", list(cfg.ratio_grid)))
    cfg.run_seeds = tuple(CONFIG.get("stage4_run_seeds", list(cfg.run_seeds)))
    cfg.random_repeat_seeds = tuple(CONFIG.get("stage4_random_repeat_seeds", list(cfg.random_repeat_seeds)))

    if ratio_grid is not None:
        cfg.ratio_grid = tuple(ratio_grid)
    if run_seeds is not None:
        cfg.run_seeds = tuple(run_seeds)
    if random_repeat_seeds is not None:
        cfg.random_repeat_seeds = tuple(random_repeat_seeds)

    # smoke test：最小可跑配置，不依赖 config 开关
    if smoke_test:
        cfg.ratio_grid = (1.0,)
        cfg.run_seeds = (42,)
        cfg.random_repeat_seeds = (42,)

    return cfg


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
    df["z_Q_accuracy"] = _zscore(df["Q_accuracy"])
    df["z_Q_consistency"] = _zscore(df["Q_consistency"])
    df["z_delta_score_emp"] = _zscore(df["delta_score_emp"])
    df["hybrid_score_Q"] = 0.5 * df["z_Q_score"] + 0.5 * df["z_delta_score_emp"]
    df["hybrid_score_consistency"] = 0.5 * df["z_Q_consistency"] + 0.5 * df["z_delta_score_emp"]
    df["hybrid_score_Q08_delta02"] = 0.8 * df["z_Q_score"] + 0.2 * df["z_delta_score_emp"]
    df["hybrid_score_accuracy08_delta02"] = 0.8 * df["z_Q_accuracy"] + 0.2 * df["z_delta_score_emp"]

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

    elif strategy == "hybrid_Q08_delta02":
        selected = pool_df.nlargest(n_pick, "hybrid_score_Q08_delta02").copy()
        selected["selection_score"] = selected["hybrid_score_Q08_delta02"]

    elif strategy == "hybrid_accuracy08_delta02":
        selected = pool_df.nlargest(n_pick, "hybrid_score_accuracy08_delta02").copy()
        selected["selection_score"] = selected["hybrid_score_accuracy08_delta02"]

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


def run_stage4_prepare_pool_and_stage3(
    smoke_test: bool = False,
    ratio_grid: Tuple[float, ...] = None,
    run_seeds: Tuple[int, ...] = None,
    random_repeat_seeds: Tuple[int, ...] = None,
):
    cfg = _build_curation_config(
        smoke_test=smoke_test,
        ratio_grid=ratio_grid,
        run_seeds=run_seeds,
        random_repeat_seeds=random_repeat_seeds,
    )
    print("[Stage4:prepare] start")
    print(f"[Stage4:prepare] random_state={CONFIG['random_state']}, test_ratio={cfg.test_ratio}")
    set_seed(CONFIG["random_state"])

    _, traj_info_df = load_all_scenes_aligned_data(CONFIG["scene_ids"])
    print(f"[Stage4:prepare] loaded trajectories={len(traj_info_df)}")

    split_df = _stratified_split_test_pool(traj_info_df, cfg)
    save_csv(split_df, os.path.join(CONFIG["interim_dir"], "stage4_curation_split.csv"))

    n_test = int((split_df["split"] == "test").sum())
    n_pool = int((split_df["split"] == "pool").sum())
    print(f"[Stage4:prepare] split done: test={n_test}, pool={n_pool}")

    pool_ids = split_df[split_df["split"] == "pool"]["global_id"].tolist()
    seeds = list(cfg.run_seeds)
    print(f"[Stage4:prepare] run stage3 pool-only repeat, seeds={seeds}")

    run_stage3_repeated_on_pool(pool_global_ids=pool_ids, seeds=seeds)
    print("[Stage4:prepare] done")

    return split_df


def _append_rows_atomic(target_path: str, new_df: pd.DataFrame):
    """追加写入 CSV，并通过临时文件 + rename 实现原子覆盖。"""
    if new_df is None or len(new_df) == 0:
        return

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path):
        old_df = pd.read_csv(target_path)
        out_df = pd.concat([old_df, new_df], axis=0, ignore_index=True)
    else:
        out_df = new_df.copy()

    tmp_path = target_path + ".tmp"
    out_df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, target_path)


def _load_completed_jobs(metrics_partial_path: str) -> Set[Tuple[str, float, int]]:
    """从部分结果中读取已完成 job 键：(strategy, ratio, run_seed)。"""
    if not os.path.exists(metrics_partial_path):
        return set()

    mdf = pd.read_csv(metrics_partial_path)
    needed = {"strategy", "ratio", "run_seed"}
    if not needed.issubset(set(mdf.columns)):
        return set()

    done = set()
    for _, r in mdf.iterrows():
        done.add((str(r["strategy"]), float(r["ratio"]), int(r["run_seed"])))
    return done


def run_stage4_curation_eval(
    smoke_test: bool = False,
    ratio_grid: Tuple[float, ...] = None,
    run_seeds: Tuple[int, ...] = None,
    random_repeat_seeds: Tuple[int, ...] = None,
):
    cfg = _build_curation_config(
        smoke_test=smoke_test,
        ratio_grid=ratio_grid,
        run_seeds=run_seeds,
        random_repeat_seeds=random_repeat_seeds,
    )
    print("[Stage4:eval] start")
    print(
        f"[Stage4:eval] smoke_test={smoke_test}, "
        f"ratio_grid={list(cfg.ratio_grid)}, run_seeds={list(cfg.run_seeds)}, random_seeds={list(cfg.random_repeat_seeds)}"
    )
    set_seed(CONFIG["random_state"])

    all_aligned, traj_info_df = load_all_scenes_aligned_data(CONFIG["scene_ids"])
    print(f"[Stage4:eval] loaded trajectories={len(traj_info_df)}")

    split_path = os.path.join(CONFIG["interim_dir"], "stage4_curation_split.csv")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"未找到 split 文件：{split_path}，请先运行 prepare 步骤。")

    split_df = pd.read_csv(split_path)

    qdf = _load_stage2_quality(CONFIG["scene_ids"])
    delta_path = CONFIG.get("stage4_stage3_delta_path", os.path.join(CONFIG["interim_dir"], "stage3_repeat_pool_only_delta_summary.csv"))
    sdf = _load_stage3_scene_delta(delta_path)

    scores_df = _build_score_table(split_df, qdf, sdf)
    save_csv(scores_df, os.path.join(CONFIG["interim_dir"], "stage4_curation_scores.csv"))

    pool_df = scores_df[scores_df["split"] == "pool"].copy()
    test_ids = scores_df[scores_df["split"] == "test"]["global_id"].tolist()
    print(f"[Stage4:eval] split loaded: pool={len(pool_df)}, test={len(test_ids)}")

    eval_std = _compute_eval_action_std(all_aligned, pool_df["global_id"].tolist())
    print("[Stage4:eval] eval action std prepared")

    # 仅跑对照 + 新增轻量双轨策略
    strategies = [
        "full", "random", "high_Q_global", "high_Q_stratified", "hybrid_Q_delta",
        "hybrid_Q08_delta02", "hybrid_accuracy08_delta02"
    ]

    if smoke_test:
        strategies = ["full", "random"]
        print(f"[Stage4:eval] smoke strategies={strategies}")
    else:
        print(f"[Stage4:eval] selected strategies count={len(strategies)} -> {strategies}")

    selection_rows = []
    metric_rows = []
    per_scene_rows = []

    partial_selection_path = os.path.join(CONFIG["interim_dir"], "stage4_curation_selection_partial.csv")
    partial_metrics_path = os.path.join(CONFIG["interim_dir"], "stage4_curation_metrics_partial.csv")
    partial_per_scene_path = os.path.join(CONFIG["interim_dir"], "stage4_curation_per_scene_metrics_partial.csv")

    completed_jobs = _load_completed_jobs(partial_metrics_path)

    all_jobs = []
    for strategy in strategies:
        ratios = (1.0,) if strategy == "full" else cfg.ratio_grid
        seeds = cfg.random_repeat_seeds if strategy == "random" else cfg.run_seeds
        for ratio in ratios:
            for run_seed in seeds:
                all_jobs.append((strategy, float(ratio), int(run_seed)))

    pending_jobs = [j for j in all_jobs if j not in completed_jobs]

    print(f"[Stage4:eval] total jobs={len(all_jobs)}, completed={len(completed_jobs)}, pending={len(pending_jobs)}")

    for job_idx, (strategy, ratio, run_seed) in enumerate(pending_jobs, start=1):
        print(f"[Stage4:eval] pending job {job_idx}/{len(pending_jobs)} strategy={strategy} ratio={ratio} seed={run_seed}")

        selected = _select_ids(pool_df, strategy, ratio, run_seed)
        train_ids = selected["global_id"].tolist()
        print(f"[Stage4:eval] selected trajectories={len(train_ids)}")

        g_m, s_df, n_windows = _train_eval_once(all_aligned, train_ids, test_ids, eval_std)
        print(
            f"[Stage4:eval] done pending job {job_idx}/{len(pending_jobs)}, "
            f"nmse={g_m['test_nmse']:.6f}, score={g_m['test_imitation_score']:.6f}, windows={n_windows}"
        )

        this_selection_rows = []
        for _, r in selected.iterrows():
            this_selection_rows.append(
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

        this_metric_rows = [
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
        ]

        this_per_scene_rows = []
        for _, srow in s_df.iterrows():
            this_per_scene_rows.append(
                {
                    "strategy": strategy,
                    "ratio": ratio,
                    "run_seed": run_seed,
                    **srow.to_dict(),
                }
            )

        # 内存聚合（本次运行）
        selection_rows.extend(this_selection_rows)
        metric_rows.extend(this_metric_rows)
        per_scene_rows.extend(this_per_scene_rows)

        # 关键：每个 job 完成后实时保存到 partial（支持中断续跑）
        _append_rows_atomic(partial_selection_path, pd.DataFrame(this_selection_rows))
        _append_rows_atomic(partial_metrics_path, pd.DataFrame(this_metric_rows))
        _append_rows_atomic(partial_per_scene_path, pd.DataFrame(this_per_scene_rows))

    # 汇总时以 partial 全量为准（含历史已完成 + 本次新增）
    selection_df = pd.read_csv(partial_selection_path) if os.path.exists(partial_selection_path) else pd.DataFrame(selection_rows)
    metrics_df = pd.read_csv(partial_metrics_path) if os.path.exists(partial_metrics_path) else pd.DataFrame(metric_rows)
    per_scene_df = pd.read_csv(partial_per_scene_path) if os.path.exists(partial_per_scene_path) else pd.DataFrame(per_scene_rows)

    random_nmse_df = metrics_df[metrics_df["strategy"] == "random"].groupby("ratio", as_index=False)["test_nmse"].mean()
    rand_mean_by_ratio = {float(r["ratio"]): float(r["test_nmse"]) for _, r in random_nmse_df.iterrows()}

    full_nmse_series = metrics_df[(metrics_df["strategy"] == "full") & (metrics_df["ratio"] == 1.0)]["test_nmse"]
    full_nmse = float(full_nmse_series.mean()) if len(full_nmse_series) > 0 else np.nan

    metrics_df["relative_to_random_nmse"] = metrics_df.apply(
        lambda x: (rand_mean_by_ratio.get(float(x["ratio"]), np.nan) - x["test_nmse"]) / rand_mean_by_ratio.get(float(x["ratio"]), np.nan)
        if pd.notna(rand_mean_by_ratio.get(float(x["ratio"]), np.nan)) else np.nan,
        axis=1,
    )
    metrics_df["relative_to_full_nmse"] = metrics_df["test_nmse"].apply(
        lambda v: (full_nmse - v) / full_nmse if (pd.notna(full_nmse) and full_nmse > 0) else np.nan
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


def run_stage4_curation(
    smoke_test: bool = False,
    ratio_grid: Tuple[float, ...] = None,
    run_seeds: Tuple[int, ...] = None,
    random_repeat_seeds: Tuple[int, ...] = None,
):
    run_stage4_prepare_pool_and_stage3(
        smoke_test=smoke_test,
        ratio_grid=ratio_grid,
        run_seeds=run_seeds,
        random_repeat_seeds=random_repeat_seeds,
    )
    return run_stage4_curation_eval(
        smoke_test=smoke_test,
        ratio_grid=ratio_grid,
        run_seeds=run_seeds,
        random_repeat_seeds=random_repeat_seeds,
    )
