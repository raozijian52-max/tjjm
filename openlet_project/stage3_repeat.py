# 文件位置：stage3_repeat.py

import os

import numpy as np
import pandas as pd

from config import CONFIG
from stage3_bc_value import run_stage3_bc_value
from utils import save_csv


# 重复运行阶段三 BC 留一场景实验
# 输入：seeds 随机种子列表
# 输出：base_all_df, leave_all_df, delta_all_df, delta_summary_df
def run_stage3_repeated(seeds=None):
    # 默认先用 3 个随机种子，计算成本可控
    if seeds is None:
        seeds = [42, 2024, 2025]

    original_seed = CONFIG["random_state"]

    base_rows = []
    leave_rows = []
    delta_rows = []

    for seed in seeds:
        print("=" * 60)
        print(f"开始重复实验：seed = {seed}")
        print("=" * 60)

        # 修改全局随机种子，复用已有阶段三主流程
        CONFIG["random_state"] = int(seed)

        base_metrics_df, leave_one_metrics_df, delta_df = run_stage3_bc_value()

        # 增加 seed 字段，方便后续聚合
        base_metrics_df = base_metrics_df.copy()
        leave_one_metrics_df = leave_one_metrics_df.copy()
        delta_df = delta_df.copy()

        base_metrics_df["seed"] = seed
        leave_one_metrics_df["seed"] = seed
        delta_df["seed"] = seed

        base_rows.append(base_metrics_df)
        leave_rows.append(leave_one_metrics_df)
        delta_rows.append(delta_df)

    # 恢复原始 seed，避免影响后续其他脚本
    CONFIG["random_state"] = original_seed

    base_all_df = pd.concat(base_rows, axis=0, ignore_index=True)
    leave_all_df = pd.concat(leave_rows, axis=0, ignore_index=True)
    delta_all_df = pd.concat(delta_rows, axis=0, ignore_index=True)

    delta_summary_df = summarize_delta_stability(delta_all_df)

    save_stage3_repeat_outputs(
        base_all_df=base_all_df,
        leave_all_df=leave_all_df,
        delta_all_df=delta_all_df,
        delta_summary_df=delta_summary_df,
    )

    return base_all_df, leave_all_df, delta_all_df, delta_summary_df


# 汇总每个场景的 delta 稳定性
# 输入：所有 seed 的 delta 结果
# 输出：每个场景一行的稳定性摘要
def summarize_delta_stability(delta_all_df):
    summary_rows = []

    for scene_id, group in delta_all_df.groupby("scene_id"):
        delta_score = group["delta_score"].astype(float)
        delta_mse = group["delta_normalized_mse"].astype(float)

        row = {
            "scene_id": scene_id,
            "n_runs": int(group["seed"].nunique()),

            # delta_score 越大，说明场景边际价值越高
            "delta_score_mean": float(delta_score.mean()),
            "delta_score_std": float(delta_score.std(ddof=1)) if len(delta_score) > 1 else 0.0,
            "delta_score_min": float(delta_score.min()),
            "delta_score_max": float(delta_score.max()),

            # 正向比例：多少次实验中 delta_score > 0
            "delta_score_positive_rate": float(np.mean(delta_score > 0)),

            # delta_normalized_mse 越大，说明移除该场景后误差上升越明显
            "delta_normalized_mse_mean": float(delta_mse.mean()),
            "delta_normalized_mse_std": float(delta_mse.std(ddof=1)) if len(delta_mse) > 1 else 0.0,
            "delta_normalized_mse_min": float(delta_mse.min()),
            "delta_normalized_mse_max": float(delta_mse.max()),
        }

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # 按平均边际价值排序，数值越大排名越靠前
    summary_df["rank_by_delta_score_mean"] = (
        summary_df["delta_score_mean"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    summary_df = summary_df.sort_values(
        ["rank_by_delta_score_mean", "scene_id"]
    ).reset_index(drop=True)

    return summary_df


# 保存重复实验结果
# 输入：四张结果表
# 输出：保存到 data/interim
def save_stage3_repeat_outputs(base_all_df, leave_all_df, delta_all_df, delta_summary_df):
    base_path = os.path.join(CONFIG["interim_dir"], "stage3_repeat_base_all.csv")
    leave_path = os.path.join(CONFIG["interim_dir"], "stage3_repeat_leave_all.csv")
    delta_path = os.path.join(CONFIG["interim_dir"], "stage3_repeat_delta_all.csv")
    summary_path = os.path.join(CONFIG["interim_dir"], "stage3_repeat_delta_summary.csv")

    save_csv(base_all_df, base_path)
    save_csv(leave_all_df, leave_path)
    save_csv(delta_all_df, delta_path)
    save_csv(delta_summary_df, summary_path)