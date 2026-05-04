# 文件位置：stage3_trajectory_value.py

import os

import numpy as np
import pandas as pd

from config import CONFIG
from stage3_meta_value import load_all_quality_datasets, load_stage3_delta_summary
from utils import save_csv




# 读取阶段三场景级价值预测表
# 输入：无
# 输出：预测表，或 None
def load_scene_value_prediction():
    pred_path = os.path.join(
        CONFIG["interim_dir"],
        "stage3_scene_value_prediction.csv"
    )

    # 元模型预测值是探索性结果，不作为阶段三主交付的硬依赖
    if not os.path.exists(pred_path):
        return None

    pred_df = pd.read_csv(pred_path)

    required_cols = [
        "scene_id",
        "delta_score_pred",
    ]

    missing_cols = [col for col in required_cols if col not in pred_df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"{pred_path} 缺少必要列：{missing_cols}")

    return pred_df


# 构造轨迹级效能价值特征
# 输入：
#   all_quality_df: 阶段二轨迹级质量表
#   delta_df: 阶段三经验 delta 表
#   pred_df: 阶段三元模型预测表
# 输出：
#   traj_value_df: 轨迹级效能价值特征表
#   final_output_df: 质量 + 效能价值的完整阶段三输出表
def build_trajectory_value_features(all_quality_df, delta_df, pred_df):
    value_source = "repeat_seed_mean"

    # 统一 delta 经验值字段名
    delta_keep_cols = [
        "scene_id",
        "delta_score_mean",
        "delta_normalized_mse_mean",
    ]

    optional_cols = [
        "delta_score_std",
        "delta_score_positive_rate",
        "delta_normalized_mse_std",
    ]

    for col in optional_cols:
        if col in delta_df.columns:
            delta_keep_cols.append(col)

    delta_part = delta_df[delta_keep_cols].copy()

    delta_part = delta_part.rename(columns={
        "delta_score_mean": "delta_score_emp",
        "delta_normalized_mse_mean": "delta_normalized_mse_emp",
        "delta_score_std": "delta_score_emp_std",
        "delta_score_positive_rate": "delta_score_positive_rate",
        "delta_normalized_mse_std": "delta_normalized_mse_emp_std",
    })

    # 合并经验边际效能价值
    final_output_df = pd.merge(
        all_quality_df,
        delta_part,
        on="scene_id",
        how="left",
    )

    # 元模型预测值是可选辅助字段，不作为主特征
    if pred_df is not None:
        pred_part = pred_df[["scene_id", "delta_score_pred"]].copy()

        final_output_df = pd.merge(
            final_output_df,
            pred_part,
            on="scene_id",
            how="left",
        )
    else:
        final_output_df["delta_score_pred"] = np.nan

    # 构造全局轨迹 ID，避免不同场景 trajectory_id 重名
    final_output_df["global_id"] = (
        final_output_df["scene_id"].astype(str) + "_" +
        final_output_df["trajectory_id"].astype(str)
    )

    # 说明字段：仅用于追溯，不作为建模特征
    final_output_df["stage3_empirical_value_source"] = value_source

    # 轨迹级效能价值特征表：只保留主流程真正需要的字段
    keep_cols = [
        "global_id",
        "trajectory_id",
        "scene_id",
        "delta_score_emp",
        "delta_normalized_mse_emp",
    ]

    # 多 seed 稳定性字段，建议保留用于报告
    extra_cols = [
        "delta_score_emp_std",
        "delta_score_positive_rate",
        "delta_normalized_mse_emp_std",
    ]

    for col in extra_cols:
        if col in final_output_df.columns:
            keep_cols.append(col)

    # 元模型预测值作为辅助字段保留，但阶段四主实验不使用
    if "delta_score_pred" in final_output_df.columns:
        keep_cols.append("delta_score_pred")

    keep_cols.append("stage3_empirical_value_source")

    traj_value_df = final_output_df[keep_cols].copy()

    return traj_value_df, final_output_df


# 对轨迹级效能价值表做基础检查
# 输入：traj_value_df
# 输出：检查摘要 DataFrame
def summarize_trajectory_value_features(traj_value_df):
    rows = []

    for scene_id, group in traj_value_df.groupby("scene_id"):
        row = {
            "scene_id": scene_id,
            "n_trajectories": int(len(group)),
            "delta_score_emp": float(group["delta_score_emp"].iloc[0]),
            "delta_normalized_mse_emp": float(group["delta_normalized_mse_emp"].iloc[0]),
        }

        if "delta_score_emp_std" in group.columns:
            row["delta_score_emp_std"] = float(group["delta_score_emp_std"].iloc[0])

        if "delta_score_positive_rate" in group.columns:
            row["delta_score_positive_rate"] = float(group["delta_score_positive_rate"].iloc[0])

        if "delta_score_pred" in group.columns:
            value = group["delta_score_pred"].iloc[0]
            row["delta_score_pred"] = float(value) if not pd.isna(value) else np.nan

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("delta_score_emp", ascending=False).reset_index(drop=True)

    return summary_df


# 保存阶段三 Step 4 输出
# 输入：轨迹价值表、完整输出表、摘要表
# 输出：无
def save_step4_outputs(traj_value_df, final_output_df, summary_df):
    traj_value_path = os.path.join(
        CONFIG["interim_dir"],
        "stage3_trajectory_value_features.csv"
    )

    final_output_path = os.path.join(
        CONFIG["interim_dir"],
        "stage3_final_value_outputs.csv"
    )

    summary_path = os.path.join(
        CONFIG["interim_dir"],
        "stage3_trajectory_value_summary.csv"
    )

    save_csv(traj_value_df, traj_value_path)
    save_csv(final_output_df, final_output_path)
    save_csv(summary_df, summary_path)


# 运行阶段三 Step 4
# 输入：无
# 输出：traj_value_df, final_output_df, summary_df
def run_stage3_trajectory_value():
    # 1. 读取阶段二轨迹级质量表
    all_quality_df = load_all_quality_datasets(CONFIG["scene_ids"])

    # 2. 读取阶段三经验 delta（固定使用 Step2 多 seed 稳定版）
    delta_df = load_stage3_delta_summary()

    # 3. 读取阶段三场景级预测（可选）
    pred_df = load_scene_value_prediction()

    # 4. 生成轨迹级效能价值特征
    traj_value_df, final_output_df = build_trajectory_value_features(
        all_quality_df=all_quality_df,
        delta_df=delta_df,
        pred_df=pred_df,
    )

    # 4. 生成摘要表
    summary_df = summarize_trajectory_value_features(traj_value_df)

    # 5. 保存
    save_step4_outputs(
        traj_value_df=traj_value_df,
        final_output_df=final_output_df,
        summary_df=summary_df,
    )

    return traj_value_df, final_output_df, summary_df