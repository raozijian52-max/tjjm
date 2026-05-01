# 文件位置：stage4_modeling_table.py

import os
import pandas as pd

from config_template import CONFIG
from utils import save_csv, save_json


# 构造全局轨迹 ID
# 输入：DataFrame，要求包含 scene_id 和 trajectory_id
# 输出：增加 global_id 的 DataFrame
def add_global_id(df):
    df = df.copy()
    df["global_id"] = df["scene_id"].astype(str) + "_" + df["trajectory_id"].astype(str)
    return df


# 读取并合并 S1-S5 阶段一特征
# 输入：无
# 输出：stage1_df，包含 global_id、trajectory_id、scene_id 和 stage1_ 前缀特征
def load_stage1_features():
    dfs = []

    for scene_id in CONFIG["scene_ids"]:
        scene_prefix = scene_id.lower()

        # 阶段一特征改为从 processed 读取 feature_matrix
        path = os.path.join(CONFIG["processed_dir"], f"{scene_prefix}_feature_matrix.csv")

        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到阶段一 feature_matrix 文件：{path}")

        df = pd.read_csv(path)

        # feature_matrix 必须至少包含 trajectory_id
        if "trajectory_id" not in df.columns:
            raise ValueError(f"{path} 缺少 trajectory_id 列，无法和其他阶段合并。")

        # 如果 feature_matrix 内没有 scene_id，就按当前循环场景补上
        df["scene_id"] = scene_id

        df = add_global_id(df)

        dfs.append(df)

    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    # 这些列不是阶段一建模特征
    non_feature_cols = {
        "global_id",
        "trajectory_id",
        "scene_id",
        "task_name",
        "object_id",
        "auto_label",
        "manual_label",
        "final_label",
        "review_needed",
        "label_reason",
    }

    # 只保留数值型阶段一特征
    feature_cols = []
    rename_map = {}

    for col in all_df.columns:
        if col in non_feature_cols:
            continue

        if pd.api.types.is_numeric_dtype(all_df[col]):
            new_col = col if col.startswith("stage1_") else f"stage1_{col}"
            rename_map[col] = new_col
            feature_cols.append(new_col)

    keep_cols = ["global_id", "trajectory_id", "scene_id"] + list(rename_map.keys())
    stage1_df = all_df[keep_cols].rename(columns=rename_map)

    return stage1_df, feature_cols


# 读取并合并 S1-S5 阶段二质量特征
# 输入：无
# 输出：stage2_df，quality_cols
def load_stage2_quality():
    dfs = []

    for scene_id in CONFIG["scene_ids"]:
        scene_prefix = scene_id.lower()

        # 阶段二质量表改为从 processed 读取
        path = os.path.join(CONFIG["processed_dir"], f"{scene_prefix}_stage2_quality_dataset.csv")

        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到阶段二质量文件：{path}")

        df = pd.read_csv(path)

        if "trajectory_id" not in df.columns:
            raise ValueError(f"{path} 缺少 trajectory_id 列，无法和其他阶段合并。")

        # 强制使用配置中的 scene_id，避免文件内部大小写或命名不一致
        df["scene_id"] = scene_id

        df = add_global_id(df)

        dfs.append(df)

    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    base_quality_cols = [
        "Q_score",
        "Q_completeness",
        "Q_accuracy",
        "Q_diversity",
        "Q_consistency",
        "Q_usability",
    ]

    missing_cols = [c for c in base_quality_cols if c not in all_df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"阶段二质量表缺少字段：{missing_cols}")

    # 如果 Q_usability 是常数，则不进入建模特征
    quality_cols = []
    for col in base_quality_cols:
        if all_df[col].nunique(dropna=True) <= 1:
            continue
        quality_cols.append(col)

    keep_cols = ["global_id", "trajectory_id", "scene_id"] + quality_cols
    stage2_df = all_df[keep_cols].copy()

    return stage2_df, quality_cols


# 读取阶段三轨迹级效能价值特征
# 输入：无
# 输出：stage3_df，stage3_cols
def load_stage3_value_features():
    path = os.path.join(CONFIG["interim_dir"], "stage3_trajectory_value_features.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到阶段三轨迹级价值文件：{path}")

    df = pd.read_csv(path)

    required_cols = [
        "global_id",
        "trajectory_id",
        "scene_id",
        "delta_score_emp",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"阶段三轨迹级价值表缺少字段：{missing_cols}")

    # 主实验只使用多 seed 留一场景得到的经验边际效能价值；
    # 不使用 delta_score_pred，因为当前只有5个场景，元模型预测泛化不稳定。
    stage3_cols = ["delta_score_emp"]

    keep_cols = ["global_id", "trajectory_id", "scene_id"] + stage3_cols
    stage3_df = df[keep_cols].copy()

    return stage3_df, stage3_cols


# 读取阶段四轨迹级 BC 标签
# 输入：无
# 输出：label_df
def load_stage4_labels():
    path = os.path.join(CONFIG["interim_dir"], "stage4_bc_trajectory_labels.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到阶段四标签文件：{path}")

    df = pd.read_csv(path)

    required_cols = [
        "global_id",
        "trajectory_id",
        "scene_id",
        "trajectory_mse",
        "trajectory_mae",
        "trajectory_normalized_mse",
        "trajectory_imitation_score",
        "n_bc_windows",
        "fold",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"阶段四标签表缺少字段：{missing_cols}")

    keep_cols = required_cols
    label_df = df[keep_cols].copy()

    return label_df


# 合并阶段一、阶段二、阶段三和阶段四标签
# 输入：各阶段表
# 输出：建模主表
def build_master_table(stage1_df, stage2_df, stage3_df, label_df):
    master_df = label_df.copy()

    # 先合并阶段一特征
    master_df = pd.merge(
        master_df,
        stage1_df.drop(columns=["trajectory_id", "scene_id"]),
        on="global_id",
        how="left",
    )

    # 再合并阶段二质量
    master_df = pd.merge(
        master_df,
        stage2_df.drop(columns=["trajectory_id", "scene_id"]),
        on="global_id",
        how="left",
    )

    # 最后合并阶段三效能价值
    master_df = pd.merge(
        master_df,
        stage3_df.drop(columns=["trajectory_id", "scene_id"]),
        on="global_id",
        how="left",
    )

    return master_df


# 检查主表是否存在缺失和重复
# 输入：master_df、特征列列表、目标列
# 输出：检查信息字典
def check_master_table(master_df, feature_cols, target_col):
    info = {}

    info["n_rows"] = int(len(master_df))
    info["n_unique_global_id"] = int(master_df["global_id"].nunique())
    info["has_duplicate_global_id"] = bool(master_df["global_id"].duplicated().any())
    info["target_missing_count"] = int(master_df[target_col].isna().sum())

    missing_feature_cols = []
    for col in feature_cols:
        if col not in master_df.columns:
            missing_feature_cols.append(col)

    info["missing_feature_cols"] = missing_feature_cols

    # 统计有缺失值的特征列数量，不在这里做填补，模型阶段统一处理
    feature_na_counts = master_df[feature_cols].isna().sum()
    info["n_feature_cols"] = int(len(feature_cols))
    info["n_feature_cols_with_na"] = int((feature_na_counts > 0).sum())

    return info


# 构建 Config A/B/C 特征列配置
# 输入：stage1_cols、quality_cols、stage3_cols
# 输出：配置字典
def build_feature_config(stage1_cols, quality_cols, stage3_cols):
    config = {
        "target_col": "trajectory_normalized_mse",
        "aux_target_col": "trajectory_imitation_score",

        # Config A：只用阶段一原始轨迹特征
        "config_A": stage1_cols,

        # Config B：阶段一 + 阶段二质量特征
        "config_B": stage1_cols + quality_cols,

        # Config C：阶段一 + 阶段二 + 阶段三效能价值
        "config_C": stage1_cols + quality_cols + stage3_cols,

        "stage1_cols": stage1_cols,
        "quality_cols": quality_cols,
        "stage3_cols": stage3_cols,
    }

    return config


# 保存阶段四 Step 2 输出
# 输入：master_df、feature_config、check_info
# 输出：无
def save_step2_outputs(master_df, feature_config, check_info):
    master_path = os.path.join(CONFIG["interim_dir"], "stage4_modeling_master_table.csv")
    config_path = os.path.join(CONFIG["interim_dir"], "stage4_feature_config.json")
    check_path = os.path.join(CONFIG["interim_dir"], "stage4_master_check.json")

    save_csv(master_df, master_path)
    save_json(feature_config, config_path)
    save_json(check_info, check_path)


# 运行阶段四 Step 2
# 输入：无
# 输出：master_df、feature_config、check_info
def run_stage4_build_master_table():
    stage1_df, stage1_cols = load_stage1_features()
    stage2_df, quality_cols = load_stage2_quality()
    stage3_df, stage3_cols = load_stage3_value_features()
    label_df = load_stage4_labels()

    master_df = build_master_table(
        stage1_df=stage1_df,
        stage2_df=stage2_df,
        stage3_df=stage3_df,
        label_df=label_df,
    )

    feature_config = build_feature_config(
        stage1_cols=stage1_cols,
        quality_cols=quality_cols,
        stage3_cols=stage3_cols,
    )

    all_feature_cols = feature_config["config_C"]
    check_info = check_master_table(
        master_df=master_df,
        feature_cols=all_feature_cols,
        target_col=feature_config["target_col"],
    )

    save_step2_outputs(
        master_df=master_df,
        feature_config=feature_config,
        check_info=check_info,
    )

    return master_df, feature_config, check_info