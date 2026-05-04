# 文件位置：run_stage4_baseline_check.py

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from config import CONFIG, ensure_dirs
from utils import save_csv


# 读取阶段四建模主表和目标列配置
# 输入：无
# 输出：master_df, target_col
def load_stage4_master():
    master_path = os.path.join(CONFIG["interim_dir"], "stage4_modeling_master_table.csv")
    config_path = os.path.join(CONFIG["interim_dir"], "stage4_feature_config.json")

    if not os.path.exists(master_path):
        raise FileNotFoundError(f"未找到建模主表：{master_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到特征配置：{config_path}")

    master_df = pd.read_csv(master_path)

    with open(config_path, "r", encoding="utf-8") as f:
        feature_config = json.load(f)

    target_col = feature_config["target_col"]

    return master_df, target_col


# 构造随机轨迹级 KFold 划分
# 输入：master_df
# 输出：splits 列表
def make_random_kfold_splits(master_df):
    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=CONFIG["random_state"],
    )

    splits = []

    for fold_id, (train_idx, test_idx) in enumerate(kf.split(master_df)):
        splits.append({
            "protocol": "random_kfold",
            "fold": f"fold_{fold_id}",
            "test_scene": "mixed",
            "train_idx": train_idx,
            "test_idx": test_idx,
        })

    return splits


# 构造留一场景划分
# 输入：master_df
# 输出：splits 列表
def make_leave_one_scene_splits(master_df):
    splits = []

    for scene_id in sorted(master_df["scene_id"].unique()):
        test_mask = master_df["scene_id"] == scene_id
        train_mask = ~test_mask

        splits.append({
            "protocol": "leave_one_scene",
            "fold": f"leave_{scene_id}",
            "test_scene": scene_id,
            "train_idx": np.where(train_mask.values)[0],
            "test_idx": np.where(test_mask.values)[0],
        })

    return splits


# 计算回归指标
# 输入：真实值、预测值
# 输出：指标字典
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # 均值预测器在每个 fold 内预测常数，相关系数通常没有定义
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        spearman_corr = np.nan
        pearson_corr = np.nan
    else:
        spearman_corr = float(spearmanr(y_true, y_pred).correlation)
        pearson_corr = float(pearsonr(y_true, y_pred)[0])

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman_corr": spearman_corr,
        "pearson_corr": pearson_corr,
    }


# 运行训练集均值 baseline
# 输入：master_df、target_col、splits
# 输出：summary_df, pred_df
def run_mean_baseline(master_df, target_col, splits):
    metric_rows = []
    pred_rows = []

    for split in splits:
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]

        train_df = master_df.iloc[train_idx]
        test_df = master_df.iloc[test_idx]

        y_train = train_df[target_col].values
        y_test = test_df[target_col].values

        train_mean = float(np.mean(y_train))
        y_pred = np.full_like(y_test, fill_value=train_mean, dtype=float)

        metrics = compute_metrics(y_test, y_pred)
        metrics.update({
            "protocol": split["protocol"],
            "config": "baseline",
            "model": "train_mean",
            "fold": split["fold"],
            "test_scene": split["test_scene"],
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "train_mean": train_mean,
        })
        metric_rows.append(metrics)

        for i, row_idx in enumerate(test_idx):
            pred_rows.append({
                "global_id": master_df.iloc[row_idx]["global_id"],
                "trajectory_id": master_df.iloc[row_idx]["trajectory_id"],
                "scene_id": master_df.iloc[row_idx]["scene_id"],
                "protocol": split["protocol"],
                "config": "baseline",
                "model": "train_mean",
                "fold": split["fold"],
                "test_scene": split["test_scene"],
                "y_true": float(y_test[i]),
                "y_pred": float(y_pred[i]),
                "abs_error": float(abs(y_test[i] - y_pred[i])),
            })

    fold_metrics_df = pd.DataFrame(metric_rows)
    pred_df = pd.DataFrame(pred_rows)

    summary = {
        "protocol": splits[0]["protocol"],
        "config": "baseline",
        "model": "train_mean",
        "n_samples": int(len(master_df)),
        "n_features_original": 0,
        "n_features_used_mean": 0,
        "mae_mean": float(fold_metrics_df["mae"].mean()),
        "mae_std": float(fold_metrics_df["mae"].std(ddof=1)),
        "rmse_mean": float(fold_metrics_df["rmse"].mean()),
        "rmse_std": float(fold_metrics_df["rmse"].std(ddof=1)),
        "r2_mean": float(fold_metrics_df["r2"].mean()),
        "r2_std": float(fold_metrics_df["r2"].std(ddof=1)),
        "spearman_mean": float(fold_metrics_df["spearman_corr"].mean(skipna=True)),
        "spearman_std": float(fold_metrics_df["spearman_corr"].std(ddof=1, skipna=True)),
        "pearson_mean": float(fold_metrics_df["pearson_corr"].mean(skipna=True)),
        "pearson_std": float(fold_metrics_df["pearson_corr"].std(ddof=1, skipna=True)),
    }

    summary_df = pd.DataFrame([summary])

    return summary_df, pred_df


# 合并 baseline 与已有模型结果，便于直接对比
# 输入：baseline_summary_df
# 输出：combined_df
def build_combined_comparison(baseline_summary_df):
    model_result_path = os.path.join(CONFIG["interim_dir"], "stage4_config_comparison.csv")

    if not os.path.exists(model_result_path):
        return baseline_summary_df.copy()

    model_df = pd.read_csv(model_result_path)

    combined_df = pd.concat(
        [baseline_summary_df, model_df],
        axis=0,
        ignore_index=True,
    )

    combined_df = combined_df.sort_values(
        ["protocol", "rmse_mean", "mae_mean"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return combined_df


# 保存 baseline 检查结果
# 输入：baseline_summary_df, baseline_pred_df, combined_df
# 输出：无
def save_outputs(baseline_summary_df, baseline_pred_df, combined_df):
    save_csv(
        baseline_summary_df,
        os.path.join(CONFIG["interim_dir"], "stage4_baseline_comparison.csv")
    )

    save_csv(
        baseline_pred_df,
        os.path.join(CONFIG["interim_dir"], "stage4_baseline_predictions.csv")
    )

    save_csv(
        combined_df,
        os.path.join(CONFIG["interim_dir"], "stage4_comparison_with_baseline.csv")
    )


# 主程序入口
# 作用：运行阶段四均值预测 baseline 检查
def main():
    ensure_dirs()

    master_df, target_col = load_stage4_master()

    random_splits = make_random_kfold_splits(master_df)
    loso_splits = make_leave_one_scene_splits(master_df)

    random_summary_df, random_pred_df = run_mean_baseline(
        master_df=master_df,
        target_col=target_col,
        splits=random_splits,
    )

    loso_summary_df, loso_pred_df = run_mean_baseline(
        master_df=master_df,
        target_col=target_col,
        splits=loso_splits,
    )

    baseline_summary_df = pd.concat(
        [random_summary_df, loso_summary_df],
        axis=0,
        ignore_index=True,
    )

    baseline_pred_df = pd.concat(
        [random_pred_df, loso_pred_df],
        axis=0,
        ignore_index=True,
    )

    combined_df = build_combined_comparison(baseline_summary_df)

    save_outputs(
        baseline_summary_df=baseline_summary_df,
        baseline_pred_df=baseline_pred_df,
        combined_df=combined_df,
    )

    print("=" * 60)
    print("阶段四补充完成：均值预测 baseline 检查")
    print("=" * 60)

    print("\n[1] Baseline 结果")
    show_cols = [
        "protocol",
        "config",
        "model",
        "mae_mean",
        "rmse_mean",
        "r2_mean",
        "spearman_mean",
        "pearson_mean",
    ]
    print(baseline_summary_df[show_cols])

    print("\n[2] Baseline + 模型结果对比")
    print(combined_df[show_cols])

    print("\n结果已保存到 data/interim：")
    print("- stage4_baseline_comparison.csv")
    print("- stage4_baseline_predictions.csv")
    print("- stage4_comparison_with_baseline.csv")


if __name__ == "__main__":
    main()