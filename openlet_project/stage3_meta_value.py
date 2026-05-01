# 文件位置：stage3_meta_value.py
# 说明：该模块用于阶段三探索性分析（质量特征与效能价值关系），
# 不作为阶段四主实验的强预测输入来源。

import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CONFIG
from utils import save_csv


# 读取单个场景的阶段二质量表
# 输入：scene_id，例如 S5
# 输出：该场景的质量 DataFrame
def load_one_scene_quality(scene_id):
    scene_prefix = scene_id.lower()
    quality_path = os.path.join(
        CONFIG["processed_dir"],
        f"{scene_prefix}_stage2_quality_dataset.csv"
    )

    if not os.path.exists(quality_path):
        raise FileNotFoundError(f"未找到阶段二质量文件：{quality_path}")

    df = pd.read_csv(quality_path)

    required_cols = [
        "trajectory_id",
        "scene_id",
        "Q_score",
        "Q_completeness",
        "Q_accuracy",
        "Q_diversity",
        "Q_consistency",
    ]

    # Q_usability 允许缺失：若字段不存在或恒定，不作为主流程强依赖
    optional_cols = ["Q_usability"]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(
            f"{quality_path} 缺少必要列：{missing_cols}"
        )

    return df


# 读取所有场景的阶段二质量表
# 输入：scene_ids
# 输出：合并后的轨迹级质量表
def load_all_quality_datasets(scene_ids):
    dfs = []

    for scene_id in scene_ids:
        df = load_one_scene_quality(scene_id)

        # 这里强制 scene_id 使用配置中的大写形式，避免文件内大小写不统一
        df["scene_id"] = scene_id
        dfs.append(df)

    all_quality_df = pd.concat(dfs, axis=0, ignore_index=True)
    return all_quality_df


# 将轨迹级质量特征聚合为场景级质量画像
# 输入：轨迹级质量表
# 输出：场景级质量表
def aggregate_quality_to_scene(all_quality_df):
    quality_cols = [
        "Q_score",
        "Q_completeness",
        "Q_accuracy",
        "Q_diversity",
        "Q_consistency",
    ]

    if "Q_usability" in all_quality_df.columns:
        quality_cols.append("Q_usability")

    rows = []

    for scene_id, group in all_quality_df.groupby("scene_id"):
        row = {
            "scene_id": scene_id,
            "n_trajectories": int(len(group)),
        }

        for col in quality_cols:
            values = group[col].astype(float)

            # 均值表示该场景的平均质量水平
            row[f"{col}_mean"] = float(values.mean())

            # 标准差表示该场景内部质量波动
            row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

            # 最小值表示最差轨迹质量，可反映短板效应
            row[f"{col}_min"] = float(values.min())

            # 最大值表示最好轨迹质量
            row[f"{col}_max"] = float(values.max())

        rows.append(row)

    scene_quality_df = pd.DataFrame(rows)
    scene_quality_df = scene_quality_df.sort_values("scene_id").reset_index(drop=True)

    return scene_quality_df


# 读取阶段三 Step 2 的稳定版 delta 结果
# 输入：无
# 输出：场景级 delta 表
def load_stage3_delta_summary():
    repeat_summary_path = os.path.join(
        CONFIG["interim_dir"],
        "stage3_repeat_delta_summary.csv"
    )

    if not os.path.exists(repeat_summary_path):
        raise FileNotFoundError(
            f"未找到稳定版阶段三 delta 文件：{repeat_summary_path}。"
            "请先运行 Stage3 Step2（多随机种子重复实验）。"
        )

    delta_df = pd.read_csv(repeat_summary_path)

    required_cols = [
        "scene_id",
        "delta_score_mean",
        "delta_score_std",
        "delta_score_positive_rate",
        "delta_normalized_mse_mean",
    ]

    missing_cols = [col for col in required_cols if col not in delta_df.columns]
    if len(missing_cols) > 0:
        raise ValueError(
            f"{repeat_summary_path} 缺少必要列：{missing_cols}"
        )

    return delta_df


# 构建场景级元模型数据表
# 输入：场景质量表、场景 delta 表
# 输出：合并后的元模型数据表
def build_scene_value_model_table(scene_quality_df, delta_df):
    model_df = pd.merge(
        scene_quality_df,
        delta_df,
        on="scene_id",
        how="inner",
    )

    if len(model_df) == 0:
        raise ValueError("质量特征表与 delta 表没有成功合并，请检查 scene_id。")

    model_df = model_df.sort_values("scene_id").reset_index(drop=True)
    return model_df


# 计算质量特征与 delta_score_mean 的相关关系
# 输入：元模型数据表
# 输出：相关分析表
def compute_quality_value_correlation(model_df):
    target_col = "delta_score_mean"

    # 相关分析只能使用阶段二质量聚合特征和样本量；
    # 不能使用 delta_score_min/max/std 等目标派生字段，否则会造成目标泄漏。
    feature_cols = []

    for col in model_df.columns:
        if col == "n_trajectories":
            feature_cols.append(col)
            continue

        if col.startswith("Q_") and pd.api.types.is_numeric_dtype(model_df[col]):
            values = model_df[col].astype(float).values

            # 常数列没有相关性意义，例如 Q_usability_mean 全部为1
            if np.nanstd(values) >= 1e-12:
                feature_cols.append(col)

    rows = []

    for col in feature_cols:
        x = model_df[col].astype(float).values
        y = model_df[target_col].astype(float).values

        # 如果某列没有波动，相关系数没有定义
        if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
            pearson_corr = np.nan
            pearson_p = np.nan
            spearman_corr = np.nan
            spearman_p = np.nan
        else:
            pearson_corr, pearson_p = pearsonr(x, y)
            spearman_corr, spearman_p = spearmanr(x, y)

        rows.append({
            "feature": col,
            "pearson_corr": pearson_corr,
            "pearson_p": pearson_p,
            "spearman_corr": spearman_corr,
            "spearman_p": spearman_p,
            "abs_pearson_corr": abs(pearson_corr) if not np.isnan(pearson_corr) else np.nan,
            "abs_spearman_corr": abs(spearman_corr) if not np.isnan(spearman_corr) else np.nan,
        })

    corr_df = pd.DataFrame(rows)

    corr_df = corr_df.sort_values(
        ["abs_spearman_corr", "abs_pearson_corr"],
        ascending=False,
    ).reset_index(drop=True)

    return corr_df


# 使用 Ridge 回归做留一场景验证
# 输入：元模型数据表
# 输出：loocv预测表、整体指标表、最终模型
def run_ridge_loocv(model_df):
    target_col = "delta_score_mean"

    # 小样本 n=5 下，主元模型只使用均值型质量特征 + 轨迹数；
    # std/min/max 仅用于解释性相关分析，不进入 Ridge 主模型。
    feature_cols = [
        "n_trajectories",
        "Q_score_mean",
        "Q_completeness_mean",
        "Q_accuracy_mean",
        "Q_diversity_mean",
        "Q_consistency_mean",
    ]

    if "Q_usability_mean" in model_df.columns:
        usability_std = np.nanstd(model_df["Q_usability_mean"].astype(float).values)
        if usability_std >= 1e-12:
            feature_cols.append("Q_usability_mean")

    missing_cols = [col for col in feature_cols if col not in model_df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"元模型缺少必要特征列：{missing_cols}")

    X = model_df[feature_cols].astype(float).values
    y = model_df[target_col].astype(float).values
    scene_ids = model_df["scene_id"].values

    loo = LeaveOneOut()

    rows = []

    # alpha 越大，正则越强；小样本下用适中的 alpha 更稳
    alpha = 1.0

    for train_idx, test_idx in loo.split(X):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rows.append({
            "scene_id": scene_ids[test_idx][0],
            "delta_score_true": float(y_test[0]),
            "delta_score_pred": float(y_pred[0]),
            "abs_error": float(abs(y_test[0] - y_pred[0])),
        })

    loocv_df = pd.DataFrame(rows)

    y_true = loocv_df["delta_score_true"].values
    y_pred = loocv_df["delta_score_pred"].values

    metrics = {
        "n_scenes": int(len(model_df)),
        "model": "Ridge_LOOCV",
        "alpha": alpha,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

    # n=5 时 R2 极不稳定，仅作参考
    try:
        metrics["r2"] = float(r2_score(y_true, y_pred))
    except Exception:
        metrics["r2"] = np.nan

    metrics_df = pd.DataFrame([metrics])

    # 最后在全部 5 个场景上训练一个最终模型，用于生成场景预测值
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    final_model.fit(X, y)

    return loocv_df, metrics_df, final_model, feature_cols


# 用最终 Ridge 元模型给场景生成预测效能价值
# 输入：元模型数据表、最终模型、特征列
# 输出：场景预测表
def predict_scene_value(model_df, final_model, feature_cols):
    X = model_df[feature_cols].astype(float).values
    pred = final_model.predict(X)

    pred_df = model_df[[
        "scene_id",
        "delta_score_mean",
        "delta_normalized_mse_mean",
    ]].copy()

    pred_df["delta_score_pred"] = pred
    pred_df["prediction_error_in_sample"] = (
        pred_df["delta_score_pred"] - pred_df["delta_score_mean"]
    )

    return pred_df


# 保存阶段三 Step 3 输出
# 输入：各类结果表
# 输出：保存到 data/interim
def save_step3_outputs(scene_quality_df, model_df, corr_df, loocv_df, metrics_df, pred_df):
    save_csv(
        scene_quality_df,
        os.path.join(CONFIG["interim_dir"], "stage3_scene_quality_table.csv")
    )

    save_csv(
        model_df,
        os.path.join(CONFIG["interim_dir"], "stage3_scene_value_model_table.csv")
    )

    save_csv(
        corr_df,
        os.path.join(CONFIG["interim_dir"], "stage3_quality_value_correlation.csv")
    )

    save_csv(
        loocv_df,
        os.path.join(CONFIG["interim_dir"], "stage3_meta_model_loocv.csv")
    )

    save_csv(
        metrics_df,
        os.path.join(CONFIG["interim_dir"], "stage3_meta_model_metrics.csv")
    )

    save_csv(
        pred_df,
        os.path.join(CONFIG["interim_dir"], "stage3_scene_value_prediction.csv")
    )


# 运行阶段三 Step 3
# 输入：无
# 输出：scene_quality_df, model_df, corr_df, loocv_df, metrics_df, pred_df
def run_stage3_meta_value():
    # 1. 读取阶段二轨迹级质量结果
    all_quality_df = load_all_quality_datasets(CONFIG["scene_ids"])

    # 2. 聚合为场景级质量画像
    scene_quality_df = aggregate_quality_to_scene(all_quality_df)

    # 3. 读取阶段三经验边际价值
    delta_df = load_stage3_delta_summary()

    # 4. 构建元模型数据表
    model_df = build_scene_value_model_table(scene_quality_df, delta_df)

    # 5. 相关分析
    corr_df = compute_quality_value_correlation(model_df)

    # 6. Ridge 留一场景验证
    loocv_df, metrics_df, final_model, feature_cols = run_ridge_loocv(model_df)

    # 7. 生成场景级预测价值
    pred_df = predict_scene_value(model_df, final_model, feature_cols)

    # 8. 保存
    save_step3_outputs(
        scene_quality_df=scene_quality_df,
        model_df=model_df,
        corr_df=corr_df,
        loocv_df=loocv_df,
        metrics_df=metrics_df,
        pred_df=pred_df,
    )

    return scene_quality_df, model_df, corr_df, loocv_df, metrics_df, pred_df