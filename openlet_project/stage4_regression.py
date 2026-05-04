# 文件位置：stage4_regression.py

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CONFIG
from utils import save_csv


# 读取阶段四建模主表和特征配置
# 输入：无
# 输出：master_df, feature_config
def load_stage4_inputs():
    master_path = os.path.join(CONFIG["interim_dir"], "stage4_modeling_master_table.csv")
    config_path = os.path.join(CONFIG["interim_dir"], "stage4_feature_config.json")

    if not os.path.exists(master_path):
        raise FileNotFoundError(f"未找到建模主表：{master_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到特征配置：{config_path}")

    master_df = pd.read_csv(master_path)

    with open(config_path, "r", encoding="utf-8") as f:
        feature_config = json.load(f)

    return master_df, feature_config


# 基于训练集删除常数和准常数特征
# 输入：训练集 DataFrame、候选特征列、dominant_ratio 阈值
# 输出：保留特征列、删除特征记录
def filter_low_variance_features(train_df, feature_cols, threshold=0.995):
    kept_cols = []
    dropped_rows = []

    for col in feature_cols:
        values = train_df[col]

        # 全空列删除
        if values.isna().all():
            dropped_rows.append({
                "feature": col,
                "reason": "all_missing",
                "dominant_ratio": np.nan,
                "std": np.nan,
            })
            continue

        # 常数列删除
        nunique = values.nunique(dropna=True)
        std = float(values.std(skipna=True)) if pd.api.types.is_numeric_dtype(values) else np.nan

        if nunique <= 1:
            dominant_ratio = float(values.value_counts(dropna=False, normalize=True).max())
            dropped_rows.append({
                "feature": col,
                "reason": "constant",
                "dominant_ratio": dominant_ratio,
                "std": std,
            })
            continue

        # 准常数列删除：某一个取值占比过高
        dominant_ratio = float(values.value_counts(dropna=False, normalize=True).max())

        if dominant_ratio >= threshold:
            dropped_rows.append({
                "feature": col,
                "reason": "quasi_constant",
                "dominant_ratio": dominant_ratio,
                "std": std,
            })
            continue

        # 极小标准差列删除，避免数值上近似常数
        if pd.notna(std) and std < 1e-12:
            dropped_rows.append({
                "feature": col,
                "reason": "near_zero_std",
                "dominant_ratio": dominant_ratio,
                "std": std,
            })
            continue

        kept_cols.append(col)

    return kept_cols, dropped_rows


# 构建模型
# 输入：模型名称 ridge / xgboost
# 输出：sklearn Pipeline；如果 xgboost 未安装则返回 None
def build_model(model_name):
    if model_name == "ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])

    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            return None

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=CONFIG["random_state"],
                n_jobs=-1,
            )),
        ])

    raise ValueError(f"未知模型名称：{model_name}")


# 计算回归指标
# 输入：真实值、预测值
# 输出：指标字典
def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

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


# 提取特征重要性
# 输入：训练好的 Pipeline、实际保留特征列、配置名、模型名
# 输出：特征重要性 DataFrame
def extract_feature_importance(model, feature_cols, config_name, model_name):
    inner_model = model.named_steps["model"]

    if hasattr(inner_model, "feature_importances_"):
        importance = inner_model.feature_importances_
        importance_type = "tree_importance"
    elif hasattr(inner_model, "coef_"):
        coef = inner_model.coef_
        importance = np.abs(coef).reshape(-1)
        importance_type = "abs_coef"
    else:
        return pd.DataFrame()

    rows = []

    for feature, value in zip(feature_cols, importance):
        rows.append({
            "config": config_name,
            "model": model_name,
            "feature": feature,
            "importance": float(value),
            "importance_type": importance_type,
        })

    importance_df = pd.DataFrame(rows)

    if len(importance_df) > 0:
        importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df


# 构造随机 KFold 划分
# 输入：master_df
# 输出：fold 列表，每个元素包含 train_idx, test_idx, fold_name
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
            "train_idx": train_idx,
            "test_idx": test_idx,
            "test_scene": "mixed",
        })

    return splits


# 构造留一场景划分
# 输入：master_df
# 输出：fold 列表，每个元素包含 train_idx, test_idx, fold_name
def make_leave_one_scene_splits(master_df):
    splits = []

    for scene_id in sorted(master_df["scene_id"].unique()):
        test_mask = master_df["scene_id"] == scene_id
        train_mask = ~test_mask

        train_idx = np.where(train_mask.values)[0]
        test_idx = np.where(test_mask.values)[0]

        splits.append({
            "protocol": "leave_one_scene",
            "fold": f"leave_{scene_id}",
            "train_idx": train_idx,
            "test_idx": test_idx,
            "test_scene": scene_id,
        })

    return splits


# 对一个 config + model + protocol 运行交叉验证
# 输入：主表、特征列、目标列、配置名、模型名、划分列表
# 输出：summary_df, prediction_df, dropped_df
def run_cv_for_config_model(master_df, feature_cols, target_col, config_name, model_name, splits):
    pred_rows = []
    fold_metric_rows = []
    dropped_rows_all = []

    for split in splits:
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]

        train_df = master_df.iloc[train_idx].copy()
        test_df = master_df.iloc[test_idx].copy()

        # 仅基于训练集做低方差过滤，避免测试集信息泄漏
        kept_cols, dropped_rows = filter_low_variance_features(
            train_df=train_df,
            feature_cols=feature_cols,
            threshold=0.995,
        )

        for row in dropped_rows:
            row["config"] = config_name
            row["model"] = model_name
            row["protocol"] = split["protocol"]
            row["fold"] = split["fold"]
            dropped_rows_all.append(row)

        if len(kept_cols) == 0:
            raise ValueError(f"{config_name}-{model_name}-{split['fold']} 没有可用特征。")

        model = build_model(model_name)

        if model is None:
            return None, None, None

        X_train = train_df[kept_cols].values
        y_train = train_df[target_col].values

        X_test = test_df[kept_cols].values
        y_test = test_df[target_col].values

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_regression_metrics(y_test, y_pred)
        metrics.update({
            "protocol": split["protocol"],
            "config": config_name,
            "model": model_name,
            "fold": split["fold"],
            "test_scene": split["test_scene"],
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_features_original": int(len(feature_cols)),
            "n_features_used": int(len(kept_cols)),
            "n_features_dropped": int(len(feature_cols) - len(kept_cols)),
        })
        fold_metric_rows.append(metrics)

        for i, row_idx in enumerate(test_idx):
            pred_rows.append({
                "global_id": master_df.iloc[row_idx]["global_id"],
                "trajectory_id": master_df.iloc[row_idx]["trajectory_id"],
                "scene_id": master_df.iloc[row_idx]["scene_id"],
                "protocol": split["protocol"],
                "config": config_name,
                "model": model_name,
                "fold": split["fold"],
                "test_scene": split["test_scene"],
                "y_true": float(y_test[i]),
                "y_pred": float(y_pred[i]),
                "abs_error": float(abs(y_test[i] - y_pred[i])),
            })

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    pred_df = pd.DataFrame(pred_rows)
    dropped_df = pd.DataFrame(dropped_rows_all)

    summary = {
        "protocol": splits[0]["protocol"],
        "config": config_name,
        "model": model_name,
        "n_samples": int(len(master_df)),
        "n_features_original": int(len(feature_cols)),
        "n_features_used_mean": float(fold_metrics_df["n_features_used"].mean()),
        "mae_mean": float(fold_metrics_df["mae"].mean()),
        "mae_std": float(fold_metrics_df["mae"].std(ddof=1)),
        "rmse_mean": float(fold_metrics_df["rmse"].mean()),
        "rmse_std": float(fold_metrics_df["rmse"].std(ddof=1)),
        "r2_mean": float(fold_metrics_df["r2"].mean()),
        "r2_std": float(fold_metrics_df["r2"].std(ddof=1)),
        "spearman_mean": float(fold_metrics_df["spearman_corr"].mean()),
        "spearman_std": float(fold_metrics_df["spearman_corr"].std(ddof=1)),
        "pearson_mean": float(fold_metrics_df["pearson_corr"].mean()),
        "pearson_std": float(fold_metrics_df["pearson_corr"].std(ddof=1)),
    }

    summary_df = pd.DataFrame([summary])

    return summary_df, pred_df, dropped_df


# 在全量数据上训练最终模型，提取特征重要性
# 输入：主表、特征列、目标列、配置名、模型名
# 输出：importance_df
def fit_full_model_importance(master_df, feature_cols, target_col, config_name, model_name):
    kept_cols, _ = filter_low_variance_features(
        train_df=master_df,
        feature_cols=feature_cols,
        threshold=0.995,
    )

    model = build_model(model_name)

    if model is None:
        return pd.DataFrame()

    X = master_df[kept_cols].values
    y = master_df[target_col].values

    model.fit(X, y)

    importance_df = extract_feature_importance(
        model=model,
        feature_cols=kept_cols,
        config_name=config_name,
        model_name=model_name,
    )

    return importance_df


# 运行阶段四回归实验
# 输入：无
# 输出：comparison_df, predictions_df, importance_df, dropped_df
def run_stage4_regression():
    master_df, feature_config = load_stage4_inputs()

    target_col = feature_config["target_col"]

    configs = {
        "config_A": feature_config["config_A"],
        "config_B": feature_config["config_B"],
        "config_C": feature_config["config_C"],
    }

    model_names = ["ridge", "xgboost"]

    protocols = {
        "random_kfold": make_random_kfold_splits(master_df),
        "leave_one_scene": make_leave_one_scene_splits(master_df),
    }

    comparison_list = []
    prediction_list = []
    dropped_list = []
    importance_list = []

    for protocol_name, splits in protocols.items():
        print(f"验证方式：{protocol_name}")

        for config_name, feature_cols in configs.items():
            print(f"  配置：{config_name}，原始特征数={len(feature_cols)}")

            for model_name in model_names:
                print(f"    模型：{model_name}")

                summary_df, pred_df, dropped_df = run_cv_for_config_model(
                    master_df=master_df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    config_name=config_name,
                    model_name=model_name,
                    splits=splits,
                )

                if summary_df is None:
                    print(f"    跳过 {model_name}：依赖未安装")
                    continue

                comparison_list.append(summary_df)
                prediction_list.append(pred_df)

                if dropped_df is not None and len(dropped_df) > 0:
                    dropped_list.append(dropped_df)

                # 特征重要性只需要每个 config/model 提取一次，不按 protocol 重复提取
                if protocol_name == "random_kfold":
                    importance_df = fit_full_model_importance(
                        master_df=master_df,
                        feature_cols=feature_cols,
                        target_col=target_col,
                        config_name=config_name,
                        model_name=model_name,
                    )

                    if len(importance_df) > 0:
                        importance_list.append(importance_df)

    comparison_df = pd.concat(comparison_list, axis=0, ignore_index=True)
    predictions_df = pd.concat(prediction_list, axis=0, ignore_index=True)

    if len(dropped_list) > 0:
        dropped_df = pd.concat(dropped_list, axis=0, ignore_index=True)
    else:
        dropped_df = pd.DataFrame(columns=["feature", "reason", "dominant_ratio", "std"])

    if len(importance_list) > 0:
        importance_df = pd.concat(importance_list, axis=0, ignore_index=True)
    else:
        importance_df = pd.DataFrame()

    comparison_df = comparison_df.sort_values(
        ["protocol", "rmse_mean", "mae_mean"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    save_stage4_regression_outputs(
        comparison_df=comparison_df,
        predictions_df=predictions_df,
        importance_df=importance_df,
        dropped_df=dropped_df,
    )

    return comparison_df, predictions_df, importance_df, dropped_df


# 保存阶段四 Step 3 输出
# 输入：结果表
# 输出：无
def save_stage4_regression_outputs(comparison_df, predictions_df, importance_df, dropped_df):
    save_csv(
        comparison_df,
        os.path.join(CONFIG["interim_dir"], "stage4_config_comparison.csv")
    )

    save_csv(
        predictions_df,
        os.path.join(CONFIG["interim_dir"], "stage4_predictions.csv")
    )

    save_csv(
        importance_df,
        os.path.join(CONFIG["interim_dir"], "stage4_feature_importance.csv")
    )

    save_csv(
        dropped_df,
        os.path.join(CONFIG["interim_dir"], "stage4_dropped_features.csv")
    )