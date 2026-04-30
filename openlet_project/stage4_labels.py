# 文件位置：stage4_labels.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from config_template import CONFIG
from stage3_bc_value import (
    set_seed,
    load_all_scenes_aligned_data,
    build_bc_dataset,
    train_bc_model,
    predict_bc,
)
from utils import save_csv


# 按轨迹构造 out-of-fold 划分
# 输入：traj_info_df，每行一条轨迹
# 输出：带 fold 列的 DataFrame
def build_oof_folds(traj_info_df):
    rng_seed = CONFIG["random_state"]
    n_folds = CONFIG.get("stage4_oof_folds", 5)

    rows = []

    # 每个场景内部独立 KFold，再合并，避免某一折缺少某个场景
    for scene_id, scene_df in traj_info_df.groupby("scene_id"):
        scene_df = scene_df.sample(frac=1.0, random_state=rng_seed).reset_index(drop=True)

        n_scene = len(scene_df)
        real_folds = min(n_folds, n_scene)

        kf = KFold(n_splits=real_folds, shuffle=True, random_state=rng_seed)

        for fold_id, (_, val_idx) in enumerate(kf.split(scene_df)):
            for idx in val_idx:
                row = scene_df.iloc[idx].to_dict()
                row["fold"] = fold_id
                rows.append(row)

    fold_df = pd.DataFrame(rows)

    # 如果不同场景的 real_folds 不一致，fold_id 仍然可用；
    # 主训练时按 fold_id 聚合所有场景的验证轨迹。
    fold_df = fold_df.sort_values(["fold", "scene_id", "global_id"]).reset_index(drop=True)

    return fold_df


# 计算全局动作标准差，用于统一 normalized_mse 口径
# 输入：all_aligned、全部轨迹 ID
# 输出：14维动作标准差
def compute_global_action_std(all_aligned, global_ids):
    action_list = []

    for global_id in global_ids:
        action = np.asarray(all_aligned[global_id]["arm_action_100hz"], dtype=np.float32)

        if action.ndim == 2 and action.shape[1] == 14:
            action = action[~np.isnan(action).any(axis=1)]
            if len(action) > 0:
                action_list.append(action)

    if len(action_list) == 0:
        raise ValueError("无法计算全局动作标准差：没有有效 arm_action_100hz。")

    all_action = np.vstack(action_list)
    std = np.std(all_action, axis=0).astype(np.float32)

    # 防止除零
    std[std < 1e-8] = 1.0

    return std


# 根据预测结果计算每个样本的误差
# 输入：真实动作、预测动作、全局动作标准差
# 输出：每个样本的 mse/mae/normalized_mse
def compute_sample_errors(y_true, y_pred, eval_y_std):
    err = y_pred - y_true

    sample_mse = np.mean(err ** 2, axis=1)
    sample_mae = np.mean(np.abs(err), axis=1)

    norm_err = err / eval_y_std.reshape(1, -1)
    sample_normalized_mse = np.mean(norm_err ** 2, axis=1)

    return sample_mse, sample_mae, sample_normalized_mse


# 将窗口级误差聚合为轨迹级标签
# 输入：meta_val_df、窗口级误差
# 输出：轨迹级标签表
def aggregate_errors_to_trajectory(meta_val_df, sample_mse, sample_mae, sample_normalized_mse):
    df = meta_val_df.copy()
    df["sample_mse"] = sample_mse
    df["sample_mae"] = sample_mae
    df["sample_normalized_mse"] = sample_normalized_mse

    rows = []

    for global_id, group in df.groupby("global_id"):
        scene_id = group["scene_id"].iloc[0]

        trajectory_mse = float(group["sample_mse"].mean())
        trajectory_mae = float(group["sample_mae"].mean())
        trajectory_normalized_mse = float(group["sample_normalized_mse"].mean())
        trajectory_imitation_score = float(np.exp(-trajectory_normalized_mse))

        rows.append({
            "global_id": global_id,
            "trajectory_id": global_id.split("_", 1)[1],
            "scene_id": scene_id,
            "trajectory_mse": trajectory_mse,
            "trajectory_mae": trajectory_mae,
            "trajectory_normalized_mse": trajectory_normalized_mse,
            "trajectory_imitation_score": trajectory_imitation_score,
            "n_bc_windows": int(len(group)),
        })

    return pd.DataFrame(rows)


# 运行一个 fold 的训练与验证
# 输入：all_aligned、fold_df、当前 fold、统一动作标准差
# 输出：该 fold 的轨迹级标签
def run_one_oof_fold(all_aligned, fold_df, fold_id, eval_y_std):
    val_ids = fold_df[fold_df["fold"] == fold_id]["global_id"].tolist()
    train_ids = fold_df[fold_df["fold"] != fold_id]["global_id"].tolist()

    if len(train_ids) == 0 or len(val_ids) == 0:
        raise ValueError(f"fold={fold_id} 的训练集或验证集为空。")

    X_train, y_train, _ = build_bc_dataset(all_aligned, train_ids)
    X_val, y_val, meta_val_df = build_bc_dataset(all_aligned, val_ids)

    # 阶段四标签生成可以用较少 epoch，目的是得到稳定的 out-of-fold 代理标签
    old_epochs = CONFIG["bc_epochs"]
    CONFIG["bc_epochs"] = CONFIG.get("stage4_bc_epochs", old_epochs)

    model, X_scaler, y_scaler, _ = train_bc_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    CONFIG["bc_epochs"] = old_epochs

    y_pred, _ = predict_bc(
        model=model,
        X=X_val,
        X_scaler=X_scaler,
        y_scaler=y_scaler,
    )

    sample_mse, sample_mae, sample_normalized_mse = compute_sample_errors(
        y_true=y_val,
        y_pred=y_pred,
        eval_y_std=eval_y_std,
    )

    fold_label_df = aggregate_errors_to_trajectory(
        meta_val_df=meta_val_df,
        sample_mse=sample_mse,
        sample_mae=sample_mae,
        sample_normalized_mse=sample_normalized_mse,
    )

    fold_label_df["fold"] = fold_id

    return fold_label_df


# 保存阶段四 Step 1 输出
# 输入：标签表、fold表、摘要表
# 输出：无
def save_stage4_label_outputs(label_df, fold_df, summary_df):
    save_csv(
        label_df,
        os.path.join(CONFIG["interim_dir"], "stage4_bc_trajectory_labels.csv")
    )

    save_csv(
        fold_df,
        os.path.join(CONFIG["interim_dir"], "stage4_oof_folds.csv")
    )

    save_csv(
        summary_df,
        os.path.join(CONFIG["interim_dir"], "stage4_bc_label_summary.csv")
    )


# 生成场景级标签摘要
# 输入：轨迹级标签表
# 输出：场景级摘要表
def summarize_stage4_labels(label_df):
    rows = []

    for scene_id, group in label_df.groupby("scene_id"):
        rows.append({
            "scene_id": scene_id,
            "n_trajectories": int(len(group)),
            "trajectory_normalized_mse_mean": float(group["trajectory_normalized_mse"].mean()),
            "trajectory_normalized_mse_std": float(group["trajectory_normalized_mse"].std(ddof=1)),
            "trajectory_imitation_score_mean": float(group["trajectory_imitation_score"].mean()),
            "trajectory_imitation_score_std": float(group["trajectory_imitation_score"].std(ddof=1)),
        })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("scene_id").reset_index(drop=True)

    return summary_df


# 运行阶段四 Step 1
# 输入：无
# 输出：label_df, summary_df
def run_stage4_bc_trajectory_labels():
    set_seed(CONFIG["random_state"])

    # 1. 读取 S1-S5 对齐数据
    all_aligned, traj_info_df = load_all_scenes_aligned_data(CONFIG["scene_ids"])

    # 2. 构造轨迹级 OOF folds
    fold_df = build_oof_folds(traj_info_df)

    # 3. 统一动作误差归一化尺度
    eval_y_std = compute_global_action_std(
        all_aligned=all_aligned,
        global_ids=fold_df["global_id"].tolist(),
    )

    # 4. 逐 fold 训练 BC，并在 held-out 轨迹上生成标签
    label_rows = []
    fold_ids = sorted(fold_df["fold"].unique())

    for fold_id in fold_ids:
        print(f"开始 OOF fold {fold_id} / {len(fold_ids) - 1}")

        fold_label_df = run_one_oof_fold(
            all_aligned=all_aligned,
            fold_df=fold_df,
            fold_id=fold_id,
            eval_y_std=eval_y_std,
        )

        label_rows.append(fold_label_df)

    label_df = pd.concat(label_rows, axis=0, ignore_index=True)
    label_df = label_df.sort_values(["scene_id", "trajectory_id"]).reset_index(drop=True)

    # 5. 摘要
    summary_df = summarize_stage4_labels(label_df)

    # 6. 保存
    save_stage4_label_outputs(
        label_df=label_df,
        fold_df=fold_df,
        summary_df=summary_df,
    )

    return label_df, summary_df