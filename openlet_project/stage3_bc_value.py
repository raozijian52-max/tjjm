# 文件位置：stage3_bc_value.py

import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG
from utils import save_csv


# 设置随机种子
# 输入：seed 整数
# 输出：无
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 获取当前可用设备
# 输入：无
# 输出：torch device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# 读取单个场景的 aligned_data.pkl
# 输入：scene_id，例如 S1
# 输出：该场景的 aligned_dict
def load_scene_aligned_data(scene_id):
    scene_prefix = scene_id.lower()
    aligned_path = os.path.join(CONFIG["interim_dir"], f"{scene_prefix}_aligned_data.pkl")

    if not os.path.exists(aligned_path):
        raise FileNotFoundError(f"未找到 {scene_id} 的对齐文件：{aligned_path}")

    with open(aligned_path, "rb") as f:
        aligned_dict = pickle.load(f)

    return aligned_dict


# 读取所有场景的对齐结果
# 输入：scene_ids 列表
# 输出：
#   all_aligned: {trajectory_id: aligned}
#   traj_info_df: 每条轨迹的 scene_id 与 trajectory_id
def load_all_scenes_aligned_data(scene_ids):
    all_aligned = {}
    rows = []

    for scene_id in scene_ids:
        scene_aligned = load_scene_aligned_data(scene_id)

        for trajectory_id, aligned in scene_aligned.items():
            # 为避免不同场景出现同名 trajectory_id，这里构造全局 ID
            global_id = f"{scene_id}_{trajectory_id}"

            aligned["global_id"] = global_id
            aligned["scene_id"] = scene_id

            all_aligned[global_id] = aligned

            rows.append({
                "global_id": global_id,
                "trajectory_id": trajectory_id,
                "scene_id": scene_id,
                "n_frames": len(aligned["time_grid_s"]),
            })

    traj_info_df = pd.DataFrame(rows)
    return all_aligned, traj_info_df


# 按轨迹划分 train/val
# 输入：轨迹信息表
# 输出：带 split 列的 DataFrame
def build_trajectory_split(traj_info_df):
    rng = np.random.RandomState(CONFIG["random_state"])
    split_rows = []

    # 每个场景内部单独划分，保证每个场景都有验证轨迹
    for scene_id, scene_df in traj_info_df.groupby("scene_id"):
        ids = scene_df["global_id"].tolist()
        rng.shuffle(ids)

        n_train = int(len(ids) * CONFIG["train_ratio"])
        train_ids = set(ids[:n_train])
        val_ids = set(ids[n_train:])

        for global_id in ids:
            if global_id in train_ids:
                split = "train"
            elif global_id in val_ids:
                split = "val"
            else:
                split = "unused"

            split_rows.append({
                "global_id": global_id,
                "scene_id": scene_id,
                "split": split,
            })

    split_df = pd.DataFrame(split_rows)
    return split_df


# 从单条轨迹构造 BC 滑动窗口样本
# 输入：
#   aligned: 单条对齐轨迹
#   window_size: 历史窗口长度
#   stride: 采样步长
# 输出：
#   X: 形状 (N, window_size * 14)
#   y: 形状 (N, 14)
def build_bc_samples_from_trajectory(aligned, window_size, stride):
    arm_state = np.asarray(aligned["arm_state_100hz"], dtype=np.float32)
    arm_action = np.asarray(aligned["arm_action_100hz"], dtype=np.float32)

    if arm_state.ndim != 2 or arm_action.ndim != 2:
        return np.empty((0, window_size * 14), dtype=np.float32), np.empty((0, 14), dtype=np.float32)

    if arm_state.shape[1] != 14 or arm_action.shape[1] != 14:
        return np.empty((0, window_size * 14), dtype=np.float32), np.empty((0, 14), dtype=np.float32)

    if len(arm_state) < window_size:
        return np.empty((0, window_size * 14), dtype=np.float32), np.empty((0, 14), dtype=np.float32)

    X_list = []
    y_list = []

    # 第 t 个样本使用 [t-window_size+1, ..., t] 的 state 预测 t 时刻 action
    for t in range(window_size - 1, len(arm_state), stride):
        state_window = arm_state[t - window_size + 1:t + 1]
        action_t = arm_action[t]

        # 丢弃含 NaN 的样本，避免训练不稳定
        if np.isnan(state_window).any() or np.isnan(action_t).any():
            continue

        X_list.append(state_window.reshape(-1))
        y_list.append(action_t)

    if len(X_list) == 0:
        return np.empty((0, window_size * 14), dtype=np.float32), np.empty((0, 14), dtype=np.float32)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    return X, y


# 从指定轨迹集合构造 BC 数据集
# 输入：
#   all_aligned: 全部轨迹字典
#   global_ids: 需要使用的轨迹 ID
# 输出：
#   X, y, sample_meta_df
def build_bc_dataset(all_aligned, global_ids):
    X_all = []
    y_all = []
    meta_rows = []

    window_size = CONFIG["bc_window_size"]
    stride = CONFIG["bc_sample_stride"]

    for global_id in global_ids:
        aligned = all_aligned[global_id]
        X, y = build_bc_samples_from_trajectory(aligned, window_size, stride)

        if len(X) == 0:
            continue

        X_all.append(X)
        y_all.append(y)

        for i in range(len(X)):
            meta_rows.append({
                "global_id": global_id,
                "scene_id": aligned["scene_id"],
                "sample_index_in_traj": i,
            })

    if len(X_all) == 0:
        raise ValueError("没有构造出任何 BC 样本，请检查 aligned_data 内容。")

    X_all = np.vstack(X_all)
    y_all = np.vstack(y_all)
    sample_meta_df = pd.DataFrame(meta_rows)

    return X_all, y_all, sample_meta_df


# 基于所有训练轨迹动作，计算统一评估用的动作标准差
# 输入：全部对齐数据、split_df
# 输出：eval_y_std，形状为 (14,)
def compute_common_eval_action_std(all_aligned, split_df):
    train_ids = split_df[split_df["split"] == "train"]["global_id"].tolist()

    _, y_train_all, _ = build_bc_dataset(all_aligned, train_ids)

    eval_y_std = np.nanstd(y_train_all, axis=0).astype(np.float32)

    # 防止某些动作维度方差极小导致除零
    eval_y_std[eval_y_std < 1e-8] = 1.0

    return eval_y_std


# MLP 行为克隆模型
# 输入：window_size * 14 维状态窗口
# 输出：14 维动作
class BCMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # 构造多层全连接网络
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    # 前向传播
    # 输入：x，形状 (batch, input_dim)
    # 输出：预测动作，形状 (batch, 14)
    def forward(self, x):
        return self.net(x)


# 训练一个 BC 模型
# 输入：训练集 X/y、验证集 X/y
# 输出：模型、X scaler、y scaler、训练日志
def train_bc_model(X_train, y_train, X_val, y_val):
    device = get_device()

    # 标准化输入和输出，提升 MLP 训练稳定性
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = X_scaler.fit_transform(X_train).astype(np.float32)
    y_train_s = y_scaler.fit_transform(y_train).astype(np.float32)

    X_val_s = X_scaler.transform(X_val).astype(np.float32)
    y_val_s = y_scaler.transform(y_val).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_s),
        torch.from_numpy(y_train_s),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["bc_batch_size"],
        shuffle=True,
        num_workers=CONFIG["bc_num_workers"],
    )

    input_dim = X_train_s.shape[1]
    output_dim = y_train_s.shape[1]

    model = BCMLP(
        input_dim=input_dim,
        hidden_dims=CONFIG["bc_hidden_dims"],
        output_dim=output_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["bc_learning_rate"],
        weight_decay=CONFIG["bc_weight_decay"],
    )

    loss_fn = nn.MSELoss()

    log_rows = []

    # 训练循环
    for epoch in range(1, CONFIG["bc_epochs"] + 1):
        model.train()
        train_losses = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # 每个 epoch 后在验证集上计算标准化 MSE
        model.eval()
        with torch.no_grad():
            val_X_tensor = torch.from_numpy(X_val_s).to(device)
            val_y_tensor = torch.from_numpy(y_val_s).to(device)
            val_pred = model(val_X_tensor)
            val_loss = loss_fn(val_pred, val_y_tensor).item()

        log_rows.append({
            "epoch": epoch,
            "train_mse_scaled": float(np.mean(train_losses)),
            "val_mse_scaled": float(val_loss),
        })

    train_log_df = pd.DataFrame(log_rows)

    return model, X_scaler, y_scaler, train_log_df


# 使用训练好的模型进行预测
# 输入：模型、scaler、原始 X
# 输出：原始动作尺度下的预测值、标准化尺度下的预测值
def predict_bc(model, X, X_scaler, y_scaler):
    device = get_device()

    X_s = X_scaler.transform(X).astype(np.float32)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_s).to(device)
        pred_s = model(X_tensor).cpu().numpy()

    pred = y_scaler.inverse_transform(pred_s)
    return pred, pred_s


# 计算 BC 离线指标
# 输入：真实动作、预测动作、统一评估动作标准差
# 输出：指标字典
def compute_bc_metrics(y_true, y_pred, eval_y_std):
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))

    # 使用统一动作标准差进行评估归一化
    # 注意：这里不再使用每个模型自己的 y_scaler，以保证 base 和 leave-one 可比较
    error_norm = (y_pred - y_true) / eval_y_std.reshape(1, -1)
    normalized_mse = float(np.mean(error_norm ** 2))

    # 分数越高越好；所有模型共用同一评估尺度
    imitation_score = float(np.exp(-normalized_mse))

    return {
        "mse": mse,
        "mae": mae,
        "normalized_mse": normalized_mse,
        "imitation_score": imitation_score,
        "n_samples": int(len(y_true)),
    }


# 按场景评估模型
# 输入：模型、scaler、验证集 X/y/meta
# 输出：每个场景一行指标
def evaluate_model_by_scene(model, X_val, y_val, meta_val_df, X_scaler, y_scaler, eval_y_std):
    y_pred, _ = predict_bc(model, X_val, X_scaler, y_scaler)

    rows = []

    for scene_id in sorted(meta_val_df["scene_id"].unique()):
        idx = meta_val_df["scene_id"].values == scene_id

        metrics = compute_bc_metrics(
            y_true=y_val[idx],
            y_pred=y_pred[idx],
            eval_y_std=eval_y_std,
        )

        metrics["scene_id"] = scene_id
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    return metrics_df


# 训练全数据 base BC 模型
# 输入：全部对齐数据与 split 表
# 输出：base metrics、训练日志、模型组件
def train_and_eval_base_model(all_aligned, split_df, eval_y_std):
    train_ids = split_df[split_df["split"] == "train"]["global_id"].tolist()
    val_ids = split_df[split_df["split"] == "val"]["global_id"].tolist()

    X_train, y_train, _ = build_bc_dataset(all_aligned, train_ids)
    X_val, y_val, meta_val_df = build_bc_dataset(all_aligned, val_ids)

    model, X_scaler, y_scaler, train_log_df = train_bc_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    base_metrics_df = evaluate_model_by_scene(
        model=model,
        X_val=X_val,
        y_val=y_val,
        meta_val_df=meta_val_df,
        X_scaler=X_scaler,
        y_scaler=y_scaler,
        eval_y_std=eval_y_std,
    )

    base_metrics_df["model_type"] = "base_all_scenes"

    return base_metrics_df, train_log_df, model, X_scaler, y_scaler


# 训练留一场景模型并评估
# 输入：全部对齐数据与 split 表
# 输出：leave-one 指标表
def train_and_eval_leave_one_models(all_aligned, split_df, eval_y_std):
    rows = []

    scene_ids = CONFIG["scene_ids"]

    for heldout_scene in scene_ids:
        # 训练集：所有非 heldout_scene 的 train 轨迹
        train_df = split_df[
            (split_df["split"] == "train") &
            (split_df["scene_id"] != heldout_scene)
        ]

        # 验证集：heldout_scene 的 val 轨迹
        val_df = split_df[
            (split_df["split"] == "val") &
            (split_df["scene_id"] == heldout_scene)
        ]

        train_ids = train_df["global_id"].tolist()
        val_ids = val_df["global_id"].tolist()

        if len(train_ids) == 0 or len(val_ids) == 0:
            rows.append({
                "scene_id": heldout_scene,
                "model_type": "leave_one_scene",
                "mse": np.nan,
                "mae": np.nan,
                "normalized_mse": np.nan,
                "imitation_score": np.nan,
                "n_samples": 0,
                "note": "empty_train_or_val",
            })
            continue

        X_train, y_train, _ = build_bc_dataset(all_aligned, train_ids)
        X_val, y_val, meta_val_df = build_bc_dataset(all_aligned, val_ids)

        model, X_scaler, y_scaler, _ = train_bc_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        metrics_df = evaluate_model_by_scene(
            model=model,
            X_val=X_val,
            y_val=y_val,
            meta_val_df=meta_val_df,
            X_scaler=X_scaler,
            y_scaler=y_scaler,
            eval_y_std=eval_y_std,
        )

        # 这里只会有 heldout_scene 一行
        metrics_row = metrics_df.iloc[0].to_dict()
        metrics_row["scene_id"] = heldout_scene
        metrics_row["model_type"] = "leave_one_scene"
        metrics_row["note"] = ""

        rows.append(metrics_row)

    leave_one_metrics_df = pd.DataFrame(rows)
    return leave_one_metrics_df


# 计算场景边际价值
# 输入：base 指标表、leave-one 指标表
# 输出：scene_delta_value_df
def compute_scene_delta_value(base_metrics_df, leave_one_metrics_df):
    base = base_metrics_df[[
        "scene_id",
        "mse",
        "mae",
        "normalized_mse",
        "imitation_score",
        "n_samples",
    ]].copy()

    leave = leave_one_metrics_df[[
        "scene_id",
        "mse",
        "mae",
        "normalized_mse",
        "imitation_score",
        "n_samples",
    ]].copy()

    base = base.rename(columns={
        "mse": "base_mse",
        "mae": "base_mae",
        "normalized_mse": "base_normalized_mse",
        "imitation_score": "base_imitation_score",
        "n_samples": "base_n_samples",
    })

    leave = leave.rename(columns={
        "mse": "leave_mse",
        "mae": "leave_mae",
        "normalized_mse": "leave_normalized_mse",
        "imitation_score": "leave_imitation_score",
        "n_samples": "leave_n_samples",
    })

    delta_df = pd.merge(base, leave, on="scene_id", how="left")

    # delta_score > 0 表示去掉该场景训练数据后，该场景验证表现变差
    delta_df["delta_score"] = (
        delta_df["base_imitation_score"] - delta_df["leave_imitation_score"]
    )

    # delta_mse > 0 表示去掉该场景训练数据后，误差变大
    delta_df["delta_normalized_mse"] = (
        delta_df["leave_normalized_mse"] - delta_df["base_normalized_mse"]
    )

    return delta_df


# 保存阶段三输出
# 输入：split、base metrics、leave-one metrics、delta value、训练日志
# 输出：无
def save_stage3_outputs(split_df, base_metrics_df, leave_one_metrics_df, delta_df, train_log_df):
    split_path = os.path.join(CONFIG["interim_dir"], "s_all_bc_split.csv")
    base_path = os.path.join(CONFIG["interim_dir"], "stage3_bc_base_metrics_by_scene.csv")
    leave_path = os.path.join(CONFIG["interim_dir"], "stage3_bc_leave_one_metrics.csv")
    delta_path = os.path.join(CONFIG["interim_dir"], "stage3_scene_delta_value.csv")
    log_path = os.path.join(CONFIG["interim_dir"], "stage3_base_train_log.csv")

    save_csv(split_df, split_path)
    save_csv(base_metrics_df, base_path)
    save_csv(leave_one_metrics_df, leave_path)
    save_csv(delta_df, delta_path)
    save_csv(train_log_df, log_path)


# 运行阶段三：离线 BC 效能版
# 输入：无
# 输出：base_metrics_df、leave_one_metrics_df、delta_df
def run_stage3_bc_value():
    set_seed(CONFIG["random_state"])

    # 1. 读取 S1-S5 或当前配置中的多场景对齐数据
    all_aligned, traj_info_df = load_all_scenes_aligned_data(CONFIG["scene_ids"])
    print("数据读取成功。")

    # 2. 按轨迹构造 train/val split
    split_df = build_trajectory_split(traj_info_df)
    print("训练测试集构造成功，开始训练base模型：")

    # 3. 基于所有训练轨迹动作，计算统一评估尺度
    eval_y_std = compute_common_eval_action_std(
        all_aligned=all_aligned,
        split_df=split_df,
    )

    # 4. 训练全数据 base 模型，并在每个场景验证集上评估
    base_metrics_df, train_log_df, _, _, _ = train_and_eval_base_model(
        all_aligned=all_aligned,
        split_df=split_df,
        eval_y_std=eval_y_std,
    )

    # 5. 训练 leave-one-scene 模型，并评估被留出场景
    print("开始训练留一模型")
    leave_one_metrics_df = train_and_eval_leave_one_models(
        all_aligned=all_aligned,
        split_df=split_df,
        eval_y_std=eval_y_std,
    )

    # 5. 计算场景边际价值
    print("计算场景边际价值")
    delta_df = compute_scene_delta_value(
        base_metrics_df=base_metrics_df,
        leave_one_metrics_df=leave_one_metrics_df,
    )

    # 6. 保存结果
    print("计算完成！结果保存中")
    save_stage3_outputs(
        split_df=split_df,
        base_metrics_df=base_metrics_df,
        leave_one_metrics_df=leave_one_metrics_df,
        delta_df=delta_df,
        train_log_df=train_log_df,
    )

    return base_metrics_df, leave_one_metrics_df, delta_df