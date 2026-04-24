# 文件位置：stage1_features.py

import os
import pickle
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.decomposition import PCA

from config import CONFIG
from stage1_read_and_manifest import read_one_h5
from utils import save_csv, save_json


# 统一采样若干帧，避免一次性解码全部图像导致过慢
# 输入：总帧数、最多采样帧数
# 输出：采样索引数组
def uniform_sample_indices(n_total, max_samples):
    if n_total <= 0:
        return np.array([], dtype=int)

    if n_total <= max_samples:
        return np.arange(n_total, dtype=int)

    return np.linspace(0, n_total - 1, num=max_samples, dtype=int)


# 解码单张图像字节流
# 输入：
#   img_bytes: 图像字节
#   mode: "color" 表示按 RGB 图像解码；"unchanged" 表示尽量保持原格式
# 输出：解码后的 numpy 数组；若失败则返回 None
def decode_image_bytes(img_bytes, mode):
    if img_bytes is None or len(img_bytes) == 0:
        return None

    try:
        with Image.open(BytesIO(img_bytes)) as img:
            if mode == "color":
                # 统一转成 RGB 三通道
                img = img.convert("RGB")
                return np.array(img)

            if mode == "unchanged":
                # 尽量保留原始格式，例如灰度深度图
                return np.array(img)

            # 未知模式时，默认返回原始数组
            return np.array(img)

    except (UnidentifiedImageError, OSError, ValueError):
        return None


# 计算二维序列的逐维均值、标准差、极差
# 输入：二维数组、特征名前缀
# 输出：特征字典
def extract_basic_stats_per_dim(seq, prefix):
    seq = np.asarray(seq, dtype=float)
    feature_dict = {}

    mean_vec = np.nanmean(seq, axis=0)
    std_vec = np.nanstd(seq, axis=0)
    range_vec = np.nanmax(seq, axis=0) - np.nanmin(seq, axis=0)

    for i in range(seq.shape[1]):
        feature_dict[f"{prefix}_mean_{i}"] = float(mean_vec[i])
        feature_dict[f"{prefix}_std_{i}"] = float(std_vec[i])
        feature_dict[f"{prefix}_range_{i}"] = float(range_vec[i])

    return feature_dict


# 计算二维序列的一阶差分绝对均值
# 输入：二维数组、时间步长、特征名前缀
# 输出：特征字典
def extract_abs_velocity_mean(seq, dt, prefix):
    seq = np.asarray(seq, dtype=float)
    feature_dict = {}

    if len(seq) < 2:
        for i in range(seq.shape[1]):
            feature_dict[f"{prefix}_vel_abs_mean_{i}"] = np.nan
        return feature_dict

    vel = np.diff(seq, axis=0) / dt
    vel_abs_mean = np.nanmean(np.abs(vel), axis=0)

    for i in range(seq.shape[1]):
        feature_dict[f"{prefix}_vel_abs_mean_{i}"] = float(vel_abs_mean[i])

    return feature_dict


# 计算离散 jerk 能量特征
# 定义：
#   v = d(theta)/dt
#   a = d(v)/dt
#   j = d(a)/dt
#   jerk_energy = sum(j^2) * dt
# 输入：二维位置序列、时间步长、特征名前缀
# 输出：每个维度的 jerk_energy 特征
def extract_jerk_energy(seq, dt, prefix):
    seq = np.asarray(seq, dtype=float)
    feature_dict = {}

    n_dim = seq.shape[1]

    # 至少需要 4 个时间点才能计算到三阶差分
    if len(seq) < 4:
        for i in range(n_dim):
            feature_dict[f"{prefix}_jerk_energy_{i}"] = np.nan
        return feature_dict

    vel = np.diff(seq, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt

    jerk_energy = np.nansum(jerk ** 2, axis=0) * dt

    for i in range(n_dim):
        feature_dict[f"{prefix}_jerk_energy_{i}"] = float(jerk_energy[i])

    return feature_dict


# 计算轨迹路径长度（这里用 L1 近似，轻量且稳定）
# 输入：二维序列
# 输出：路径长度标量
def compute_path_l1(seq):
    seq = np.asarray(seq, dtype=float)

    if len(seq) < 2:
        return 0.0

    diff = np.diff(seq, axis=0)
    path_l1 = np.nansum(np.abs(diff))
    return float(path_l1)


# 在全场景所有轨迹的 effector_state 上拟合一个统一的 PCA(1)
# 输入：aligned_dict，键为 trajectory_id，值为对齐后的轨迹字典
# 输出：pca_model, pca_info
def fit_global_effector_pca(aligned_dict):
    effector_frames = []

    for trajectory_id in sorted(aligned_dict.keys()):
        aligned = aligned_dict[trajectory_id]
        effector_state = aligned.get("effector_state_100hz")

        if effector_state is None:
            continue

        effector_state = np.asarray(effector_state, dtype=float)

        # 只保留没有 NaN 的有效时间帧
        valid_mask = ~np.isnan(effector_state).any(axis=1)
        valid_frames = effector_state[valid_mask]

        if len(valid_frames) > 0:
            effector_frames.append(valid_frames)

    # 如果没有足够数据，则返回 None，后续自动回退到均值方案
    if len(effector_frames) == 0:
        pca_info = {
            "status": "no_valid_effector_frames",
            "n_fit_samples": 0,
            "n_features": 12,
            "explained_variance_ratio": None,
            "components": None,
        }
        return None, pca_info

    fit_data = np.concatenate(effector_frames, axis=0)

    if len(fit_data) < 2:
        pca_info = {
            "status": "not_enough_effector_frames",
            "n_fit_samples": int(len(fit_data)),
            "n_features": int(fit_data.shape[1]),
            "explained_variance_ratio": None,
            "components": None,
        }
        return None, pca_info

    pca_model = PCA(n_components=1)
    pca_model.fit(fit_data)

    # 固定主成分符号方向，减少不同运行中符号翻转带来的可解释性波动
    if float(np.sum(pca_model.components_[0])) < 0:
        pca_model.components_[0] = -1.0 * pca_model.components_[0]

    pca_info = {
        "status": "ok",
        "n_fit_samples": int(len(fit_data)),
        "n_features": int(fit_data.shape[1]),
        "explained_variance_ratio": float(pca_model.explained_variance_ratio_[0]),
        "components": pca_model.components_[0].tolist(),
        "mean": pca_model.mean_.tolist(),
    }

    return pca_model, pca_info


# 对一维信号中的 NaN 做线性插值；首尾缺失用最近有效值填充
# 输入：一维数组
# 输出：补齐后的数组
def fill_nan_1d(signal):
    signal = np.asarray(signal, dtype=float)

    if len(signal) == 0:
        return signal

    valid_mask = ~np.isnan(signal)

    # 如果全是 NaN，就直接返回 0 向量，避免后续分位数报错
    if valid_mask.sum() == 0:
        return np.zeros(len(signal), dtype=float)

    x = np.arange(len(signal))
    filled = signal.copy()
    filled[~valid_mask] = np.interp(x[~valid_mask], x[valid_mask], signal[valid_mask])

    return filled


# 用全局 PCA 第一主成分把 effector_state 投影成 1 维抓取代理信号
# 输入：effector_state_100hz，pca_model
# 输出：一维信号，形状 (T,)
def build_effector_scalar_signal(effector_state, pca_model=None):
    effector_state = np.asarray(effector_state, dtype=float)

    # 若没有可用 PCA 模型，则回退到简单均值
    if pca_model is None:
        scalar_signal = np.nanmean(effector_state, axis=1)
        scalar_signal = fill_nan_1d(scalar_signal)
        return scalar_signal

    valid_mask = ~np.isnan(effector_state).any(axis=1)
    scalar_signal = np.full(len(effector_state), np.nan, dtype=float)

    if valid_mask.sum() > 0:
        projected = pca_model.transform(effector_state[valid_mask])[:, 0]
        scalar_signal[valid_mask] = projected

    scalar_signal = fill_nan_1d(scalar_signal)
    return scalar_signal


# 将 effector 一维信号离散为 3 个状态
# 输入：一维信号
# 输出：状态序列（0/1/2）
def discretize_effector_state(scalar_signal):
    scalar_signal = np.asarray(scalar_signal, dtype=float)

    q1 = np.nanpercentile(scalar_signal, 33.3)
    q2 = np.nanpercentile(scalar_signal, 66.7)

    states = np.zeros(len(scalar_signal), dtype=int)
    states[scalar_signal > q1] = 1
    states[scalar_signal > q2] = 2

    return states


# 提取离散语义特征
# 输入：effector_state_100hz、全局 PCA 模型
# 输出：语义特征字典
def extract_semantic_features(effector_state, effector_pca_model=None):
    feature_dict = {}

    scalar_signal = build_effector_scalar_signal(
        effector_state=effector_state,
        pca_model=effector_pca_model,
    )
    states = discretize_effector_state(scalar_signal)

    # 各状态占比
    for s in [0, 1, 2]:
        feature_dict[f"semantic_state_ratio_{s}"] = float(np.mean(states == s))

    # 转换频率
    if len(states) < 2:
        transition_count = 0
    else:
        transition_count = int(np.sum(states[1:] != states[:-1]))

    feature_dict["semantic_transition_rate"] = float(transition_count / max(len(states), 1))

    # 序列熵
    probs = np.array([np.mean(states == s) for s in [0, 1, 2]], dtype=float)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    feature_dict["semantic_entropy"] = float(entropy)

    # 末状态与主导状态
    feature_dict["semantic_final_state"] = int(states[-1]) if len(states) > 0 else -1
    feature_dict["semantic_dominant_state"] = int(np.bincount(states).argmax()) if len(states) > 0 else -1

    return feature_dict


# 提取单路相机的 RGB 统计特征
# 输入：color_bytes 列表、最多采样帧数、直方图 bin 数、特征名前缀
# 输出：RGB 特征字典
def extract_rgb_features(color_bytes_list, max_frames, hist_bins, prefix):
    feature_dict = {}

    # 先初始化所有字段，避免不同轨迹列不齐
    channel_names = ["r", "g", "b"]
    for ch in channel_names:
        feature_dict[f"{prefix}_rgb_mean_{ch}"] = np.nan
        feature_dict[f"{prefix}_rgb_std_{ch}"] = np.nan
        for b in range(hist_bins):
            feature_dict[f"{prefix}_rgb_hist_{ch}_{b}"] = np.nan

    if color_bytes_list is None or len(color_bytes_list) == 0:
        return feature_dict

    sample_idx = uniform_sample_indices(len(color_bytes_list), max_frames)

    rgb_pixels = []

    for idx in sample_idx:
        img = decode_image_bytes(color_bytes_list[idx], "color")

        if img is None:
            continue

        # Pillow 解码后已经是 RGB，不需要再做 BGR->RGB 翻转
        rgb_pixels.append(img.reshape(-1, 3))

    if len(rgb_pixels) == 0:
        return feature_dict

    rgb_pixels = np.concatenate(rgb_pixels, axis=0).astype(np.float32)

    # 计算每个通道的均值、标准差和归一化直方图
    for ch_idx, ch_name in enumerate(channel_names):
        channel_data = rgb_pixels[:, ch_idx]

        feature_dict[f"{prefix}_rgb_mean_{ch_name}"] = float(np.mean(channel_data))
        feature_dict[f"{prefix}_rgb_std_{ch_name}"] = float(np.std(channel_data))

        hist, _ = np.histogram(channel_data, bins=hist_bins, range=(0, 256), density=False)
        hist = hist.astype(float)

        if hist.sum() > 0:
            hist = hist / hist.sum()

        for b in range(hist_bins):
            feature_dict[f"{prefix}_rgb_hist_{ch_name}_{b}"] = float(hist[b])

    return feature_dict


# 提取单路相机的深度统计特征
# 输入：depth_bytes 列表、最多采样帧数、特征名前缀
# 输出：深度特征字典
def extract_depth_features(depth_bytes_list, max_frames, prefix):
    feature_dict = {
        f"{prefix}_depth_mean": np.nan,
        f"{prefix}_depth_std": np.nan,
        f"{prefix}_depth_hole_ratio": np.nan,
    }

    if depth_bytes_list is None or len(depth_bytes_list) == 0:
        return feature_dict

    sample_idx = uniform_sample_indices(len(depth_bytes_list), max_frames)

    valid_values = []
    hole_ratios = []

    for idx in sample_idx:
        depth = decode_image_bytes(depth_bytes_list[idx], "unchanged")

        if depth is None:
            continue

        depth = depth.astype(np.float32)

        # 若深度图被解码成三通道，这里转为单通道均值
        if depth.ndim == 3:
            depth = np.mean(depth, axis=2)

        invalid_mask = np.isnan(depth) | (depth <= 0)
        hole_ratio = float(np.mean(invalid_mask))
        hole_ratios.append(hole_ratio)

        valid = depth[~invalid_mask]
        if valid.size > 0:
            valid_values.append(valid)

    if len(hole_ratios) > 0:
        feature_dict[f"{prefix}_depth_hole_ratio"] = float(np.mean(hole_ratios))

    if len(valid_values) > 0:
        valid_values = np.concatenate(valid_values, axis=0)
        feature_dict[f"{prefix}_depth_mean"] = float(np.mean(valid_values))
        feature_dict[f"{prefix}_depth_std"] = float(np.std(valid_values))

    return feature_dict


# 提取单条轨迹的视觉特征
# 输入：read_one_h5 读出的原始轨迹字典
# 输出：视觉特征字典
def extract_vision_features(traj):
    feature_dict = {}

    # 轻量配置：每路最多采样 12 帧，RGB 直方图 4 bin
    max_frames = 12
    hist_bins = 4

    for cam_name in CONFIG["camera_names"]:
        cam = traj["cameras"].get(cam_name)

        if cam is None:
            # 相机缺失时也要补齐字段
            color_bytes = None
            depth_bytes = None
        else:
            color_bytes = cam["color_bytes"]
            depth_bytes = cam["depth_bytes"]

        rgb_features = extract_rgb_features(
            color_bytes_list=color_bytes,
            max_frames=max_frames,
            hist_bins=hist_bins,
            prefix=cam_name,
        )

        depth_features = extract_depth_features(
            depth_bytes_list=depth_bytes,
            max_frames=max_frames,
            prefix=cam_name,
        )

        feature_dict.update(rgb_features)
        feature_dict.update(depth_features)

    return feature_dict


# 提取单条轨迹的低维关节与语义特征
# 输入：第二步对齐后的单条轨迹字典、全局 PCA 模型
# 输出：关节与语义特征字典
def extract_lowdim_features(aligned, effector_pca_model=None):
    feature_dict = {}

    dt = 1.0 / 100.0

    arm_state = aligned["arm_state_100hz"]
    arm_action = aligned["arm_action_100hz"]
    effector_state = aligned["effector_state_100hz"]

    # arm_state：均值/标准差/极差 + 速度绝对均值 + jerk
    feature_dict.update(extract_basic_stats_per_dim(arm_state, "arm_state"))
    feature_dict.update(extract_abs_velocity_mean(arm_state, dt, "arm_state"))
    feature_dict.update(extract_jerk_energy(arm_state, dt, "arm_state"))

    # arm_action：均值/标准差/极差 + 速度绝对均值
    # 当前先不对 arm_action 加 jerk，保持轻量
    feature_dict.update(extract_basic_stats_per_dim(arm_action, "arm_action"))
    feature_dict.update(extract_abs_velocity_mean(arm_action, dt, "arm_action"))

    # effector_state：均值/标准差/极差 + 速度绝对均值 + jerk + 语义特征
    if effector_state is not None:
        feature_dict.update(extract_basic_stats_per_dim(effector_state, "effector_state"))
        feature_dict.update(extract_abs_velocity_mean(effector_state, dt, "effector_state"))
        feature_dict.update(extract_jerk_energy(effector_state, dt, "effector_state"))
        feature_dict.update(extract_semantic_features(effector_state, effector_pca_model))
        effector_path_l1 = compute_path_l1(effector_state)
    else:
        # 若不存在 effector_state，则补齐字段为 NaN
        for i in range(12):
            feature_dict[f"effector_state_mean_{i}"] = np.nan
            feature_dict[f"effector_state_std_{i}"] = np.nan
            feature_dict[f"effector_state_range_{i}"] = np.nan
            feature_dict[f"effector_state_vel_abs_mean_{i}"] = np.nan
            feature_dict[f"effector_state_jerk_energy_{i}"] = np.nan

        feature_dict["semantic_state_ratio_0"] = np.nan
        feature_dict["semantic_state_ratio_1"] = np.nan
        feature_dict["semantic_state_ratio_2"] = np.nan
        feature_dict["semantic_transition_rate"] = np.nan
        feature_dict["semantic_entropy"] = np.nan
        feature_dict["semantic_final_state"] = np.nan
        feature_dict["semantic_dominant_state"] = np.nan

        effector_path_l1 = np.nan

    # 全局特征
    duration_s = float(aligned["time_grid_s"][-1]) if len(aligned["time_grid_s"]) > 0 else 0.0
    arm_path_l1 = compute_path_l1(arm_state)

    feature_dict["global_duration_s"] = duration_s
    feature_dict["global_arm_path_l1"] = arm_path_l1
    feature_dict["global_effector_path_l1"] = effector_path_l1

    return feature_dict


# 提取单条轨迹的全部基础特征
# 输入：trajectory_id、文件路径、对齐后的轨迹字典、全局 PCA 模型
# 输出：一行特征字典
def extract_one_trajectory_features(trajectory_id, file_path, aligned, effector_pca_model=None):
    # 重新读取原始 H5，用于视觉字节流
    traj = read_one_h5(file_path)

    feature_dict = {
        "trajectory_id": trajectory_id,
        "scene_id": aligned["scene_id"],
        "task_name": aligned["task_name"],
    }

    # 低维统计 + 语义特征
    lowdim_features = extract_lowdim_features(
        aligned=aligned,
        effector_pca_model=effector_pca_model,
    )
    feature_dict.update(lowdim_features)

    # 视觉统计特征
    vision_features = extract_vision_features(traj)
    feature_dict.update(vision_features)

    return feature_dict


# 读取第二步和第三步的中间结果
# 输入：无
# 输出：manifest_df、aligned_dict、final_label_df
def load_step4_inputs():
    manifest_path = os.path.join(CONFIG["interim_dir"], "s3_manifest.csv")
    aligned_path = os.path.join(CONFIG["interim_dir"], "s3_aligned_data.pkl")
    final_label_path = os.path.join(CONFIG["interim_dir"], "s3_final_labels.csv")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError("未找到 s3_manifest.csv，请先运行阶段一第1步。")

    if not os.path.exists(aligned_path):
        raise FileNotFoundError("未找到 s3_aligned_data.pkl，请先运行阶段一第2步。")

    if not os.path.exists(final_label_path):
        raise FileNotFoundError("未找到 s3_final_labels.csv，请先运行阶段一第3步。")

    manifest_df = pd.read_csv(manifest_path)

    with open(aligned_path, "rb") as f:
        aligned_dict = pickle.load(f)

    final_label_df = pd.read_csv(final_label_path)

    return manifest_df, aligned_dict, final_label_df


# 保存第四步输出
# 输入：特征矩阵、合并后的阶段一数据表、PCA 信息
# 输出：无
def save_step4_outputs(feature_df, stage1_dataset_df, pca_info):
    feature_path = os.path.join(CONFIG["interim_dir"], "s3_feature_matrix.csv")
    dataset_path = os.path.join(CONFIG["interim_dir"], "s3_stage1_dataset.csv")
    pca_info_path = os.path.join(CONFIG["interim_dir"], "s3_effector_pca_info.json")

    save_csv(feature_df, feature_path)
    save_csv(stage1_dataset_df, dataset_path)
    save_json(pca_info, pca_info_path)


# 运行阶段一第4步
# 输入：无
# 输出：feature_df、stage1_dataset_df
def run_stage1_step4():
    manifest_df, aligned_dict, final_label_df = load_step4_inputs()

    # 先在全场景所有轨迹的 effector_state 上拟合统一 PCA
    effector_pca_model, pca_info = fit_global_effector_pca(aligned_dict)

    feature_rows = []

    # 按 manifest 顺序逐条提取特征
    for _, row in manifest_df.iterrows():
        trajectory_id = row["trajectory_id"]
        file_path = row["file_path"]

        aligned = aligned_dict[trajectory_id]
        feature_row = extract_one_trajectory_features(
            trajectory_id=trajectory_id,
            file_path=file_path,
            aligned=aligned,
            effector_pca_model=effector_pca_model,
        )
        feature_rows.append(feature_row)

    feature_df = pd.DataFrame(feature_rows)

    # 与最终标签表合并，形成阶段一可直接建模的数据表
    stage1_dataset_df = pd.merge(
        feature_df,
        final_label_df[["trajectory_id", "auto_label", "manual_label", "final_label"]],
        on="trajectory_id",
        how="left",
    )

    save_step4_outputs(feature_df, stage1_dataset_df, pca_info)

    return feature_df, stage1_dataset_df