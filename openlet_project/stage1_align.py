# 文件位置：stage1_align.py

import os
import pickle

import numpy as np
import pandas as pd

from config import CONFIG
from stage1_read_and_manifest import read_one_h5, run_stage1_step1
from utils import save_csv

# 将纳秒时间戳转为秒
# 输入：timestamp_ns，一维纳秒数组
# 输出：秒级 numpy 数组
def ns_to_seconds(timestamp_ns):
    if timestamp_ns is None:
        return None
    return np.asarray(timestamp_ns, dtype=np.int64) * 1e-9

# 对时间戳排序，并去掉重复时间点
# 输入：时间戳数组、对应的数值数组
# 输出：去重后的时间戳和数值
def prepare_interp_source(time_s, value):
    time_s = np.asarray(time_s, dtype=float)
    value = np.asarray(value)

    # 先按时间排序
    order = np.argsort(time_s)
    time_s = time_s[order]
    value = value[order]

    # 去重：重复时间戳只保留第一次出现
    unique_time, unique_index = np.unique(time_s, return_index=True)
    value = value[unique_index]

    return unique_time, value

# 构建 arm action/state 的公共 100Hz 时间轴
# 输入：action 时间戳、state 时间戳、目标频率
# 输出：
#   rel_grid_s      相对起点的时间轴
#   abs_grid_s      绝对秒时间轴
#   grid_start_abs  对齐起点的绝对秒
def build_common_time_grid(action_ts_ns, state_ts_ns, target_hz=100.0):
    action_s = ns_to_seconds(action_ts_ns)
    state_s = ns_to_seconds(state_ts_ns)

    # 公共时间段：从两个序列共同覆盖的区间开始到结束
    start_abs = max(action_s[0], state_s[0])
    end_abs = min(action_s[-1], state_s[-1])

    if end_abs <= start_abs:
        raise ValueError("action/state 没有重叠时间区间，无法构建公共时间轴。")

    dt = 1.0 / target_hz
    duration = end_abs - start_abs

    # 用相对时间构造网格，避免大绝对时间戳造成不必要的浮点误差
    rel_grid_s = np.arange(0.0, duration + 1e-12, dt)
    abs_grid_s = start_abs + rel_grid_s

    return rel_grid_s, abs_grid_s, start_abs

# 将低维连续序列线性插值到目标时间轴
# 输入：源时间戳（纳秒）、源值、目标绝对时间轴
# 输出：重采样后的二维数组
def linear_resample_to_grid(source_ts_ns, source_value, target_abs_grid_s):
    if source_ts_ns is None or source_value is None:
        return None

    source_s = ns_to_seconds(source_ts_ns)
    source_value = np.asarray(source_value)

    if source_value.ndim == 1:
        source_value = source_value.reshape(-1, 1)

    source_s, source_value = prepare_interp_source(source_s, source_value)

    # 只有一个点时无法插值，直接重复该点
    if len(source_s) == 1:
        return np.repeat(source_value, len(target_abs_grid_s), axis=0)

    n_time = len(target_abs_grid_s)
    n_dim = source_value.shape[1]

    output = np.full((n_time, n_dim), np.nan, dtype=float)

    # 只对源时间范围内的目标点进行插值；范围外保留 NaN
    valid_mask = (target_abs_grid_s >= source_s[0]) & (target_abs_grid_s <= source_s[-1])

    if not np.any(valid_mask):
        return output

    for j in range(n_dim):
        output[valid_mask, j] = np.interp(
            target_abs_grid_s[valid_mask],
            source_s,
            source_value[:, j]
        )

    return output

# 把单个时间戳序列匹配到 100Hz 动作网格
# 输入：
#   timestamp_ns     图像时间戳（纳秒）
#   rel_grid_s       相对时间网格
#   grid_start_abs   网格起点绝对秒
#   arm_action_100hz 对齐后的 arm 动作
# 输出：
#   每个有效时间点的最近动作索引与动作标签
def match_timestamps_to_actions(timestamp_ns, rel_grid_s, grid_start_abs, arm_action_100hz):
    if timestamp_ns is None:
        return {
            "valid_time_s": np.array([]),
            "nearest_action_index": np.array([], dtype=int),
            "nearest_action_time_s": np.array([]),
            "matched_arm_action": np.empty((0, arm_action_100hz.shape[1])),
            "offset_s": np.array([]),
        }

    abs_time_s = ns_to_seconds(timestamp_ns)
    rel_time_s = abs_time_s - grid_start_abs

    # 只保留落在公共时间轴区间内的图像帧
    valid_mask = (rel_time_s >= rel_grid_s[0]) & (rel_time_s <= rel_grid_s[-1])
    rel_time_s = rel_time_s[valid_mask]

    if len(rel_time_s) == 0:
        return {
            "valid_time_s": np.array([]),
            "nearest_action_index": np.array([], dtype=int),
            "nearest_action_time_s": np.array([]),
            "matched_arm_action": np.empty((0, arm_action_100hz.shape[1])),
            "offset_s": np.array([]),
        }

    # 用 searchsorted 找最近的网格点
    idx_right = np.searchsorted(rel_grid_s, rel_time_s, side="left")
    idx_left = np.clip(idx_right - 1, 0, len(rel_grid_s) - 1)
    idx_right = np.clip(idx_right, 0, len(rel_grid_s) - 1)

    left_dist = np.abs(rel_time_s - rel_grid_s[idx_left])
    right_dist = np.abs(rel_time_s - rel_grid_s[idx_right])

    nearest_idx = np.where(left_dist <= right_dist, idx_left, idx_right)

    nearest_time_s = rel_grid_s[nearest_idx]
    offset_s = np.abs(rel_time_s - nearest_time_s)
    matched_action = arm_action_100hz[nearest_idx]

    return {
        "valid_time_s": rel_time_s,
        "nearest_action_index": nearest_idx,
        "nearest_action_time_s": nearest_time_s,
        "matched_arm_action": matched_action,
        "offset_s": offset_s,
    }

# 对齐单条轨迹
# 输入：单条轨迹原始字典
# 输出：对齐后的轨迹字典
def align_one_trajectory(traj, target_hz=100.0):
    trajectory_id = traj["trajectory_id"]

    # 以 arm 为主时间基准
    arm_action = traj["joints_action"].get(CONFIG["main_action_group"])
    arm_state = traj["joints_state"].get(CONFIG["main_state_group"])

    if arm_action is None or arm_state is None:
        raise ValueError(f"{trajectory_id} 缺少 arm action/state，无法对齐。")

    if arm_action["position"] is None or arm_state["position"] is None:
        raise ValueError(f"{trajectory_id} 的 arm position 为空，无法对齐。")

    if arm_action["timestamp_ns"] is None or arm_state["timestamp_ns"] is None:
        raise ValueError(f"{trajectory_id} 的 arm timestamp 为空，无法对齐。")

    # 1. 构建公共 100Hz 时间轴
    rel_grid_s, abs_grid_s, grid_start_abs = build_common_time_grid(
        arm_action["timestamp_ns"],
        arm_state["timestamp_ns"],
        target_hz=target_hz,
    )

    # 2. arm 主序列对齐
    arm_action_100hz = linear_resample_to_grid(
        arm_action["timestamp_ns"],
        arm_action["position"],
        abs_grid_s,
    )
    arm_state_100hz = linear_resample_to_grid(
        arm_state["timestamp_ns"],
        arm_state["position"],
        abs_grid_s,
    )

    # 3. effector 辅助序列对齐到同一时间轴
    effector_action = traj["joints_action"].get(CONFIG["aux_group"])
    effector_state = traj["joints_state"].get(CONFIG["aux_group"])

    effector_action_100hz = None
    effector_state_100hz = None

    if effector_action is not None and effector_action["position"] is not None and effector_action["timestamp_ns"] is not None:
        effector_action_100hz = linear_resample_to_grid(
            effector_action["timestamp_ns"],
            effector_action["position"],
            abs_grid_s,
        )

    if effector_state is not None and effector_state["position"] is not None and effector_state["timestamp_ns"] is not None:
        effector_state_100hz = linear_resample_to_grid(
            effector_state["timestamp_ns"],
            effector_state["position"],
            abs_grid_s,
        )

    # 4. 三路相机的 color/depth 时间戳分别匹配最近动作
    camera_matches = {}

    for cam_name, cam in traj["cameras"].items():
        camera_matches[cam_name] = {
            "color": match_timestamps_to_actions(
                cam["color_timestamp_ns"],
                rel_grid_s,
                grid_start_abs,
                arm_action_100hz,
            ),
            "depth": match_timestamps_to_actions(
                cam["depth_timestamp_ns"],
                rel_grid_s,
                grid_start_abs,
                arm_action_100hz,
            ),
        }

    aligned = {
        "trajectory_id": trajectory_id,
        "scene_id": traj["scene_id"],
        "task_name": traj["task_name"],

        # 100Hz 公共时间轴
        "time_grid_s": rel_grid_s,

        # 对齐后的低维序列
        "arm_action_100hz": arm_action_100hz,
        "arm_state_100hz": arm_state_100hz,
        "effector_action_100hz": effector_action_100hz,
        "effector_state_100hz": effector_state_100hz,

        # 图像-动作匹配结果
        "camera_matches": camera_matches,
    }

    return aligned

# 生成单条轨迹的对齐摘要
# 输入：对齐后的轨迹字典
# 输出：一行摘要信息
def summarize_aligned_trajectory(aligned):
    row = {
        "trajectory_id": aligned["trajectory_id"],
        "scene_id": aligned["scene_id"],
        "task_name": aligned["task_name"],
        "aligned_len_100hz": len(aligned["time_grid_s"]),
        "aligned_duration_s": float(aligned["time_grid_s"][-1]) if len(aligned["time_grid_s"]) > 0 else 0.0,

        "arm_action_dim": aligned["arm_action_100hz"].shape[1],
        "arm_state_dim": aligned["arm_state_100hz"].shape[1],
    }

    if aligned["effector_action_100hz"] is not None:
        row["effector_action_dim"] = aligned["effector_action_100hz"].shape[1]
        row["effector_action_nan_ratio"] = float(np.isnan(aligned["effector_action_100hz"]).mean())
    else:
        row["effector_action_dim"] = None
        row["effector_action_nan_ratio"] = None

    if aligned["effector_state_100hz"] is not None:
        row["effector_state_dim"] = aligned["effector_state_100hz"].shape[1]
        row["effector_state_nan_ratio"] = float(np.isnan(aligned["effector_state_100hz"]).mean())
    else:
        row["effector_state_dim"] = None
        row["effector_state_nan_ratio"] = None

    # 三路相机匹配摘要
    for cam_name, cam_match in aligned["camera_matches"].items():
        color_match = cam_match["color"]
        depth_match = cam_match["depth"]

        row[f"{cam_name}_color_matched_n"] = len(color_match["valid_time_s"])
        row[f"{cam_name}_depth_matched_n"] = len(depth_match["valid_time_s"])

        row[f"{cam_name}_color_mean_offset_ms"] = (
            float(color_match["offset_s"].mean() * 1000.0)
            if len(color_match["offset_s"]) > 0 else None
        )
        row[f"{cam_name}_depth_mean_offset_ms"] = (
            float(depth_match["offset_s"].mean() * 1000.0)
            if len(depth_match["offset_s"]) > 0 else None
        )

    return row

# 保存对齐结果
# 输入：aligned_dict、summary_df
# 输出：无
def save_alignment_outputs(aligned_dict, summary_df):
    # 保存详细对齐结果
    aligned_save_path = os.path.join(CONFIG["interim_dir"], "s3_aligned_data.pkl")
    with open(aligned_save_path, "wb") as f:
        pickle.dump(aligned_dict, f)

    # 保存摘要表
    summary_save_path = os.path.join(CONFIG["interim_dir"], "s3_align_summary.csv")
    save_csv(summary_df, summary_save_path)

# 运行阶段一第2步
# 输入：无
# 输出：aligned_result_dict, summary_df
def run_stage1_step2():
    manifest_path = os.path.join(CONFIG["interim_dir"], "s3_manifest.csv")

    # 如果第一步结果不存在，就先自动补跑第一步
    if not os.path.exists(manifest_path):
        run_stage1_step1()

    manifest_df = pd.read_csv(manifest_path)

    aligned_result_dict = {}
    summary_rows = []

    for _, row in manifest_df.iterrows():
        file_path = row["file_path"]
        traj = read_one_h5(file_path)

        aligned = align_one_trajectory(traj, target_hz=100.0)
        aligned_result_dict[aligned["trajectory_id"]] = aligned

        summary_rows.append(summarize_aligned_trajectory(aligned))

    summary_df = pd.DataFrame(summary_rows)
    save_alignment_outputs(aligned_result_dict, summary_df)

    return aligned_result_dict, summary_df
