# 文件位置：stage1_label.py

import os
import pickle

import numpy as np
import pandas as pd

from config import CONFIG
from utils import save_csv, save_json


# 读取第二步保存的对齐结果
# 输入：无
# 输出：aligned_result_dict
def load_aligned_data():
    aligned_path = os.path.join(CONFIG["interim_dir"], "s3_aligned_data.pkl")

    if not os.path.exists(aligned_path):
        raise FileNotFoundError(
            f"未找到对齐结果文件：{aligned_path}，请先运行阶段一第2步。"
        )

    with open(aligned_path, "rb") as f:
        aligned_result_dict = pickle.load(f)

    return aligned_result_dict


# 计算一个序列在前段和后段的均值、后段稳定性
# 输入：
#   seq: 二维数组，形状 (T, D)
#   seg_ratio: 取前后各多少比例作为起始段和结束段
# 输出：
#   start_mean, end_mean, end_std
def get_start_end_stats(seq, seg_ratio=0.1):
    seq = np.asarray(seq, dtype=float)

    if seq.ndim != 2:
        raise ValueError("输入序列必须是二维数组。")

    n = len(seq)
    seg_n = max(10, int(n * seg_ratio))
    seg_n = min(seg_n, n)

    start_part = seq[:seg_n]
    end_part = seq[-seg_n:]

    start_mean = np.nanmean(start_part, axis=0)
    end_mean = np.nanmean(end_part, axis=0)
    end_std = np.nanstd(end_part, axis=0)

    return start_mean, end_mean, end_std


# 计算布尔序列中最长连续 True 片段长度
# 输入：一维布尔数组
# 输出：最长连续 True 的长度（单位：帧）
def longest_true_segment(mask):
    max_len = 0
    curr_len = 0

    for v in mask:
        if v:
            curr_len += 1
            if curr_len > max_len:
                max_len = curr_len
        else:
            curr_len = 0

    return int(max_len)


# 从 effector 序列中提取“中间是否发生抓握事件”的信号
# 输入：
#   effector_state: 形状 (T, D)
# 输出：
#   抓握事件相关信号字典
def compute_effector_event_signals(effector_state):
    effector_state = np.asarray(effector_state, dtype=float)

    # 基础防御：空数组或维度不对时，直接返回0信号
    if effector_state.ndim != 2 or effector_state.size == 0:
        return {
            "effector_peak_offset_l1": 0.0,
            "effector_active_ratio": 0.0,
            "stable_grasp_segment_len": 0,
            "stable_grasp_segment_s": 0.0,
            "effector_range_l1": 0.0,
        }

    # 去掉“整列全是 NaN”的通道，避免后续统计被污染
    valid_col_mask = np.any(np.isfinite(effector_state), axis=0)
    if not np.any(valid_col_mask):
        return {
            "effector_peak_offset_l1": 0.0,
            "effector_active_ratio": 0.0,
            "stable_grasp_segment_len": 0,
            "stable_grasp_segment_s": 0.0,
            "effector_range_l1": 0.0,
        }

    effector_state = effector_state[:, valid_col_mask]

    # 标记“这一帧是否至少有一个有效维度”
    row_valid_mask = np.any(np.isfinite(effector_state), axis=1)
    if not np.any(row_valid_mask):
        return {
            "effector_peak_offset_l1": 0.0,
            "effector_active_ratio": 0.0,
            "stable_grasp_segment_len": 0,
            "stable_grasp_segment_s": 0.0,
            "effector_range_l1": 0.0,
        }

    # 用前10%作为初始参考姿态，但只作“参考中心”，不直接做首尾硬比较
    start_mean, _, _ = get_start_end_stats(effector_state, seg_ratio=0.1)

    # 如果起始段某些维度全为 NaN，用该维度全轨迹中位数兜底
    fallback_center = np.nanmedian(effector_state, axis=0)
    start_mean = np.where(np.isfinite(start_mean), start_mean, fallback_center)

    # 每一帧相对于初始参考姿态的平均绝对偏移
    # 对整行全 NaN 的帧，不参与均值计算，偏移直接记为 0
    offset_series = np.zeros(len(effector_state), dtype=float)
    offset_raw = np.abs(effector_state - start_mean.reshape(1, -1))

    valid_rows_for_offset = np.any(np.isfinite(offset_raw), axis=1)
    if np.any(valid_rows_for_offset):
        offset_series[valid_rows_for_offset] = np.nanmean(
            offset_raw[valid_rows_for_offset], axis=1
        )

    # 每一帧的速度幅值，用相邻帧差分近似
    # 对无效差分帧，不把它当成“稳定”，因此记为 +inf
    diff_seq = np.diff(effector_state, axis=0)
    speed_series = np.full(len(effector_state), np.inf, dtype=float)

    # 第一帧没有前向差分，如果这一帧有效，速度记为 0
    if row_valid_mask[0]:
        speed_series[0] = 0.0

    if len(diff_seq) > 0:
        diff_valid_mask = np.any(np.isfinite(diff_seq), axis=1)

        if np.any(diff_valid_mask):
            valid_idx = np.where(diff_valid_mask)[0] + 1
            speed_values = np.nanmean(np.abs(diff_seq[diff_valid_mask]), axis=1)
            speed_series[valid_idx] = speed_values

    # 全程平均开合幅度
    range_l1 = float(
        np.nanmean(np.nanmax(effector_state, axis=0) - np.nanmin(effector_state, axis=0))
    )

    # 只在有效行上计算峰值
    peak_offset_l1 = float(np.nanmax(offset_series[row_valid_mask]))

    # 如果峰值太小，说明几乎没有真实动作
    if peak_offset_l1 <= 1e-8:
        return {
            "effector_peak_offset_l1": 0.0,
            "effector_active_ratio": 0.0,
            "stable_grasp_segment_len": 0,
            "stable_grasp_segment_s": 0.0,
            "effector_range_l1": range_l1,
        }

    # “明显偏离初始状态”的帧：偏移至少达到峰值的 60%
    moved_mask = np.zeros(len(effector_state), dtype=bool)
    moved_mask[row_valid_mask] = offset_series[row_valid_mask] >= (0.6 * peak_offset_l1)

    # “相对稳定”的帧：速度处于该轨迹内部较低水平
    finite_speed = speed_series[np.isfinite(speed_series)]
    if len(finite_speed) == 0:
        speed_threshold = 0.0
    else:
        speed_threshold = float(np.percentile(finite_speed, 40))

    stable_mask = np.isfinite(speed_series) & (speed_series <= speed_threshold)

    # 候选抓握片段：既明显偏移，又相对稳定，同时这一帧必须有效
    grasp_like_mask = row_valid_mask & moved_mask & stable_mask

    stable_grasp_segment_len = longest_true_segment(grasp_like_mask)

    # active_ratio 只在有效帧上统计，更公平
    effector_active_ratio = float(np.mean(moved_mask[row_valid_mask]))

    # 当前对齐步长固定为 0.01s
    stable_grasp_segment_s = stable_grasp_segment_len * 0.01

    return {
        "effector_peak_offset_l1": peak_offset_l1,
        "effector_active_ratio": effector_active_ratio,
        "stable_grasp_segment_len": stable_grasp_segment_len,
        "stable_grasp_segment_s": stable_grasp_segment_s,
        "effector_range_l1": range_l1,
    }


# 计算单条轨迹的自动标签信号
# 输入：对齐后的单条轨迹字典
# 输出：一行信号字典
def compute_label_signals(aligned):
    trajectory_id = aligned["trajectory_id"]
    time_grid_s = aligned["time_grid_s"]

    arm_state = aligned["arm_state_100hz"]
    effector_state = aligned["effector_state_100hz"]

    # 基础长度与时长
    aligned_len = len(time_grid_s)
    duration_s = float(time_grid_s[-1]) if aligned_len > 0 else 0.0

    # arm 信号：保留“是否明显动过”这一类运动学证据
    arm_start_mean, arm_end_mean, arm_end_std = get_start_end_stats(arm_state, seg_ratio=0.1)

    arm_change_l1 = float(np.nanmean(np.abs(arm_end_mean - arm_start_mean)))
    arm_range_l1 = float(np.nanmean(np.nanmax(arm_state, axis=0) - np.nanmin(arm_state, axis=0)))
    arm_final_stability = float(np.nanmean(arm_end_std))

    # effector 信号：不再只看首尾差，而是看“中间是否出现过抓握事件”
    if effector_state is not None:
        eff_start_mean, eff_end_mean, eff_end_std = get_start_end_stats(effector_state, seg_ratio=0.1)

        # 这个字段保留为诊断字段，但不再作为硬判定依据
        effector_change_l1 = float(np.nanmean(np.abs(eff_end_mean - eff_start_mean)))
        effector_final_stability = float(np.nanmean(eff_end_std))
        effector_nan_ratio = float(np.isnan(effector_state).mean())

        event_signals = compute_effector_event_signals(effector_state)

        effector_range_l1 = event_signals["effector_range_l1"]
        effector_peak_offset_l1 = event_signals["effector_peak_offset_l1"]
        effector_active_ratio = event_signals["effector_active_ratio"]
        stable_grasp_segment_len = event_signals["stable_grasp_segment_len"]
        stable_grasp_segment_s = event_signals["stable_grasp_segment_s"]
    else:
        effector_change_l1 = np.nan
        effector_range_l1 = np.nan
        effector_final_stability = np.nan
        effector_nan_ratio = np.nan
        effector_peak_offset_l1 = np.nan
        effector_active_ratio = np.nan
        stable_grasp_segment_len = np.nan
        stable_grasp_segment_s = np.nan

    row = {
        "trajectory_id": trajectory_id,
        "aligned_len_100hz": aligned_len,
        "duration_s": duration_s,

        "arm_change_l1": arm_change_l1,
        "arm_range_l1": arm_range_l1,
        "arm_final_stability": arm_final_stability,

        # 旧字段保留，便于对比，但不再作为核心成功依据
        "effector_change_l1": effector_change_l1,
        "effector_final_stability": effector_final_stability,
        "effector_nan_ratio": effector_nan_ratio,

        # 新的核心抓握事件字段
        "effector_range_l1": effector_range_l1,
        "effector_peak_offset_l1": effector_peak_offset_l1,
        "effector_active_ratio": effector_active_ratio,
        "stable_grasp_segment_len": stable_grasp_segment_len,
        "stable_grasp_segment_s": stable_grasp_segment_s,
    }

    return row


# 根据所有轨迹的信号，估计自动判定阈值
# 输入：label_signals_df
# 输出：阈值字典
def estimate_label_thresholds(label_signals_df):
    df = label_signals_df.copy()

    thresholds = {
        # 轨迹太短一般不给自动成功
        "min_duration_s": max(1.5, float(np.nanpercentile(df["duration_s"], 10))),

        # arm 至少明显动过
        "min_arm_range_l1": float(np.nanpercentile(df["arm_range_l1"], 30)),

        # effector 至少出现过明显开合/偏移
        "min_effector_range_l1": float(np.nanpercentile(df["effector_range_l1"], 35)),
        "min_effector_peak_offset_l1": float(np.nanpercentile(df["effector_peak_offset_l1"], 35)),

        # 抓握样稳定片段至少要持续一小段时间
        # 用秒更直观，避免不同长度轨迹直接比帧数
        "min_stable_grasp_segment_s": max(
            0.20,
            float(np.nanpercentile(df["stable_grasp_segment_s"], 35))
        ),
    }

    return thresholds


# 对单条轨迹应用自动判定规则
# 输入：一行信号、阈值字典
# 输出：auto_label、review_needed、reason
def apply_auto_label_rule(signal_row, thresholds):
    reasons = []

    # 新逻辑：
    # 1. 轨迹够长
    # 2. arm 有明显运动
    # 3. effector 全程确实发生过明显开合/偏移
    # 4. 中后段某处存在持续稳定片段
    duration_ok = signal_row["duration_s"] >= thresholds["min_duration_s"]
    arm_move_ok = signal_row["arm_range_l1"] >= thresholds["min_arm_range_l1"]
    eff_range_ok = signal_row["effector_range_l1"] >= thresholds["min_effector_range_l1"]
    eff_peak_ok = signal_row["effector_peak_offset_l1"] >= thresholds["min_effector_peak_offset_l1"]
    stable_segment_ok = signal_row["stable_grasp_segment_s"] >= thresholds["min_stable_grasp_segment_s"]

    if not duration_ok:
        reasons.append("duration_too_short")

    if not arm_move_ok:
        reasons.append("arm_motion_too_small")

    if not eff_range_ok:
        reasons.append("effector_range_too_small")

    if not eff_peak_ok:
        reasons.append("no_clear_effector_event")

    if not stable_segment_ok:
        reasons.append("no_stable_grasp_segment")

    # 所有规则都满足才给自动成功
    if duration_ok and arm_move_ok and eff_range_ok and eff_peak_ok and stable_segment_ok:
        auto_label = 1
        review_needed = False
        reason = "auto_success"
    else:
        auto_label = 0
        review_needed = True
        reason = "|".join(reasons)

    return auto_label, review_needed, reason


# 构建自动标签表
# 输入：label_signals_df、thresholds
# 输出：auto_label_df
def build_auto_labels(label_signals_df, thresholds):
    rows = []

    for _, row in label_signals_df.iterrows():
        auto_label, review_needed, reason = apply_auto_label_rule(row, thresholds)

        rows.append({
            "trajectory_id": row["trajectory_id"],
            "auto_label": auto_label,
            "review_needed": review_needed,
            "label_reason": reason,
        })

    auto_label_df = pd.DataFrame(rows)
    return auto_label_df


# 构建人工复核队列
# 输入：signals_df、auto_label_df
# 输出：review_queue_df
def build_review_queue(label_signals_df, auto_label_df):
    merged = pd.merge(label_signals_df, auto_label_df, on="trajectory_id", how="left")

    # 只导出 auto_label = 0 的样本供人工复核
    review_queue_df = merged[merged["auto_label"] == 0].copy()

    # 增加人工填写列
    review_queue_df["manual_label"] = np.nan
    review_queue_df["review_note"] = ""

    # 人工复核时最常看的字段放前面
    front_cols = [
        "trajectory_id",
        "auto_label",
        "label_reason",
        "manual_label",
        "review_note",
        "duration_s",
        "arm_range_l1",
        "effector_range_l1",
        "effector_peak_offset_l1",
        "stable_grasp_segment_len",
        "stable_grasp_segment_s",
        "effector_active_ratio",
        "effector_change_l1",
        "effector_final_stability",
    ]

    remain_cols = [col for col in review_queue_df.columns if col not in front_cols]
    review_queue_df = review_queue_df[front_cols + remain_cols]

    return review_queue_df


# 合并自动标签与人工复核标签
# 输入：auto_label_df、manual_review_path
# 输出：final_label_df
def merge_final_labels(auto_label_df, manual_review_path):
    final_df = auto_label_df.copy()
    final_df["manual_label"] = np.nan
    final_df["final_label"] = final_df["auto_label"]

    # 如果人工复核文件存在，则覆盖对应样本
    if os.path.exists(manual_review_path):
        manual_df = pd.read_csv(manual_review_path)

        # 只保留有效人工标签
        valid_manual = manual_df[manual_df["manual_label"].isin([0, 1])].copy()

        if len(valid_manual) > 0:
            final_df = pd.merge(
                final_df,
                valid_manual[["trajectory_id", "manual_label"]],
                on="trajectory_id",
                how="left",
                suffixes=("", "_new"),
            )

            # 修改这里 #
            # 用人工标签覆盖自动标签
            final_df["manual_label"] = final_df["manual_label_new"].combine_first(final_df["manual_label"])
            final_df["final_label"] = final_df["manual_label"].combine_first(final_df["auto_label"])
            final_df = final_df.drop(columns=["manual_label_new"])
            # 到这里 #

    return final_df


# 保存第三步输出文件
# 输入：signals_df、auto_label_df、review_queue_df、final_label_df、thresholds
# 输出：无
def save_step3_outputs(signals_df, auto_label_df, review_queue_df, final_label_df, thresholds):
    signals_path = os.path.join(CONFIG["interim_dir"], "s3_label_signals.csv")
    auto_label_path = os.path.join(CONFIG["interim_dir"], "s3_auto_labels.csv")
    review_queue_path = os.path.join(CONFIG["interim_dir"], "s3_review_queue.csv")
    manual_template_path = os.path.join(CONFIG["interim_dir"], "s3_manual_review_template.csv")
    final_label_path = os.path.join(CONFIG["interim_dir"], "s3_final_labels.csv")
    threshold_path = os.path.join(CONFIG["interim_dir"], "s3_label_thresholds.json")

    save_csv(signals_df, signals_path)
    save_csv(auto_label_df, auto_label_path)
    save_csv(review_queue_df, review_queue_path)
    save_csv(review_queue_df[["trajectory_id", "auto_label", "manual_label", "review_note"]], manual_template_path)
    save_csv(final_label_df, final_label_path)
    save_json(thresholds, threshold_path)


# 运行阶段一第3步
# 输入：无
# 输出：signals_df, auto_label_df, review_queue_df, final_label_df
def run_stage1_step3():
    aligned_result_dict = load_aligned_data()

    # 1. 计算每条轨迹的判定信号
    signal_rows = []
    for trajectory_id in sorted(aligned_result_dict.keys()):
        aligned = aligned_result_dict[trajectory_id]
        signal_rows.append(compute_label_signals(aligned))

    signals_df = pd.DataFrame(signal_rows)

    # 2. 基于全体轨迹估计阈值
    thresholds = estimate_label_thresholds(signals_df)

    # 3. 自动标签
    auto_label_df = build_auto_labels(signals_df, thresholds)

    # 4. 人工复核队列
    review_queue_df = build_review_queue(signals_df, auto_label_df)

    # 5. 合并 final label
    manual_review_path = os.path.join(CONFIG["interim_dir"], "s3_manual_review.csv")
    final_label_df = merge_final_labels(auto_label_df, manual_review_path)

    # 6. 保存结果
    save_step3_outputs(
        signals_df=signals_df,
        auto_label_df=auto_label_df,
        review_queue_df=review_queue_df,
        final_label_df=final_label_df,
        thresholds=thresholds,
    )

    return signals_df, auto_label_df, review_queue_df, final_label_df