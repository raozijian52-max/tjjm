import json
import os
import pickle
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from config import CONFIG
from utils import save_csv, save_json


DIMENSION_MAP = {
    "completeness": ["C3", "C4", "C5", "C6"],
    "accuracy": ["C8", "C10", "C11"],
    "diversity": ["C12", "C13", "C14", "C15"],
    "consistency": ["C16", "C17", "C18", "C19"],
    "usability": ["C21", "C22", "C23"],
}


INDICATOR_DIRECTIONS = {
    "C3_visual_completeness": 1,
    "C4_depth_validity": 1,
    "C5_joint_completeness": 1,
    "C6_attribute_completeness": 1,
    "C8_joint_anomaly_quality": 1,
    "C10_joint_noise_quality": 1,
    "C11_timestamp_consistency": 1,
    "C12_scene_entropy": 1,
    "C13_object_diversity": 1,
    "C14_atomic_skill_diversity": 1,
    "C15_motion_mode_diversity": 1,
    "C16_multimodal_alignment": 1,
    "C17_visual_joint_mi": 1,
    "C18_joint_coordination": 1,
    "C19_duplicate_uniqueness": 1,
    "C21_label_completeness": 1,
    "C22_metadata_standardization": 1,
    "C23_scene_description_completeness": 1,
}


def _safe_ratio(numerator, denominator):
    """安全计算比例，并把结果限制在 0 到 1 之间。"""
    if denominator is None or denominator <= 0:
        return np.nan
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _safe_mean(values):
    """计算列表均值，自动跳过 NaN 或空值。"""
    values = [v for v in values if pd.notna(v)]
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def _finite_array(seq):
    """把输入序列统一转成二维浮点数组，方便后续按时间维处理。"""
    arr = np.asarray(seq, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def uniform_sample_indices(n_total, max_samples):
    """从总帧数里均匀抽取最多 max_samples 个索引。"""
    if n_total <= 0:
        return np.array([], dtype=int)
    if n_total <= max_samples:
        return np.arange(n_total, dtype=int)
    return np.linspace(0, n_total - 1, num=max_samples, dtype=int)


def decode_image_bytes(img_bytes, mode):
    """把 H5 里保存的图片字节流解码成 numpy 图像数组。"""
    if img_bytes is None or len(img_bytes) == 0:
        return None
    try:
        with Image.open(BytesIO(img_bytes)) as img:
            if mode == "color":
                img = img.convert("RGB")
            return np.array(img)
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def simple_kmeans(X, n_clusters, random_state=42, max_iter=100):
    """用 numpy 实现一个轻量版 KMeans，避免额外依赖 sklearn。"""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(random_state)
    if len(X) <= n_clusters:
        return np.arange(len(X), dtype=int)

    centers = X[rng.choice(len(X), size=n_clusters, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centers[k] = X[mask].mean(axis=0)

    return labels


def normalized_mutual_information(labels_a, labels_b):
    """计算两个离散序列的归一化互信息，用于衡量视觉变化和动作变化的关联。"""
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if len(labels_a) == 0 or len(labels_a) != len(labels_b):
        return np.nan

    a_values, a_inverse = np.unique(labels_a, return_inverse=True)
    b_values, b_inverse = np.unique(labels_b, return_inverse=True)
    contingency = np.zeros((len(a_values), len(b_values)), dtype=float)

    for a, b in zip(a_inverse, b_inverse):
        contingency[a, b] += 1.0

    total = contingency.sum()
    if total <= 0:
        return np.nan

    p_ab = contingency / total
    p_a = p_ab.sum(axis=1)
    p_b = p_ab.sum(axis=0)

    expected = p_a[:, None] * p_b[None, :]
    valid = p_ab > 0
    mi = float(np.sum(p_ab[valid] * np.log(p_ab[valid] / expected[valid])))

    h_a = float(-np.sum(p_a[p_a > 0] * np.log(p_a[p_a > 0])))
    h_b = float(-np.sum(p_b[p_b > 0] * np.log(p_b[p_b > 0])))
    denom = (h_a + h_b) / 2.0
    if denom <= 1e-12:
        return 0.0
    return float(mi / denom)


def _robust_zscore(values):
    """计算稳健 z-score，用中位数和 MAD 降低极端值影响。"""
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if not np.isfinite(mad) or mad <= 1e-12:
        std = np.nanstd(values)
        if not np.isfinite(std) or std <= 1e-12:
            return np.zeros_like(values, dtype=float)
        return (values - np.nanmean(values)) / std
    return 0.6745 * (values - median) / mad


def load_stage2_inputs():
    """读取阶段2所需的阶段1输出文件。"""
    manifest_path = os.path.join(CONFIG["interim_dir"], "s3_manifest.csv")
    metadata_path = os.path.join(CONFIG["interim_dir"], "s3_raw_metadata.csv")
    aligned_path = os.path.join(CONFIG["interim_dir"], "s3_aligned_data.pkl")
    feature_path = os.path.join(CONFIG["interim_dir"], "s3_feature_matrix.csv")
    final_label_path = os.path.join(CONFIG["interim_dir"], "s3_final_labels.csv")

    missing = [
        path for path in [manifest_path, metadata_path, aligned_path, feature_path, final_label_path]
        if not os.path.exists(path)
    ]
    if missing:
        raw_dir = CONFIG["raw_dir"]
        raw_h5_count = 0
        if os.path.isdir(raw_dir):
            raw_h5_count = len([name for name in os.listdir(raw_dir) if name.endswith(".h5")])

        raise FileNotFoundError(
            "Missing stage1 outputs required by stage2: "
            + ", ".join(missing)
            + "\nPlease put raw .h5 files under "
            + raw_dir
            + f" first. Current .h5 count there: {raw_h5_count}."
            + "\nThen run: python openlet_project/run_stage1.py"
            + "\nAfter stage1 finishes, run: python openlet_project/run_stage2.py"
        )

    manifest_df = pd.read_csv(manifest_path)
    metadata_df = pd.read_csv(metadata_path)
    feature_df = pd.read_csv(feature_path)
    final_label_df = pd.read_csv(final_label_path)

    with open(aligned_path, "rb") as f:
        aligned_dict = pickle.load(f)

    return manifest_df, metadata_df, feature_df, final_label_df, aligned_dict


def compute_visual_completeness(aligned, expected_fps=30.0):
    """C3：计算视觉帧完整性，即实际匹配帧数和理论 30fps 帧数的比例。"""
    duration_s = float(aligned["time_grid_s"][-1]) if len(aligned["time_grid_s"]) > 0 else 0.0
    expected_frames = max(1.0, duration_s * expected_fps)
    ratios = []

    for cam_match in aligned.get("camera_matches", {}).values():
        color_match = cam_match.get("color", {})
        matched_n = len(color_match.get("valid_time_s", []))
        ratios.append(_safe_ratio(matched_n, expected_frames))

    return _safe_mean(ratios)


def compute_depth_validity(feature_row):
    """C4：根据深度图空洞率计算深度图有效性，空洞越少分数越高。"""
    hole_cols = [col for col in feature_row.index if col.endswith("_depth_hole_ratio")]
    hole_ratio = _safe_mean([feature_row[col] for col in hole_cols])
    if pd.isna(hole_ratio):
        return np.nan
    return float(np.clip(1.0 - hole_ratio, 0.0, 1.0))


def compute_joint_completeness(aligned):
    """C5：计算关节数据完整性，综合时间轴长度和 NaN 缺失情况。"""
    time_grid = np.asarray(aligned["time_grid_s"], dtype=float)
    duration_s = float(time_grid[-1]) if len(time_grid) > 0 else 0.0
    expected_len = int(np.floor(duration_s * 100.0)) + 1
    length_ratio = _safe_ratio(len(time_grid), expected_len)

    nan_ratios = []
    for key in ["arm_action_100hz", "arm_state_100hz", "effector_action_100hz", "effector_state_100hz"]:
        arr = aligned.get(key)
        if arr is not None:
            arr = np.asarray(arr, dtype=float)
            nan_ratios.append(float(np.isnan(arr).mean()))

    nan_quality = 1.0 - _safe_mean(nan_ratios) if len(nan_ratios) > 0 else np.nan
    return _safe_mean([length_ratio, nan_quality])


def compute_attribute_completeness(manifest_row, metadata_row):
    """C6：检查轨迹清单和元数据中关键字段是否齐全。"""
    required_values = [
        manifest_row.get("scene_id"),
        manifest_row.get("task_name"),
        manifest_row.get("object_id"),
        manifest_row.get("trajectory_id"),
        metadata_row.get("manufacturer"),
        metadata_row.get("equipment_model"),
        metadata_row.get("created_at"),
        metadata_row.get("collection_duration_in_ms"),
    ]
    valid = [pd.notna(v) and str(v).strip() != "" for v in required_values]
    return float(np.mean(valid))


def compute_joint_anomaly_quality(aligned):
    """C8：用速度、加速度、jerk 的异常比例估计关节数据质量。"""
    seq = _finite_array(aligned["arm_state_100hz"])
    if len(seq) < 4:
        return np.nan

    velocity = np.nanmean(np.abs(np.diff(seq, axis=0)), axis=1)
    acceleration = np.nanmean(np.abs(np.diff(seq, n=2, axis=0)), axis=1)
    jerk = np.nanmean(np.abs(np.diff(seq, n=3, axis=0)), axis=1)

    anomaly_masks = []
    for signal in [velocity, acceleration, jerk]:
        z = np.abs(_robust_zscore(signal))
        anomaly_masks.append(z > 3.5)

    max_len = max(len(mask) for mask in anomaly_masks)
    padded = []
    for mask in anomaly_masks:
        if len(mask) < max_len:
            mask = np.pad(mask, (max_len - len(mask), 0), constant_values=False)
        padded.append(mask)

    anomaly_ratio = float(np.mean(np.any(np.vstack(padded), axis=0)))
    return float(np.clip(1.0 - anomaly_ratio, 0.0, 1.0))


def compute_joint_noise_quality(aligned):
    """C10：用二阶差分的高频能量占比估计关节噪声，噪声越小分数越高。"""
    seq = _finite_array(aligned["arm_state_100hz"])
    if len(seq) < 4:
        return np.nan

    centered = seq - np.nanmean(seq, axis=0, keepdims=True)
    total_energy = float(np.nansum(centered ** 2))
    high_freq = np.diff(seq, n=2, axis=0)
    high_freq_energy = float(np.nansum(high_freq ** 2))

    if total_energy <= 1e-12:
        return 1.0

    noise_ratio = high_freq_energy / (high_freq_energy + total_energy)
    return float(np.clip(1.0 - noise_ratio, 0.0, 1.0))


def _timestamp_consistency_one(timestamp_ns):
    """计算单个时间戳序列的单调性和连续性。"""
    if timestamp_ns is None:
        return np.nan
    ts = np.asarray(timestamp_ns, dtype=np.int64)
    if len(ts) < 2:
        return np.nan
    diffs = np.diff(ts)
    monotonic_ratio = float(np.mean(diffs > 0))
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return 0.0
    median_dt = np.median(positive)
    if median_dt <= 0:
        return monotonic_ratio
    continuous_ratio = float(np.mean((diffs > 0) & (diffs <= 3.0 * median_dt)))
    return _safe_mean([monotonic_ratio, continuous_ratio])


def compute_timestamp_consistency(file_path):
    """C11：回读原始 H5，综合检查关节和相机时间戳是否连续递增。"""
    try:
        from stage1_read_and_manifest import read_one_h5

        traj = read_one_h5(file_path)
    except Exception:
        return np.nan

    scores = []
    for group_dict in [traj.get("joints_action", {}), traj.get("joints_state", {})]:
        for joint_group in group_dict.values():
            scores.append(_timestamp_consistency_one(joint_group.get("timestamp_ns")))

    for camera in traj.get("cameras", {}).values():
        scores.append(_timestamp_consistency_one(camera.get("color_timestamp_ns")))
        scores.append(_timestamp_consistency_one(camera.get("depth_timestamp_ns")))

    return _safe_mean(scores)


def compute_dataset_entropy(series):
    """计算某个类别字段在全数据集上的归一化熵。"""
    values = series.dropna().astype(str)
    if len(values) == 0:
        return np.nan
    probs = values.value_counts(normalize=True).to_numpy(dtype=float)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    max_entropy = np.log(max(len(probs), 2))
    return float(entropy / max_entropy)


def compute_object_diversity(manifest_df):
    """C13：计算操作物体多样性；当前单物体场景通常接近 0。"""
    if "object_id" not in manifest_df or len(manifest_df) == 0:
        return np.nan
    unique_n = manifest_df["object_id"].dropna().nunique()
    return float((unique_n - 1) / max(len(manifest_df) - 1, 1))


def compute_atomic_skill_diversity(feature_row):
    """C14：根据阶段1的离散语义状态占比，计算原子技能多样性。"""
    ratio_cols = [
        "semantic_state_ratio_0",
        "semantic_state_ratio_1",
        "semantic_state_ratio_2",
    ]
    ratios = np.array([feature_row.get(col, np.nan) for col in ratio_cols], dtype=float)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if len(ratios) == 0:
        return np.nan
    entropy = -np.sum(ratios * np.log(ratios + 1e-12))
    return float(entropy / np.log(3.0))


def build_motion_mode_diversity(feature_df):
    """C15：对关节运动特征聚类，用轨迹所在簇的稀有度表示运动模式多样性。"""
    feature_cols = [
        col for col in feature_df.columns
        if col.startswith("arm_state_range_")
        or col.startswith("arm_state_vel_abs_mean_")
        or col.startswith("arm_action_range_")
        or col.startswith("arm_action_vel_abs_mean_")
    ]
    if len(feature_cols) == 0 or len(feature_df) < 2:
        return pd.Series(np.nan, index=feature_df["trajectory_id"])

    X = feature_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    n_clusters = min(4, max(1, int(np.sqrt(len(X)))))
    if n_clusters <= 1:
        return pd.Series(0.0, index=feature_df["trajectory_id"])

    labels = simple_kmeans(
        X.to_numpy(dtype=float),
        n_clusters=n_clusters,
        random_state=CONFIG["random_state"],
    )
    counts = pd.Series(labels).value_counts()
    rarity = np.array([1.0 - counts[label] / len(labels) for label in labels], dtype=float)
    return pd.Series(rarity, index=feature_df["trajectory_id"])


def compute_multimodal_alignment(aligned, max_allowed_offset_s=0.05):
    """C16：根据图像帧和最近动作帧的时间偏移计算多模态对齐度。"""
    offset_means = []
    for cam_match in aligned.get("camera_matches", {}).values():
        for stream_name in ["color", "depth"]:
            match = cam_match.get(stream_name, {})
            offsets = np.asarray(match.get("offset_s", []), dtype=float)
            if len(offsets) > 0:
                offset_means.append(float(np.nanmean(offsets)))

    mean_offset = _safe_mean(offset_means)
    if pd.isna(mean_offset):
        return np.nan
    return float(np.clip(1.0 - mean_offset / max_allowed_offset_s, 0.0, 1.0))


def _discretize_for_mi(values, n_bins=5):
    """把连续信号按分位数离散化，供互信息计算使用。"""
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or np.nanstd(values) <= 1e-12:
        return np.zeros(len(values), dtype=int)

    quantiles = np.nanpercentile(values, np.linspace(0, 100, n_bins + 1)[1:-1])
    quantiles = np.unique(quantiles)
    return np.digitize(values, quantiles)


def compute_visual_joint_mi(file_path, aligned, max_frames=80):
    """C17：用图像亮度变化和动作幅值变化的互信息近似视觉-关节一致性。"""
    try:
        from stage1_read_and_manifest import read_one_h5

        traj = read_one_h5(file_path)
    except Exception:
        return np.nan

    mi_scores = []
    for cam_name in CONFIG["camera_names"]:
        cam = traj.get("cameras", {}).get(cam_name)
        match = aligned.get("camera_matches", {}).get(cam_name, {}).get("color", {})
        if cam is None or cam.get("color_bytes") is None:
            continue

        brightness = []
        action_norm = []
        sample_idx = uniform_sample_indices(len(cam["color_bytes"]), max_frames)
        matched_actions = np.asarray(match.get("matched_arm_action", []), dtype=float)

        if len(matched_actions) == 0:
            continue

        for idx in sample_idx:
            if idx >= len(matched_actions):
                continue
            img = decode_image_bytes(cam["color_bytes"][idx], "color")
            if img is None:
                continue
            brightness.append(float(np.mean(img)))
            action_norm.append(float(np.linalg.norm(matched_actions[idx])))

        if len(brightness) >= 5 and np.nanstd(brightness) > 1e-12 and np.nanstd(action_norm) > 1e-12:
            binned_brightness = _discretize_for_mi(brightness)
            binned_action = _discretize_for_mi(action_norm)
            mi_scores.append(float(normalized_mutual_information(binned_brightness, binned_action)))

    return _safe_mean(mi_scores)


def compute_joint_coordination(aligned):
    """C18：用关节通道之间的平均相关性近似关节间运动协调性。"""
    seq = _finite_array(aligned["arm_state_100hz"])
    if len(seq) < 3 or seq.shape[1] < 2:
        return np.nan

    corr = np.corrcoef(seq, rowvar=False)
    if corr.ndim != 2:
        return np.nan

    upper = corr[np.triu_indices_from(corr, k=1)]
    upper = upper[np.isfinite(upper)]
    if len(upper) == 0:
        return np.nan
    return float(np.mean(np.abs(upper)))


def build_duplicate_uniqueness(feature_df):
    """C19：用特征空间最近邻距离估计轨迹是否重复，距离越大唯一性越高。"""
    numeric_cols = [
        col for col in feature_df.columns
        if col not in ["trajectory_id", "scene_id", "task_name"]
        and pd.api.types.is_numeric_dtype(feature_df[col])
    ]
    if len(feature_df) < 2 or len(numeric_cols) == 0:
        return pd.Series(np.nan, index=feature_df["trajectory_id"])

    X = feature_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    std = X.std(axis=0).replace(0, 1.0)
    X = (X - X.mean(axis=0)) / std

    X_np = X.to_numpy(dtype=float)
    distances = np.linalg.norm(X_np[:, None, :] - X_np[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest_dist = np.min(distances, axis=1)

    max_dist = np.nanmax(nearest_dist)
    if not np.isfinite(max_dist) or max_dist <= 1e-12:
        uniqueness = np.zeros(len(nearest_dist), dtype=float)
    else:
        uniqueness = nearest_dist / max_dist

    return pd.Series(uniqueness, index=feature_df["trajectory_id"])


def compute_label_completeness(final_label_row):
    """C21：检查该轨迹是否已经有最终成功/失败标签。"""
    if final_label_row is None:
        return 0.0
    value = final_label_row.get("final_label", np.nan)
    return float(pd.notna(value))


def compute_scene_description_completeness(manifest_row, metadata_row):
    """C23：检查场景、任务、物体、设备等描述字段的完整程度。"""
    fields = [
        manifest_row.get("scene_id"),
        manifest_row.get("task_name"),
        manifest_row.get("object_id"),
        metadata_row.get("resource_id"),
        metadata_row.get("owner"),
        metadata_row.get("region"),
        metadata_row.get("manufacturer"),
        metadata_row.get("equipment_model"),
        metadata_row.get("equipment_sn"),
    ]
    valid = [pd.notna(v) and str(v).strip() != "" for v in fields]
    return float(np.mean(valid))


def compute_quality_indicators(manifest_df, metadata_df, feature_df, final_label_df, aligned_dict):
    """汇总计算每条轨迹的 18 个阶段2原始质量指标。"""
    metadata_map = metadata_df.set_index("trajectory_id").to_dict(orient="index") if len(metadata_df) > 0 else {}
    feature_map = feature_df.set_index("trajectory_id").to_dict(orient="index") if len(feature_df) > 0 else {}
    label_map = final_label_df.set_index("trajectory_id").to_dict(orient="index") if len(final_label_df) > 0 else {}

    scene_entropy = compute_dataset_entropy(manifest_df["scene_id"]) if "scene_id" in manifest_df else np.nan
    object_diversity = compute_object_diversity(manifest_df)
    motion_mode_diversity = build_motion_mode_diversity(feature_df)
    duplicate_uniqueness = build_duplicate_uniqueness(feature_df)

    rows = []
    for _, manifest_row in manifest_df.iterrows():
        trajectory_id = manifest_row["trajectory_id"]
        aligned = aligned_dict.get(trajectory_id)
        if aligned is None:
            continue

        metadata_row = metadata_map.get(trajectory_id, {})
        feature_row = pd.Series(feature_map.get(trajectory_id, {}))
        label_row = label_map.get(trajectory_id)

        row = {
            "trajectory_id": trajectory_id,
            "scene_id": manifest_row.get("scene_id"),
            "task_name": manifest_row.get("task_name"),
            "C3_visual_completeness": compute_visual_completeness(aligned),
            "C4_depth_validity": compute_depth_validity(feature_row),
            "C5_joint_completeness": compute_joint_completeness(aligned),
            "C6_attribute_completeness": compute_attribute_completeness(manifest_row, metadata_row),
            "C8_joint_anomaly_quality": compute_joint_anomaly_quality(aligned),
            "C10_joint_noise_quality": compute_joint_noise_quality(aligned),
            "C11_timestamp_consistency": compute_timestamp_consistency(manifest_row.get("file_path")),
            "C12_scene_entropy": scene_entropy,
            "C13_object_diversity": object_diversity,
            "C14_atomic_skill_diversity": compute_atomic_skill_diversity(feature_row),
            "C15_motion_mode_diversity": motion_mode_diversity.get(trajectory_id, np.nan),
            "C16_multimodal_alignment": compute_multimodal_alignment(aligned),
            "C17_visual_joint_mi": compute_visual_joint_mi(manifest_row.get("file_path"), aligned),
            "C18_joint_coordination": compute_joint_coordination(aligned),
            "C19_duplicate_uniqueness": duplicate_uniqueness.get(trajectory_id, np.nan),
            "C21_label_completeness": compute_label_completeness(label_row),
            "C22_metadata_standardization": float(bool(manifest_row.get("is_uuid_file_name", False))),
            "C23_scene_description_completeness": compute_scene_description_completeness(manifest_row, metadata_row),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def minmax_normalize_indicators(indicator_df):
    """把 18 个原始指标统一 Min-Max 归一化到 0 到 1。"""
    norm_df = indicator_df[["trajectory_id", "scene_id", "task_name"]].copy()
    details = {}

    for col, direction in INDICATOR_DIRECTIONS.items():
        values = pd.to_numeric(indicator_df[col], errors="coerce").astype(float)
        col_min = float(np.nanmin(values)) if np.any(np.isfinite(values)) else np.nan
        col_max = float(np.nanmax(values)) if np.any(np.isfinite(values)) else np.nan

        if not np.isfinite(col_min) or not np.isfinite(col_max):
            norm_values = pd.Series(0.5, index=indicator_df.index, dtype=float)
        elif abs(col_max - col_min) <= 1e-12:
            norm_values = pd.Series(1.0, index=indicator_df.index, dtype=float)
        else:
            norm_values = (values - col_min) / (col_max - col_min)
            if direction < 0:
                norm_values = 1.0 - norm_values
            norm_values = norm_values.fillna(norm_values.mean()).fillna(0.5)

        norm_df[col] = norm_values.clip(0.0, 1.0)
        details[col] = {
            "min": col_min,
            "max": col_max,
            "direction": direction,
            "missing_ratio": float(values.isna().mean()),
        }

    return norm_df, details


def compute_entropy_weights(norm_df):
    """用熵权法根据指标区分度自动计算 18 个指标的权重。"""
    cols = list(INDICATOR_DIRECTIONS.keys())
    X = norm_df[cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    if X.shape[0] == 0:
        return pd.DataFrame(columns=["indicator", "weight", "entropy", "divergence"])

    column_sums = X.sum(axis=0)
    P = np.divide(
        X,
        column_sums.reshape(1, -1),
        out=np.full_like(X, 1.0 / max(X.shape[0], 1)),
        where=column_sums.reshape(1, -1) > 1e-12,
    )

    k = 1.0 / np.log(max(X.shape[0], 2))
    entropy = -k * np.sum(P * np.log(P + 1e-12), axis=0)
    divergence = 1.0 - entropy

    if np.sum(divergence) <= 1e-12:
        weights = np.full(len(cols), 1.0 / len(cols))
    else:
        weights = divergence / np.sum(divergence)

    return pd.DataFrame({
        "indicator": cols,
        "weight": weights,
        "entropy": entropy,
        "divergence": divergence,
    })


def compute_quality_scores(norm_df, weight_df):
    """根据熵权合成综合 Q_score，并计算五个维度分量分数。"""
    cols = list(INDICATOR_DIRECTIONS.keys())
    weight_map = weight_df.set_index("indicator")["weight"].to_dict()

    score_df = norm_df[["trajectory_id", "scene_id", "task_name"]].copy()
    score_df["Q_score"] = 0.0

    for col in cols:
        score_df["Q_score"] += norm_df[col] * weight_map.get(col, 0.0)

    for dimension, prefixes in DIMENSION_MAP.items():
        dim_cols = [
            col for col in cols
            if any(col.startswith(prefix + "_") for prefix in prefixes)
        ]
        dim_weight_sum = sum(weight_map.get(col, 0.0) for col in dim_cols)
        if dim_weight_sum <= 1e-12:
            score_df[f"Q_{dimension}"] = norm_df[dim_cols].mean(axis=1)
        else:
            score_df[f"Q_{dimension}"] = sum(
                norm_df[col] * weight_map.get(col, 0.0) for col in dim_cols
            ) / dim_weight_sum

    return score_df


def compute_pca_robustness(norm_df, score_df):
    """用 PCA 第一主成分和 Q_score 做相关性检查，作为稳健性参考。"""
    cols = list(INDICATOR_DIRECTIONS.keys())
    X = norm_df[cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.5, posinf=1.0, neginf=0.0)

    if len(X) < 2:
        return {
            "status": "not_enough_samples",
            "spearman_corr_with_entropy_q": None,
            "explained_variance_ratio": None,
        }

    centered = X - np.mean(X, axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = centered @ vt[0]
    pc1 = (pc1 - np.min(pc1)) / (np.max(pc1) - np.min(pc1) + 1e-12)

    corr = pd.Series(pc1).corr(score_df["Q_score"], method="spearman")
    total_variance = float(np.sum(singular_values ** 2))
    explained = float((singular_values[0] ** 2) / total_variance) if total_variance > 1e-12 else 0.0
    return {
        "status": "ok",
        "spearman_corr_with_entropy_q": float(corr) if pd.notna(corr) else None,
        "explained_variance_ratio": explained,
    }


def save_stage2_outputs(indicator_df, norm_df, weight_df, score_df, norm_details, pca_info):
    """保存阶段2的原始指标、归一化指标、权重、质量分数和检查信息。"""
    raw_path = os.path.join(CONFIG["interim_dir"], "s3_quality_indicators_raw.csv")
    norm_path = os.path.join(CONFIG["interim_dir"], "s3_quality_indicators_norm.csv")
    weights_path = os.path.join(CONFIG["interim_dir"], "s3_quality_entropy_weights.csv")
    scores_path = os.path.join(CONFIG["interim_dir"], "s3_quality_scores.csv")
    details_path = os.path.join(CONFIG["interim_dir"], "s3_quality_normalization.json")
    pca_path = os.path.join(CONFIG["interim_dir"], "s3_quality_pca_check.json")
    processed_path = os.path.join(CONFIG["processed_dir"], "s3_stage2_quality_dataset.csv")

    save_csv(indicator_df, raw_path)
    save_csv(norm_df, norm_path)
    save_csv(weight_df, weights_path)
    save_csv(score_df, scores_path)
    save_json(norm_details, details_path)
    save_json(pca_info, pca_path)
    save_csv(score_df, processed_path)


def run_stage2_quality():
    """阶段2总入口：读取阶段1结果，计算指标、权重、分数并保存文件。"""
    manifest_df, metadata_df, feature_df, final_label_df, aligned_dict = load_stage2_inputs()

    indicator_df = compute_quality_indicators(
        manifest_df=manifest_df,
        metadata_df=metadata_df,
        feature_df=feature_df,
        final_label_df=final_label_df,
        aligned_dict=aligned_dict,
    )
    norm_df, norm_details = minmax_normalize_indicators(indicator_df)
    weight_df = compute_entropy_weights(norm_df)
    score_df = compute_quality_scores(norm_df, weight_df)
    pca_info = compute_pca_robustness(norm_df, score_df)

    save_stage2_outputs(
        indicator_df=indicator_df,
        norm_df=norm_df,
        weight_df=weight_df,
        score_df=score_df,
        norm_details=norm_details,
        pca_info=pca_info,
    )

    return indicator_df, norm_df, weight_df, score_df


if __name__ == "__main__":
    raw, norm, weights, scores = run_stage2_quality()
    print(
        f"[Stage2] done: raw={raw.shape}, norm={norm.shape}, "
        f"weights={weights.shape}, scores={scores.shape}"
    )
