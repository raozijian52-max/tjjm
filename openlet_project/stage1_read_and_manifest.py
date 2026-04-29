# 文件位置：stage1_read_and_manifest.py

import json
import os
import tarfile

import h5py
import numpy as np
import pandas as pd

from config import CONFIG, get_scene_prefix
from utils import is_uuid_like, save_csv, to_python


TAR_EXTENSIONS = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2")


def is_tar_file(file_name):
    return file_name.lower().endswith(TAR_EXTENSIONS)


def is_h5_file(file_name):
    return file_name.lower().endswith(".h5")


def safe_member_parts(member_name):
    # Keep tar members inside our extraction folder and avoid absolute/parent paths.
    normalized = member_name.replace("\\", "/")
    parts = []
    for part in normalized.split("/"):
        if part in ("", ".", ".."):
            continue
        parts.append(part)
    return parts


def extract_h5_from_tar(tar_path):
    extract_root = os.path.join(CONFIG["interim_dir"], "extracted_h5")
    tar_stem = os.path.basename(tar_path)
    for suffix in [".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar"]:
        if tar_stem.lower().endswith(suffix):
            tar_stem = tar_stem[: -len(suffix)]
            break

    output_root = os.path.join(extract_root, tar_stem)
    extracted_files = []

    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not is_h5_file(member.name):
                continue

            parts = safe_member_parts(member.name)
            if not parts:
                continue

            output_path = os.path.join(output_root, *parts)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            source = tar.extractfile(member)
            if source is None:
                continue

            with source, open(output_path, "wb") as f:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

            extracted_files.append(output_path)

    extracted_files.sort()
    return extracted_files


# 读取 H5 对象的属性
# 输入：h5 group 或 dataset
# 输出：属性字典
def read_attrs(h5_obj):
    attrs = {}
    for key, value in h5_obj.attrs.items():
        attrs[key] = to_python(value)
    return attrs


# 安全读取 dataset
# 输入：h5 group、dataset 名称
# 输出：dataset 内容；若不存在则返回 None
def safe_read_dataset(group, dataset_name):
    if dataset_name not in group:
        return None

    value = group[dataset_name][()]

    if isinstance(value, np.generic):
        return value.item()

    return value


# 读取 metadata.json
# 输入：h5 文件对象
# 输出：metadata 字典
def read_metadata_json(h5_file):
    if "metadata.json" not in h5_file:
        return {}

    raw_meta = h5_file["metadata.json"][()]

    if isinstance(raw_meta, bytes):
        raw_meta = raw_meta.decode("utf-8", errors="replace")

    if isinstance(raw_meta, np.ndarray):
        raw_meta = raw_meta.item()

    if not isinstance(raw_meta, str):
        return {}

    try:
        return json.loads(raw_meta)
    except json.JSONDecodeError:
        return {"raw_metadata_text": raw_meta}


# 读取单个 joints group
# 输入：如 joints/action/arm 这样的 group
# 输出：包含 position、velocity、timestamp 的字典
def read_joint_group(group):
    result = {
        "position": None,
        "velocity": None,
        "timestamp_ns": None,
        "attrs": read_attrs(group),
    }

    position = safe_read_dataset(group, "position")
    velocity = safe_read_dataset(group, "velocity")
    timestamp_ns = safe_read_dataset(group, "timestamp")

    if position is not None:
        result["position"] = np.asarray(position)

    if velocity is not None:
        result["velocity"] = np.asarray(velocity)

    if timestamp_ns is not None:
        result["timestamp_ns"] = np.asarray(timestamp_ns, dtype=np.int64)

    return result


# 将 object 数组中的图像字节流转为 Python bytes 列表
# 输入：H5 中读取出来的图像数组
# 输出：bytes 列表
def normalize_bytes_list(data):
    if data is None:
        return None

    if isinstance(data, (bytes, bytearray)):
        return [bytes(data)]

    if not isinstance(data, np.ndarray):
        return None

    output = []

    for item in data:
        if item is None:
            output.append(b"")
            continue

        if isinstance(item, (bytes, bytearray)):
            output.append(bytes(item))
            continue

        if isinstance(item, np.ndarray):
            output.append(item.tobytes())
            continue

        try:
            output.append(bytes(item))
        except Exception:
            output.append(b"")

    return output


# 读取单路相机 group
# 输入：如 cameras/head 这样的 group
# 输出：包含 color/depth 字节流和时间戳的字典
def read_camera_group(group):
    result = {
        "color_bytes": None,
        "color_timestamp_ns": None,
        "depth_bytes": None,
        "depth_timestamp_ns": None,
        "attrs": read_attrs(group),
    }

    # 读取 color
    if "color" in group:
        color_group = group["color"]
        color_data = safe_read_dataset(color_group, "data")
        color_ts = safe_read_dataset(color_group, "timestamp")

        result["color_bytes"] = normalize_bytes_list(color_data)

        if color_ts is not None:
            result["color_timestamp_ns"] = np.asarray(color_ts, dtype=np.int64)

    # 读取 depth
    if "depth" in group:
        depth_group = group["depth"]
        depth_data = safe_read_dataset(depth_group, "data")
        depth_ts = safe_read_dataset(depth_group, "timestamp")

        result["depth_bytes"] = normalize_bytes_list(depth_data)

        if depth_ts is not None:
            result["depth_timestamp_ns"] = np.asarray(depth_ts, dtype=np.int64)

    return result


# 读取单个 H5 文件
# 输入：H5 文件路径
# 输出：一条轨迹的原始字典
def read_one_h5(file_path):
    trajectory_id = os.path.splitext(os.path.basename(file_path))[0]

    result = {
        "trajectory_id": trajectory_id,
        "file_path": file_path,
        "scene_id": CONFIG["scene_id"],
        "task_name": CONFIG["task_name"],
        "metadata": {},
        "root_attrs": {},
        "joints_action": {},
        "joints_state": {},
        "cameras": {},
    }

    with h5py.File(file_path, "r") as h5_file:
        # 读取 metadata
        result["metadata"] = read_metadata_json(h5_file)

        # 读取根属性
        result["root_attrs"] = read_attrs(h5_file)

        # 读取 joints/action/*
        if "joints" in h5_file and "action" in h5_file["joints"]:
            action_group = h5_file["joints"]["action"]
            for name in action_group.keys():
                result["joints_action"][name] = read_joint_group(action_group[name])

        # 读取 joints/state/*
        if "joints" in h5_file and "state" in h5_file["joints"]:
            state_group = h5_file["joints"]["state"]
            for name in state_group.keys():
                result["joints_state"][name] = read_joint_group(state_group[name])

        # 读取 cameras/*
        if "cameras" in h5_file:
            camera_group = h5_file["cameras"]
            for name in camera_group.keys():
                result["cameras"][name] = read_camera_group(camera_group[name])

    return result


# 提取单条轨迹的简要摘要
# 输入：read_one_h5 的输出字典
# 输出：适合放进 metadata 表的一行字典
def summarize_trajectory(traj):
    metadata = traj["metadata"]

    resource_info = metadata.get("resource_info", {}) if isinstance(metadata, dict) else {}
    equipment_info = metadata.get("equipment_info", {}) if isinstance(metadata, dict) else {}

    arm_action = traj["joints_action"].get("arm")
    arm_state = traj["joints_state"].get("arm")
    effector_action = traj["joints_action"].get("effector")
    effector_state = traj["joints_state"].get("effector")

    def get_shape(x):
        if x is None:
            return None
        return tuple(x.shape)

    def get_dim(joint_group):
        if joint_group is None:
            return None
        if joint_group["position"] is None:
            return None
        if joint_group["position"].ndim != 2:
            return None
        return int(joint_group["position"].shape[1])

    row = {
        "trajectory_id": traj["trajectory_id"],
        "file_path": traj["file_path"],
        "file_name": os.path.basename(traj["file_path"]),
        "scene_id": traj["scene_id"],
        "task_name": traj["task_name"],

        "resource_id": resource_info.get("id"),
        "owner": resource_info.get("owner"),
        "region": resource_info.get("region"),
        "created_at": resource_info.get("created_at"),

        "manufacturer": equipment_info.get("manufacturer"),
        "equipment_model": equipment_info.get("model"),
        "equipment_sn": equipment_info.get("sn"),

        "collected_at": metadata.get("collected_at") if isinstance(metadata, dict) else None,
        "collection_duration_in_ms": metadata.get("collection_duration_in_ms") if isinstance(metadata, dict) else None,
        "duration_in_ms": metadata.get("duration_in_ms") if isinstance(metadata, dict) else None,
        "file_size_in_bytes": metadata.get("file_size_in_bytes") if isinstance(metadata, dict) else None,

        "action_groups": sorted(list(traj["joints_action"].keys())),
        "state_groups": sorted(list(traj["joints_state"].keys())),
        "camera_names": sorted(list(traj["cameras"].keys())),

        "arm_action_shape": get_shape(arm_action["position"]) if arm_action else None,
        "arm_state_shape": get_shape(arm_state["position"]) if arm_state else None,
        "effector_action_shape": get_shape(effector_action["position"]) if effector_action else None,
        "effector_state_shape": get_shape(effector_state["position"]) if effector_state else None,

        "arm_action_dim": get_dim(arm_action),
        "arm_state_dim": get_dim(arm_state),
        "effector_action_dim": get_dim(effector_action),
        "effector_state_dim": get_dim(effector_state),
    }

    # 统计三路相机的帧数
    for cam_name in traj["cameras"]:
        cam = traj["cameras"][cam_name]
        row[f"{cam_name}_color_n"] = len(cam["color_bytes"]) if cam["color_bytes"] is not None else 0
        row[f"{cam_name}_depth_n"] = len(cam["depth_bytes"]) if cam["depth_bytes"] is not None else 0

    return row


# 扫描 raw_dir 下所有 h5 文件
# 输入：无
# 输出：H5 文件路径列表
def scan_h5_files():
    raw_dir = CONFIG["raw_dir"]
    files = []

    for dir_path, _, file_names in os.walk(raw_dir):
        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)

            if is_h5_file(file_name):
                files.append(file_path)
                continue

            if is_tar_file(file_name):
                files.extend(extract_h5_from_tar(file_path))

    files.sort()
    return files


# 构建 manifest 表
# 输入：H5 文件路径列表
# 输出：manifest DataFrame
def build_manifest(file_list):
    rows = []

    for file_path in file_list:
        file_name = os.path.basename(file_path)
        trajectory_id = os.path.splitext(file_name)[0]

        rows.append({
            "trajectory_id": trajectory_id,
            "file_name": file_name,
            "file_path": file_path,
            "scene_id": CONFIG["scene_id"],
            "task_name": CONFIG["task_name"],
            "object_id": CONFIG["object_id"],
            "is_uuid_file_name": is_uuid_like(trajectory_id),
        })

    return pd.DataFrame(rows)


# 构建 raw_metadata 表
# 输入：manifest DataFrame
# 输出：metadata DataFrame
def build_raw_metadata(manifest_df):
    rows = []

    for _, row in manifest_df.iterrows():
        file_path = row["file_path"]

        try:
            traj = read_one_h5(file_path)
            summary = summarize_trajectory(traj)
            summary["read_success"] = True
            summary["read_error"] = None
        except Exception as e:
            summary = {
                "trajectory_id": row["trajectory_id"],
                "file_name": row["file_name"],
                "file_path": row["file_path"],
                "scene_id": row["scene_id"],
                "task_name": row["task_name"],
                "read_success": False,
                "read_error": str(e),
            }

        rows.append(summary)

    return pd.DataFrame(rows)


# 运行阶段一第1步
# 输入：无
# 输出：manifest_df, raw_metadata_df
def run_stage1_step1():
    file_list = scan_h5_files()

    manifest_df = build_manifest(file_list)
    raw_metadata_df = build_raw_metadata(manifest_df)

    scene_prefix = get_scene_prefix()
    manifest_save_path = os.path.join(CONFIG["interim_dir"], f"{scene_prefix}_manifest.csv")
    metadata_save_path = os.path.join(CONFIG["interim_dir"], f"{scene_prefix}_raw_metadata.csv")

    save_csv(manifest_df, manifest_save_path)
    save_csv(raw_metadata_df, metadata_save_path)

    return manifest_df, raw_metadata_df
