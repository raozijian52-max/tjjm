# 文件位置：config_template.py

import os


# 模板配置（供阶段四及后续并行场景运行）
# 说明：请仅修改 raw_dir / scene_id / task_name / object_id / expected_num_trajectories 等场景相关项。
CONFIG = {
    "project_root": ".",

    # 原始数据目录（按需修改）
    "raw_dir": r"../../original_data/s5_小型工具装箱-将货架上的零件放置在旁边桌面上的固定位置/h5",

    # 中间结果目录
    "interim_dir": "./data/interim",

    # 最终结果目录
    "processed_dir": "./data/processed",

    # 当前场景信息（按需修改）
    "scene_id": "S5",
    "task_name": "small_tool_packing",
    "object_id": "parts_on_shelf",

    # 多场景列表
    "scene_ids": ["S1", "S2", "S3", "S4", "S5"],

    # 当前场景数据规模（阶段一检查用）
    "expected_num_trajectories": 33,

    # 主关节组
    "main_action_group": "arm",
    "main_state_group": "arm",
    "aux_group": "effector",

    # 保留的相机视角
    "camera_names": ["head", "hand_left", "hand_right"],

    # 划分参数
    "train_ratio": 0.8,
    "random_state": 42,

    # 阶段三 BC 参数
    "bc_window_size": 5,
    "bc_sample_stride": 2,
    "bc_batch_size": 512,
    "bc_epochs": 35,
    "bc_learning_rate": 1e-3,
    "bc_hidden_dims": [128, 128],
    "bc_weight_decay": 1e-5,
    "bc_num_workers": 0,
}


def get_config(key):
    return CONFIG[key]


def set_config(key, value):
    CONFIG[key] = value


def ensure_dirs():
    os.makedirs(CONFIG["raw_dir"], exist_ok=True)
    os.makedirs(CONFIG["interim_dir"], exist_ok=True)
    os.makedirs(CONFIG["processed_dir"], exist_ok=True)


def get_scene_prefix():
    scene_id = str(CONFIG.get("scene_id", "S3")).strip()
    if not scene_id:
        scene_id = "S3"
    return scene_id.lower()
