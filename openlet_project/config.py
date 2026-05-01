# 文件位置：config.py

import os


# 全局配置
# 作用：统一管理项目路径、场景信息和基础参数
CONFIG = {
    # 项目根目录
    "project_root": ".",

    # 原始数据目录
    # "raw_dir": r"../../original_data/s6/h5",

    # 中间结果目录
    "interim_dir": "./data/interim",

    # 最终结果目录
    "processed_dir": "./data/processed",

    # 当前场景信息

    "scene_id": "S1",
    "task_name": "single_item_grasp",
    "object_id": "single_item_constant",

    # "scene_id": "S2",
    # "task_name": "material_issuance_card_grabbing",
    # "object_id": "material_issuance_card",

    # "scene_id": "S3",
    # "task_name": "FMCG_placing",
    # "object_id": "FMCG",

    # "scene_id": "S4",
    # "task_name": "parts_offline_into_box",
    # "object_id": "parts",

    # "scene_id": "S5",
    # "task_name": "small_tool_packing",
    # "object_id": "parts_on_shelf",


    # 多场景列表：阶段三正式使用 S1-S5
    "scene_ids": ["S1", "S2", "S3", "S4", "S5"], # ["S1", "S2", "S3", "S4", "S5"]

    # 当前场景数据规模，仅保留给阶段一检查使用
    "expected_num_trajectories": 33,

    # 主关节组
    "main_action_group": "arm",
    "main_state_group": "arm",
    "aux_group": "effector",

    # 保留的相机视角
    "camera_names": ["head", "hand_left", "hand_right"],

    # 训练验证划分参数：阶段三按轨迹划分，不做帧级随机划分
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

    # 阶段三/四 BC 输入输出模式
    # 主实验使用 "arm_only"：
    #   输入过去5帧 arm_state，输出当前 arm_action
    # 可选稳健性实验使用 "arm_effector"：
    #   输入过去5帧 [arm_state, effector_state]，输出当前 [arm_action, effector_action]
    "bc_mode": "arm_only",

    # 阶段四参数：轨迹级 out-of-fold BC 标签
    "stage4_oof_folds": 5,
    "stage4_bc_epochs": 25,
}


# 获取配置值
# 输入：配置键名
# 输出：对应配置值
def get_config(key):
    return CONFIG[key]


# 更新配置值
# 输入：配置键名、配置值
# 输出：无，直接修改全局 CONFIG
def set_config(key, value):
    CONFIG[key] = value


# 创建必要目录
# 输入：无
# 输出：无
def ensure_dirs():
    # os.makedirs(CONFIG["raw_dir"], exist_ok=True)
    os.makedirs(CONFIG["interim_dir"], exist_ok=True)
    os.makedirs(CONFIG["processed_dir"], exist_ok=True)


# 获取当前场景文件前缀（如 S1 -> s1）
def get_scene_prefix():
    scene_id = str(CONFIG.get("scene_id", "S3")).strip()
    if not scene_id:
        scene_id = "S3"
    return scene_id.lower()