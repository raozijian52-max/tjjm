import os


# 全局配置
# 作用：统一管理项目路径、场景信息和基础参数
CONFIG = {
    # 项目根目录
    "project_root": ".",

    # 原始数据目录
    "raw_dir": "./data/raw",

    # 中间结果目录
    "interim_dir": "./data/interim",

    # 最终结果目录
    "processed_dir": "./data/processed",

    # 当前场景信息
    "scene_id": "S3",
    "task_name": "single_item_grasp",
    "object_id": "single_product_constant",

    # 当前数据规模
    "expected_num_trajectories": 146,

    # 主关节组
    "main_action_group": "arm",
    "main_state_group": "arm",
    "aux_group": "effector",

    # 保留的相机视角
    "camera_names": ["head", "hand_left", "hand_right"],

    # 训练验证划分参数
    "train_ratio": 0.8,
    "random_state": 42,
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
    os.makedirs(CONFIG["raw_dir"], exist_ok=True)
    os.makedirs(CONFIG["interim_dir"], exist_ok=True)
    os.makedirs(CONFIG["processed_dir"], exist_ok=True)