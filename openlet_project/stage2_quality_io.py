import os
import pickle

import pandas as pd

from config import CONFIG
from utils import save_csv, save_json


LAST_STAGE2_SCENE_PREFIX = None


def load_stage2_inputs():
    """读取阶段2所需的阶段1输出文件。"""
    global LAST_STAGE2_SCENE_PREFIX

    interim_dir = CONFIG["interim_dir"]

    # 自动识别当前可用的场景前缀（如 s1/s2/s3）
    scene_prefix = None
    for name in sorted(os.listdir(interim_dir)):
        if name.endswith("_manifest.csv") and name.startswith("s"):
            candidate = name.split("_", 1)[0]
            required_paths = [
                os.path.join(interim_dir, f"{candidate}_manifest.csv"),
                os.path.join(interim_dir, f"{candidate}_raw_metadata.csv"),
                os.path.join(interim_dir, f"{candidate}_aligned_data.pkl"),
                os.path.join(interim_dir, f"{candidate}_feature_matrix.csv"),
                os.path.join(interim_dir, f"{candidate}_final_labels.csv"),
            ]
            if all(os.path.exists(p) for p in required_paths):
                scene_prefix = candidate
                break

    if scene_prefix is None:
        scene_prefix = "s3"

    LAST_STAGE2_SCENE_PREFIX = scene_prefix
    print(f"[Stage2] 当前读取场景前缀: {scene_prefix}")

    manifest_path = os.path.join(interim_dir, f"{scene_prefix}_manifest.csv")
    metadata_path = os.path.join(interim_dir, f"{scene_prefix}_raw_metadata.csv")
    aligned_path = os.path.join(interim_dir, f"{scene_prefix}_aligned_data.pkl")
    feature_path = os.path.join(interim_dir, f"{scene_prefix}_feature_matrix.csv")
    final_label_path = os.path.join(interim_dir, f"{scene_prefix}_final_labels.csv")

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

def save_stage2_outputs(indicator_df, norm_df, weight_df, score_df, norm_details, pca_info):
    """保存阶段2的原始指标、归一化指标、权重、质量分数和检查信息。"""
    scene_prefix = LAST_STAGE2_SCENE_PREFIX or "s3"

    raw_path = os.path.join(CONFIG["interim_dir"], f"{scene_prefix}_quality_indicators_raw.csv")
    norm_path = os.path.join(CONFIG["interim_dir"], f"{scene_prefix}_quality_indicators_norm.csv")
    weights_path = os.path.join(CONFIG["interim_dir"], f"{scene_prefix}_quality_entropy_weights.csv")
    scores_path = os.path.join(CONFIG["interim_dir"], f"{scene_prefix}_quality_scores.csv")
    details_path = os.path.join(CONFIG["interim_dir"], f"{scene_prefix}_quality_normalization.json")
    pca_path = os.path.join(CONFIG["interim_dir"], f"{scene_prefix}_quality_pca_check.json")
    processed_path = os.path.join(CONFIG["processed_dir"], f"{scene_prefix}_stage2_quality_dataset.csv")

    save_csv(indicator_df, raw_path)
    save_csv(norm_df, norm_path)
    save_csv(weight_df, weights_path)
    save_csv(score_df, scores_path)
    save_json(norm_details, details_path)
    save_json(pca_info, pca_path)
    save_csv(score_df, processed_path)
