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
