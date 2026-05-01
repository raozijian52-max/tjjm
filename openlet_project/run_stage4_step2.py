# 文件位置：run_stage4_step2.py

from config import ensure_dirs
from stage4_modeling_table import run_stage4_build_master_table


# 主程序入口
# 作用：生成阶段四唯一建模主表和 Config A/B/C 特征配置
def main():
    ensure_dirs()

    master_df, feature_config, check_info = run_stage4_build_master_table()

    print("=" * 60)
    print("阶段四 Step 2 完成：建模主表生成")
    print("=" * 60)

    print("\n[1] 主表形状")
    print(master_df.shape)

    print("\n[2] Config 特征数量")
    print(f"Config A: {len(feature_config['config_A'])}")
    print(f"Config B: {len(feature_config['config_B'])}")
    print(f"Config C: {len(feature_config['config_C'])}")

    print("\n[3] 质量特征列")
    print(feature_config["quality_cols"])

    print("\n[4] 阶段三特征列")
    print(feature_config["stage3_cols"])

    print("\n[5] 主表检查")
    print(check_info)

    print("\n[6] 主表示例")
    show_cols = [
        "global_id",
        "scene_id",
        "trajectory_normalized_mse",
        "trajectory_imitation_score",
        "delta_score_emp",
    ]
    show_cols = [c for c in show_cols if c in master_df.columns]
    print(master_df[show_cols].head())

    print("\n结果已保存到 data/interim：")
    print("- stage4_modeling_master_table.csv")
    print("- stage4_feature_config.json")
    print("- stage4_master_check.json")


if __name__ == "__main__":
    main()