# 文件位置：run_stage3_step4.py

from config import ensure_dirs
from stage3_trajectory_value import run_stage3_trajectory_value


# 主程序入口
# 作用：运行阶段三 Step 4，生成轨迹级效能价值特征
def main():
    ensure_dirs()

    traj_value_df, final_output_df, summary_df = run_stage3_trajectory_value()

    print("=" * 60)
    print("阶段三 Step 4 完成：轨迹级效能价值特征生成")
    print("=" * 60)

    print("\n[1] 轨迹级效能价值表形状")
    print(f"traj_value_df shape: {traj_value_df.shape}")
    print(f"final_output_df shape: {final_output_df.shape}")

    print("\n[2] 场景级继承值摘要")
    print(summary_df)

    print("\n[3] 轨迹级效能价值表示例")
    show_cols = [
        "global_id",
        "trajectory_id",
        "scene_id",
        "delta_score_emp",
        "delta_score_pred",
        "delta_score_recommended",
        "stage3_value_mode",
        "stage3_empirical_value_source",
    ]
    show_cols = [col for col in show_cols if col in traj_value_df.columns]
    print(traj_value_df[show_cols].head())

    print("\n结果已保存到 data/interim：")
    print("- stage3_trajectory_value_features.csv")
    print("- stage3_final_value_outputs.csv")
    print("- stage3_trajectory_value_summary.csv")


if __name__ == "__main__":
    main()