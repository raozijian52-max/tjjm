# 文件位置：run_stage3_step1.py

from config import ensure_dirs
from stage3_bc_value import run_stage3_bc_value


# 主程序入口
# 作用：运行阶段三离线 BC 效能评估，并打印核心结果
def main():
    ensure_dirs()

    base_metrics_df, leave_one_metrics_df, delta_df = run_stage3_bc_value()

    print("=" * 60)
    print("阶段三完成：离线 BC 效能评估 + 留一场景边际价值")
    print("=" * 60)

    print("\n[1] 全数据 base 模型：各场景验证集效能")
    print(base_metrics_df[[
        "scene_id",
        "mse",
        "mae",
        "normalized_mse",
        "imitation_score",
        "n_samples",
    ]])

    print("\n[2] 留一场景模型：被留出场景验证集效能")
    print(leave_one_metrics_df[[
        "scene_id",
        "mse",
        "mae",
        "normalized_mse",
        "imitation_score",
        "n_samples",
    ]])

    print("\n[3] 场景边际价值 delta")
    print(delta_df[[
        "scene_id",
        "base_imitation_score",
        "leave_imitation_score",
        "delta_score",
        "base_normalized_mse",
        "leave_normalized_mse",
        "delta_normalized_mse",
    ]])

    print("\n结果已保存到 data/interim：")
    print("- s_all_bc_split.csv")
    print("- stage3_bc_base_metrics_by_scene.csv")
    print("- stage3_bc_leave_one_metrics.csv")
    print("- stage3_scene_delta_value.csv")
    print("- stage3_base_train_log.csv")


if __name__ == "__main__":
    main()