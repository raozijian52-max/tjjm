# 文件位置：run_stage3_step2.py

from config import ensure_dirs
from stage3_repeat import run_stage3_repeated


# 主程序入口
# 作用：重复运行阶段三留一场景实验，检查 delta_score 稳定性
def main():
    ensure_dirs()

    # 默认使用 3 个 seed。算力允许时可以改为 [42, 2024, 2025, 7, 123]
    seeds = [42, 2024, 2025]

    base_all_df, leave_all_df, delta_all_df, delta_summary_df = run_stage3_repeated(
        seeds=seeds
    )

    print("=" * 60)
    print("阶段三 Step 2 完成：重复随机种子稳定性检验")
    print("=" * 60)

    print("\n[1] 每个场景的 delta 稳定性摘要")
    print(delta_summary_df[[
        "scene_id",
        "n_runs",
        "delta_score_mean",
        "delta_score_std",
        "delta_score_positive_rate",
        "delta_normalized_mse_mean",
        "delta_normalized_mse_std",
        "rank_by_delta_score_mean",
    ]])

    print("\n[2] 所有 seed 的原始 delta 结果")
    print(delta_all_df[[
        "seed",
        "scene_id",
        "delta_score",
        "delta_normalized_mse",
        "base_imitation_score",
        "leave_imitation_score",
    ]])

    print("\n结果已保存到 data/interim：")
    print("- stage3_repeat_base_all.csv")
    print("- stage3_repeat_leave_all.csv")
    print("- stage3_repeat_delta_all.csv")
    print("- stage3_repeat_delta_summary.csv")


if __name__ == "__main__":
    main()