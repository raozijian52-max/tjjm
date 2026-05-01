# 文件位置：run_stage4_step1.py

from config import ensure_dirs
from stage4_labels import run_stage4_bc_trajectory_labels


# 主程序入口
# 作用：生成阶段四轨迹级 BC 离线效能标签
def main():
    ensure_dirs()

    label_df, summary_df = run_stage4_bc_trajectory_labels()

    print("=" * 60)
    print("阶段四 Step 1 完成：轨迹级 BC 离线效能标签生成")
    print("=" * 60)

    print("\n[1] 标签表形状")
    print(label_df.shape)

    print("\n[2] 场景级标签摘要")
    print(summary_df)

    print("\n[3] 标签表示例")
    print(label_df.head())

    print("\n结果已保存到 data/interim：")
    print("- stage4_bc_trajectory_labels.csv")
    print("- stage4_oof_folds.csv")
    print("- stage4_bc_label_summary.csv")


if __name__ == "__main__":
    main()