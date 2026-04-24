# 文件位置：run_stage1_step1.py

from config import CONFIG, ensure_dirs
from stage1_read_and_manifest import run_stage1_step1


# 主程序入口
# 作用：运行阶段一第1步，并打印基础检查结果
def main():
    ensure_dirs()

    manifest_df, raw_metadata_df = run_stage1_step1()

    print("=" * 60)
    print("阶段一第1步完成：H5读取与manifest构建")
    print("=" * 60)

    print(f"manifest 条数: {len(manifest_df)}")
    print(f"metadata 条数: {len(raw_metadata_df)}")

    if "read_success" in raw_metadata_df.columns:
        print(f"读取成功条数: {raw_metadata_df['read_success'].sum()}")

    # 打印几个关键字段，方便初步检查
    check_cols = [
        "trajectory_id",
        "arm_action_dim",
        "arm_state_dim",
        "effector_action_dim",
        "effector_state_dim",
        "camera_names",
        "read_success",
    ]

    check_cols = [col for col in check_cols if col in raw_metadata_df.columns]
    print(raw_metadata_df[check_cols].head())


if __name__ == "__main__":
    main()