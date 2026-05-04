from config import ensure_dirs
from stage1_read_and_manifest import run_stage1_step1
from stage1_align import run_stage1_step2
from stage1_label import run_stage1_step3
from stage1_features import run_stage1_step4


# 主程序入口
# 作用：顺序执行阶段一全部步骤，并将日志简化为“每步一行”
def main():
    ensure_dirs()

    manifest_df, raw_metadata_df = run_stage1_step1()
    print(f"[Step1] 完成：manifest={len(manifest_df)}，metadata={len(raw_metadata_df)}")

    aligned_result_dict, summary_df = run_stage1_step2()
    print(f"[Step2] 完成：aligned={len(aligned_result_dict)}，summary={len(summary_df)}")

    signals_df, auto_label_df, review_queue_df, final_label_df = run_stage1_step3()
    print(
        f"[Step3] 完成：signals={len(signals_df)}，auto={len(auto_label_df)}，"
        f"review={len(review_queue_df)}，final={len(final_label_df)}"
    )

    feature_df, stage1_dataset_df = run_stage1_step4()
    print(f"[Step4] 完成：feature_shape={feature_df.shape}，dataset_shape={stage1_dataset_df.shape}")


if __name__ == "__main__":
    main()
