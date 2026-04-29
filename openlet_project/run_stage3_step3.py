# 文件位置：run_stage3_step3.py

from config import ensure_dirs
from stage3_meta_value import run_stage3_meta_value


# 主程序入口
# 作用：运行阶段三 Step 3，构建质量到效能价值的元模型
def main():
    ensure_dirs()

    scene_quality_df, model_df, corr_df, loocv_df, metrics_df, pred_df = run_stage3_meta_value()

    print("=" * 60)
    print("阶段三 Step 3 完成：质量特征 -> 效能价值元模型")
    print("=" * 60)

    print("\n[1] 场景级质量画像")
    show_quality_cols = [
        "scene_id",
        "n_trajectories",
        "Q_score_mean",
        "Q_completeness_mean",
        "Q_accuracy_mean",
        "Q_diversity_mean",
        "Q_consistency_mean",
        "Q_usability_mean",
    ]
    print(scene_quality_df[show_quality_cols])

    print("\n[2] 元模型训练表")
    show_model_cols = [
        "scene_id",
        "Q_score_mean",
        "Q_completeness_mean",
        "Q_accuracy_mean",
        "Q_diversity_mean",
        "Q_consistency_mean",
        "Q_usability_mean",
        "delta_score_mean",
        "delta_normalized_mse_mean",
    ]
    show_model_cols = [c for c in show_model_cols if c in model_df.columns]
    print(model_df[show_model_cols])

    print("\n[3] 质量特征与 delta_score_mean 的相关性 Top 10")
    print(corr_df[[
        "feature",
        "pearson_corr",
        "spearman_corr",
        "pearson_p",
        "spearman_p",
    ]].head(10))

    print("\n[4] Ridge 留一场景预测结果")
    print(loocv_df)

    print("\n[5] Ridge 留一场景整体指标")
    print(metrics_df)

    print("\n[6] 场景级效能价值预测")
    print(pred_df)

    print("\n结果已保存到 data/interim：")
    print("- stage3_scene_quality_table.csv")
    print("- stage3_scene_value_model_table.csv")
    print("- stage3_quality_value_correlation.csv")
    print("- stage3_meta_model_loocv.csv")
    print("- stage3_meta_model_metrics.csv")
    print("- stage3_scene_value_prediction.csv")


if __name__ == "__main__":
    main()