# 文件位置：run_stage4_step3.py

from config import ensure_dirs
from stage4_regression import run_stage4_regression


# 主程序入口
# 作用：运行阶段四 Config A/B/C 回归对照实验
def main():
    ensure_dirs()

    comparison_df, predictions_df, importance_df, dropped_df = run_stage4_regression()

    print("=" * 60)
    print("阶段四 Step 3 完成：Config A/B/C 回归对照实验")
    print("=" * 60)

    print("\n[1] Config 对照结果")
    show_cols = [
        "protocol",
        "config",
        "model",
        "n_features_original",
        "n_features_used_mean",
        "mae_mean",
        "rmse_mean",
        "r2_mean",
        "spearman_mean",
        "pearson_mean",
    ]
    print(comparison_df[show_cols])

    print("\n[2] XGBoost / Ridge 特征重要性 Top 20")
    if len(importance_df) > 0:
        print(importance_df.head(20))
    else:
        print("没有可用特征重要性。")

    print("\n[3] 被低方差规则删除的特征数量")
    if len(dropped_df) > 0:
        print(dropped_df.groupby(["config", "reason"])["feature"].nunique())
    else:
        print("没有特征被删除。")

    print("\n结果已保存到 data/interim：")
    print("- stage4_config_comparison.csv")
    print("- stage4_predictions.csv")
    print("- stage4_feature_importance.csv")
    print("- stage4_dropped_features.csv")


if __name__ == "__main__":
    main()