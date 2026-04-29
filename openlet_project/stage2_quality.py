from stage2_quality_indicators import compute_quality_indicators
from stage2_quality_io import load_stage2_inputs, save_stage2_outputs
from stage2_quality_scoring import (
    compute_entropy_weights,
    compute_pca_robustness,
    compute_quality_scores,
    minmax_normalize_indicators,
)


def run_stage2_quality():
    """阶段2总入口：读取阶段1结果，计算指标、权重、分数并保存文件。"""
    manifest_df, metadata_df, feature_df, final_label_df, aligned_dict = load_stage2_inputs()

    indicator_df = compute_quality_indicators(
        manifest_df=manifest_df,
        metadata_df=metadata_df,
        feature_df=feature_df,
        final_label_df=final_label_df,
        aligned_dict=aligned_dict,
    )
    norm_df, norm_details = minmax_normalize_indicators(indicator_df)
    weight_df = compute_entropy_weights(norm_df)
    score_df = compute_quality_scores(norm_df, weight_df)
    pca_info = compute_pca_robustness(norm_df, score_df)

    save_stage2_outputs(
        indicator_df=indicator_df,
        norm_df=norm_df,
        weight_df=weight_df,
        score_df=score_df,
        norm_details=norm_details,
        pca_info=pca_info,
    )

    return indicator_df, norm_df, weight_df, score_df


if __name__ == "__main__":
    raw, norm, weights, scores = run_stage2_quality()
    print(
        f"[Stage2] done: raw={raw.shape}, norm={norm.shape}, "
        f"weights={weights.shape}, scores={scores.shape}"
    )
