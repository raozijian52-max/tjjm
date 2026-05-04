import pandas as pd

from config import ensure_dirs
from stage5_causal import run_stage5_causal_analysis


def _print_balance_brief(balance_df: pd.DataFrame):
    if balance_df.empty:
        print("[Stage5] balance summary is empty")
        return

    grp = (
        balance_df.groupby(["treatment", "outcome", "heterogeneity_group", "match_method"], as_index=False)
        .agg(
            n_covariates=("covariate", "count"),
            mean_abs_smd_before=("abs_smd_before", "mean"),
            mean_abs_smd_after=("abs_smd_after", "mean"),
            n_good_after=("abs_smd_after", lambda x: int((x < 0.1).sum())),
        )
        .sort_values(["treatment", "outcome", "heterogeneity_group", "match_method"])
    )
    print("\n[Stage5] balance summary (brief)")
    print(grp.to_string(index=False))


def main():
    ensure_dirs()

    (
        causal_df,
        effect_df,
        balance_df,
        matched_pairs_df,
        weights_df,
        overlap_df,
        treatment_info_df,
    ) = run_stage5_causal_analysis()

    print("[Stage5] done (quasi-causal robustness analysis)")
    print("[Stage5] note: this stage estimates net association after controlling observable confounders;")
    print("[Stage5] it is NOT a strict causal proof under unobserved confounding.")
    print(f"causal_df={causal_df.shape}")
    print(f"effect_df={effect_df.shape}")
    print(f"balance_df={balance_df.shape}")
    print(f"matched_pairs_df={matched_pairs_df.shape}")
    print(f"weights_df={weights_df.shape}")
    print(f"overlap_df={overlap_df.shape}")

    print("\n[Stage5] treatment summary")
    print(treatment_info_df.to_string(index=False))

    if not effect_df.empty:
        need_cols = [
            "treatment",
            "outcome",
            "heterogeneity_group",
            "match_method",
            "n_treated_matched",
            "att_raw",
            "att_beneficial_direction",
            "ci95_lo_beneficial",
            "ci95_hi_beneficial",
            "mean_abs_smd_before",
            "mean_abs_smd_after",
        ]
        core_cols = [c for c in need_cols if c in effect_df.columns]
        core = effect_df[core_cols].sort_values(
            [c for c in ["treatment", "outcome", "heterogeneity_group", "match_method"] if c in core_cols]
        )
        print("\n[Stage5] effect estimates (core)")
        print(core.to_string(index=False))

    _print_balance_brief(balance_df)


if __name__ == "__main__":
    main()
