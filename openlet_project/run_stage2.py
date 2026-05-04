import argparse

from config import ensure_dirs
from stage2_quality import run_stage2_pca_only, run_stage2_quality


def main():
    parser = argparse.ArgumentParser(description="Run stage2 quality pipeline.")
    parser.add_argument(
        "--pca-only",
        action="store_true",
        help="Reuse cached stage2 outputs and only recompute PCA check json.",
    )
    args = parser.parse_args()

    ensure_dirs()

    if args.pca_only:
        pca_info = run_stage2_pca_only()
        print(
            "[Stage2] PCA-only completed: "
            f"status={pca_info.get('status')}, "
            f"selected_n_components={pca_info.get('selected_n_components')}"
        )
        return

    indicator_df, norm_df, weight_df, score_df = run_stage2_quality()
    print(
        f"[Stage2] completed: indicators={indicator_df.shape}, "
        f"normalized={norm_df.shape}, weights={weight_df.shape}, scores={score_df.shape}"
    )


if __name__ == "__main__":
    main()
