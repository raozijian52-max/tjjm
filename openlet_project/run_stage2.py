from config import ensure_dirs
from stage2_quality import run_stage2_quality


def main():
    ensure_dirs()
    indicator_df, norm_df, weight_df, score_df = run_stage2_quality()
    print(
        f"[Stage2] completed: indicators={indicator_df.shape}, "
        f"normalized={norm_df.shape}, weights={weight_df.shape}, scores={score_df.shape}"
    )


if __name__ == "__main__":
    main()
