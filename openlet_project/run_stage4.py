import argparse

from config import ensure_dirs
from stage4_curation import run_stage4_curation


# 说明：阶段四统一入口（数据策展验证版）。
def parse_args():
    parser = argparse.ArgumentParser(description="Run stage4 curation pipeline.")
    parser.add_argument(
        "--step",
        choices=["all", "curation"],
        default="all",
        help="Which stage4 step to run.",
    )
    return parser.parse_args()


def run_curation_step():
    run_stage4_curation()
    print("[Stage4:curation] done")


def main():
    args = parse_args()
    ensure_dirs()

    if args.step == "all":
        run_curation_step()
        print("[Stage4] all done")
        return

    if args.step == "curation":
        run_curation_step()
        return


if __name__ == "__main__":
    main()
