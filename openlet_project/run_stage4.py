import argparse

from config import ensure_dirs
from stage4_labels import run_stage4_bc_trajectory_labels
from stage4_modeling_table import run_stage4_build_master_table
from stage4_regression import run_stage4_regression


# 说明：阶段四统一入口。
# --step=all（默认）按顺序运行全部；也可单独运行某一步。
def parse_args():
    parser = argparse.ArgumentParser(description="Run stage4 pipeline.")
    parser.add_argument(
        "--step",
        choices=["all", "labels", "table", "regression"],
        default="all",
        help="Which stage4 step to run.",
    )
    return parser.parse_args()


def run_labels_step():
    run_stage4_bc_trajectory_labels()
    print("[Step1:labels] done")


def run_table_step():
    run_stage4_build_master_table()
    print("[Step2:table] done")


def run_regression_step():
    run_stage4_regression()
    print("[Step3:regression] done")


def main():
    args = parse_args()
    ensure_dirs()

    if args.step == "all":
        run_labels_step()
        run_table_step()
        run_regression_step()
        print("[Stage4] all done")
        return

    if args.step == "labels":
        run_labels_step()
        return

    if args.step == "table":
        run_table_step()
        return

    if args.step == "regression":
        run_regression_step()
        return


if __name__ == "__main__":
    main()
