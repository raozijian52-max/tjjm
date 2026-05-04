# 文件位置：run_stage3.py

import argparse
import os

import pandas as pd

from config import CONFIG, ensure_dirs
from stage3_bc_value import run_stage3_bc_value
from stage3_meta_value import run_stage3_meta_value
from stage3_repeat import run_stage3_repeated
from stage3_trajectory_value import run_stage3_trajectory_value


# 说明：阶段三统一入口。
# --step=all（默认）按顺序运行全部；也可单独运行某一步。
def parse_args():
    parser = argparse.ArgumentParser(description="Run stage3 pipeline.")
    parser.add_argument(
        "--step",
        choices=["all", "bc", "repeat", "meta", "trajectory"],
        default="all",
        help="Which stage3 step to run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 2024, 2025],
        help="Seeds for --step repeat (default: 42 2024 2025).",
    )
    return parser.parse_args()


def run_bc_step():
    run_stage3_bc_value()

    split_path = os.path.join(CONFIG["interim_dir"], "s_all_bc_split.csv")
    split_df = pd.read_csv(split_path)

    print("[Step1:bc] done")
    print(split_df.groupby(["scene_id", "split"]).size())


def run_repeat_step(seeds):
    run_stage3_repeated(seeds=seeds)
    print(f"[Step2:repeat] done, seeds={seeds}")


def run_meta_step():
    # 注意：meta 为探索性分析，不作为阶段四主输入
    run_stage3_meta_value()
    print("[Step3:meta] done (exploratory)")


def run_trajectory_step():
    run_stage3_trajectory_value()
    print("[Step4:trajectory] done")


def main():
    args = parse_args()
    ensure_dirs()

    if args.step == "all":
        # 正式结果以多 seed repeat 为准；
        # 单独 bc 仅作为调试入口，不在 all 中重复运行，避免额外耗时。
        run_repeat_step(args.seeds)
        run_meta_step()
        run_trajectory_step()
        print("[Stage3] all done")
        return

    if args.step == "bc":
        run_bc_step()
        return

    if args.step == "repeat":
        run_repeat_step(args.seeds)
        return

    if args.step == "meta":
        run_meta_step()
        return

    if args.step == "trajectory":
        run_trajectory_step()
        return


if __name__ == "__main__":
    main()
