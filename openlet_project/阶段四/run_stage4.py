import argparse

from config import ensure_dirs
from stage4_curation import (
    run_stage4_curation,
    run_stage4_prepare_pool_and_stage3,
    run_stage4_curation_eval,
)


# 说明：阶段四统一入口（数据策展验证版）。
def parse_args():
    parser = argparse.ArgumentParser(description="Run stage4 curation pipeline.")
    parser.add_argument(
        "--step",
        choices=["all", "prepare", "curation", "eval"],
        default="all",
        help="Which stage4 step to run.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run minimal smoke setting (full+random, ratio=1.0, seed=42).",
    )
    parser.add_argument(
        "--ratio-grid",
        type=float,
        nargs="+",
        default=None,
        help="Override ratio grid, e.g. --ratio-grid 0.5 1.0",
    )
    parser.add_argument(
        "--run-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override run seeds for non-random strategies.",
    )
    parser.add_argument(
        "--random-repeat-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override repeat seeds for random strategy.",
    )
    return parser.parse_args()


def run_prepare_step(args):
    run_stage4_prepare_pool_and_stage3(
        smoke_test=args.smoke,
        ratio_grid=tuple(args.ratio_grid) if args.ratio_grid is not None else None,
        run_seeds=tuple(args.run_seeds) if args.run_seeds is not None else None,
        random_repeat_seeds=tuple(args.random_repeat_seeds) if args.random_repeat_seeds is not None else None,
    )
    print("[Stage4:prepare] done")


def run_eval_step(args):
    run_stage4_curation_eval(
        smoke_test=args.smoke,
        ratio_grid=tuple(args.ratio_grid) if args.ratio_grid is not None else None,
        run_seeds=tuple(args.run_seeds) if args.run_seeds is not None else None,
        random_repeat_seeds=tuple(args.random_repeat_seeds) if args.random_repeat_seeds is not None else None,
    )
    print("[Stage4:eval] done")


def run_curation_step(args):
    run_stage4_curation(
        smoke_test=args.smoke,
        ratio_grid=tuple(args.ratio_grid) if args.ratio_grid is not None else None,
        run_seeds=tuple(args.run_seeds) if args.run_seeds is not None else None,
        random_repeat_seeds=tuple(args.random_repeat_seeds) if args.random_repeat_seeds is not None else None,
    )
    print("[Stage4:curation] done")


def main():
    args = parse_args()
    ensure_dirs()

    if args.step == "all":
        run_prepare_step(args)
        run_eval_step(args)
        print("[Stage4] all done")
        return

    if args.step == "prepare":
        run_prepare_step(args)
        return

    if args.step == "eval":
        run_eval_step(args)
        return

    if args.step == "curation":
        run_curation_step(args)
        return


if __name__ == "__main__":
    main()
