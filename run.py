#!/usr/bin/env python3
import argparse
import sys

from dotenv import load_dotenv

from src.config import (
    BLOG_CHARTS_DIR,
    BLOG_DIR,
    CHARTS_DIR,
    DATA_DIR,
    MODELS,
    OUTPUT_DIR,
    RANDOM_SEED,
    SAMPLE_DIR,
    TRIALS_PER_CONDITION,
)


def ensure_dirs():
    for d in [DATA_DIR, OUTPUT_DIR, CHARTS_DIR, BLOG_DIR, BLOG_CHARTS_DIR, SAMPLE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def cmd_collect(args):
    from src.collect import collect_all
    from src.prompts import generate_trial_plan

    models = [args.model] if args.model else MODELS
    plan = generate_trial_plan(seed=RANDOM_SEED, trials=args.trials, models=models)

    if args.dry_run:
        print(f"Dry run — would make {len(plan)} API calls")
        print(f"  Models: {', '.join(models)}")
        print(f"  Conditions: 6 x {args.trials} trials x 10 questions")
        est_cost = len(plan) * 0.0003
        print(f"  Estimated cost: ${est_cost:.2f}")
        return

    collect_all(plan)


def cmd_analyze(args):
    from src.analyze import run_analysis
    from src.metrics import compute_all_metrics

    print("Computing metrics...")
    compute_all_metrics()
    print("Running statistical analysis...")
    run_analysis()
    print("Analysis complete. Results in output/")


def cmd_visualize(args):
    from src.visualize import generate_all_charts

    print("Generating charts...")
    generate_all_charts()
    print(f"Charts saved to {CHARTS_DIR}/")


def cmd_blog(args):
    from src.blog import generate_blog_post

    print("Generating blog post...")
    generate_blog_post()
    print(f"Blog post saved to {BLOG_DIR}/post.md")


def cmd_all(args):
    cmd_collect(args)
    cmd_analyze(args)
    cmd_visualize(args)
    cmd_blog(args)


def main():
    load_dotenv()
    ensure_dirs()

    parser = argparse.ArgumentParser(
        description="LLM Incentive Signal Experiment"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="Run data collection")
    p_collect.add_argument("--trials", type=int, default=TRIALS_PER_CONDITION)
    p_collect.add_argument("--model", type=str, default=None, help="Single model to test")
    p_collect.add_argument("--dry-run", action="store_true")
    p_collect.set_defaults(func=cmd_collect)

    p_analyze = sub.add_parser("analyze", help="Compute metrics and statistics")
    p_analyze.set_defaults(func=cmd_analyze)

    p_viz = sub.add_parser("visualize", help="Generate charts")
    p_viz.set_defaults(func=cmd_visualize)

    p_blog = sub.add_parser("blog", help="Generate blog post markdown")
    p_blog.set_defaults(func=cmd_blog)

    p_all = sub.add_parser("all", help="Run full pipeline")
    p_all.add_argument("--trials", type=int, default=TRIALS_PER_CONDITION)
    p_all.add_argument("--model", type=str, default=None)
    p_all.add_argument("--dry-run", action="store_true")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
