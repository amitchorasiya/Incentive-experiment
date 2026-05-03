import numpy as np
import pandas as pd
from scipy import stats

from .config import CONDITIONS, OUTPUT_DIR

METRIC_COLUMNS = [
    "word_count", "char_count", "sentence_count", "paragraph_count",
    "bullet_point_count", "code_block_count", "example_count", "detail_score",
    "sentiment_polarity", "sentiment_subjectivity",
    "hedging_count", "hedging_density",
    "confidence_count", "confidence_density",
    "superlative_count", "superlative_density",
    "exclamation_count", "question_mark_count", "avg_sentence_length",
    "cost_usd", "accuracy_pct",
]


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    available_metrics = [m for m in METRIC_COLUMNS if m in df.columns]
    for model in df["model"].unique():
        for condition in CONDITIONS:
            subset = df[(df["model"] == model) & (df["condition"] == condition)]
            for metric in available_metrics:
                values = subset[metric].dropna()
                n = len(values)
                if n == 0:
                    continue
                mean = values.mean()
                se = values.std(ddof=1) / np.sqrt(n) if n > 1 else 0
                rows.append({
                    "model": model,
                    "condition": condition,
                    "metric": metric,
                    "mean": round(mean, 4),
                    "median": round(values.median(), 4),
                    "std": round(values.std(ddof=1), 4) if n > 1 else 0,
                    "se": round(se, 4),
                    "ci_lower": round(mean - 1.96 * se, 4),
                    "ci_upper": round(mean + 1.96 * se, 4),
                    "min": round(values.min(), 4),
                    "max": round(values.max(), 4),
                    "n": n,
                })
    return pd.DataFrame(rows)


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = group1.var(ddof=1)
    var2 = group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (group2.mean() - group1.mean()) / pooled_std


def run_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    treatments = [c for c in CONDITIONS if c != "control"]
    rows = []

    available_metrics = [m for m in METRIC_COLUMNS if m in df.columns]
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        for metric in available_metrics:
            control_vals = model_df[model_df["condition"] == "control"][metric].dropna()
            if len(control_vals) < 2:
                continue

            for treatment in treatments:
                treat_vals = model_df[model_df["condition"] == treatment][metric].dropna()
                if len(treat_vals) < 2:
                    continue

                t_stat, p_value = stats.ttest_ind(
                    control_vals, treat_vals, equal_var=False
                )
                d = cohens_d(control_vals, treat_vals)

                try:
                    u_stat, u_p = stats.mannwhitneyu(
                        control_vals, treat_vals, alternative="two-sided"
                    )
                except ValueError:
                    u_stat, u_p = np.nan, np.nan

                p_adjusted = min(p_value * len(treatments), 1.0)

                rows.append({
                    "model": model,
                    "metric": metric,
                    "treatment": treatment,
                    "control_mean": round(control_vals.mean(), 4),
                    "treatment_mean": round(treat_vals.mean(), 4),
                    "diff": round(treat_vals.mean() - control_vals.mean(), 4),
                    "cohens_d": round(d, 4),
                    "t_statistic": round(t_stat, 4),
                    "p_value_raw": round(p_value, 6),
                    "p_value_adjusted": round(p_adjusted, 6),
                    "significant_at_05": p_adjusted < 0.05,
                    "mann_whitney_U": round(u_stat, 4) if not np.isnan(u_stat) else None,
                    "mann_whitney_p": round(u_p, 6) if not np.isnan(u_p) else None,
                    "effect_size": (
                        "large" if abs(d) >= 0.8
                        else "medium" if abs(d) >= 0.5
                        else "small" if abs(d) >= 0.2
                        else "negligible"
                    ),
                })

    return pd.DataFrame(rows)


def run_analysis():
    metrics_df = pd.read_csv(OUTPUT_DIR / "metrics.csv")

    summary = compute_summary_stats(metrics_df)
    summary.to_csv(OUTPUT_DIR / "summary_stats.csv", index=False)
    print(f"  Summary stats: {len(summary)} rows")

    tests = run_statistical_tests(metrics_df)
    tests.to_csv(OUTPUT_DIR / "statistical_tests.csv", index=False)

    sig_count = tests["significant_at_05"].sum()
    print(f"  Statistical tests: {len(tests)} comparisons, {sig_count} significant (p<0.05)")

    return summary, tests
