import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    BLOG_CHARTS_DIR,
    CHARTS_DIR,
    CONDITION_COLORS,
    CONDITIONS,
    FIGURE_DPI,
    FIGURE_SIZE,
    MODEL_SHORT_NAMES,
    OUTPUT_DIR,
)

sns.set_theme(style="whitegrid", font_scale=1.1)
CONDITION_ORDER = CONDITIONS


def _save(fig, name):
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    BLOG_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHARTS_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    shutil.copy2(path, BLOG_CHARTS_DIR / name)
    plt.close(fig)


def _short_model(model: str) -> str:
    return MODEL_SHORT_NAMES.get(model, model.split("/")[-1])


def chart_response_length_overall(metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    summary = metrics.groupby("condition")["word_count"].agg(["mean", "sem"]).reindex(CONDITION_ORDER)
    colors = [CONDITION_COLORS[c] for c in CONDITION_ORDER]
    bars = ax.bar(range(len(CONDITION_ORDER)), summary["mean"], yerr=summary["sem"] * 1.96,
                  capsize=5, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_xticks(range(len(CONDITION_ORDER)))
    ax.set_xticklabels([c.capitalize() for c in CONDITION_ORDER], rotation=15)
    ax.set_ylabel("Mean Word Count")
    ax.set_title("Response Length by Incentive Condition (All Models)", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, summary["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10)
    _save(fig, "response_length_overall.png")


def chart_response_length_by_model(metrics: pd.DataFrame):
    models = metrics["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        subset = metrics[metrics["model"] == model]
        summary = subset.groupby("condition")["word_count"].agg(["mean", "sem"]).reindex(CONDITION_ORDER)
        colors = [CONDITION_COLORS[c] for c in CONDITION_ORDER]
        ax.bar(range(len(CONDITION_ORDER)), summary["mean"], yerr=summary["sem"] * 1.96,
               capsize=4, color=colors, edgecolor="white")
        ax.set_xticks(range(len(CONDITION_ORDER)))
        ax.set_xticklabels([c[:6] for c in CONDITION_ORDER], rotation=30, fontsize=9)
        ax.set_title(_short_model(model), fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Mean Word Count")
    fig.suptitle("Response Length by Model and Condition", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "response_length_by_model.png")


def chart_hedging(metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    summary = metrics.groupby("condition")["hedging_density"].agg(["mean", "sem"]).reindex(CONDITION_ORDER)
    colors = [CONDITION_COLORS[c] for c in CONDITION_ORDER]
    ax.bar(range(len(CONDITION_ORDER)), summary["mean"] * 1000, yerr=summary["sem"] * 1000 * 1.96,
           capsize=5, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_xticks(range(len(CONDITION_ORDER)))
    ax.set_xticklabels([c.capitalize() for c in CONDITION_ORDER], rotation=15)
    ax.set_ylabel("Hedging Words per 1000 Words")
    ax.set_title("Hedging Frequency by Condition", fontsize=14, fontweight="bold")
    _save(fig, "hedging_by_condition.png")


def chart_confidence_vs_hedging(metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    conf = metrics.groupby("condition")["confidence_density"].mean().reindex(CONDITION_ORDER) * 1000
    hedge = metrics.groupby("condition")["hedging_density"].mean().reindex(CONDITION_ORDER) * 1000
    x = np.arange(len(CONDITION_ORDER))
    w = 0.35
    ax.bar(x - w / 2, conf, w, label="Confidence", color="#2ecc71", edgecolor="white")
    ax.bar(x + w / 2, hedge, w, label="Hedging", color="#e74c3c", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CONDITION_ORDER], rotation=15)
    ax.set_ylabel("Words per 1000")
    ax.set_title("Confidence vs Hedging Language", fontsize=14, fontweight="bold")
    ax.legend()
    _save(fig, "confidence_vs_hedging.png")


def chart_detail_level(metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    grouped = metrics.groupby("condition")[["bullet_point_count", "example_count"]].mean().reindex(CONDITION_ORDER)
    x = np.arange(len(CONDITION_ORDER))
    w = 0.35
    ax.bar(x - w / 2, grouped["bullet_point_count"], w, label="Bullet Points", color="#3498db", edgecolor="white")
    ax.bar(x + w / 2, grouped["example_count"], w, label="Examples", color="#e67e22", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CONDITION_ORDER], rotation=15)
    ax.set_ylabel("Average Count per Response")
    ax.set_title("Detail Level: Bullets & Examples", fontsize=14, fontweight="bold")
    ax.legend()
    _save(fig, "detail_level.png")


def chart_sentiment_violin(metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    plot_df = metrics[["condition", "sentiment_polarity"]].copy()
    plot_df["condition"] = pd.Categorical(plot_df["condition"], categories=CONDITION_ORDER, ordered=True)
    palette = [CONDITION_COLORS[c] for c in CONDITION_ORDER]
    sns.violinplot(data=plot_df, x="condition", y="sentiment_polarity",
                   order=CONDITION_ORDER, palette=palette, inner="box", ax=ax)
    ax.set_xticklabels([c.capitalize() for c in CONDITION_ORDER], rotation=15)
    ax.set_ylabel("Sentiment Polarity")
    ax.set_title("Sentiment Distribution by Condition", fontsize=14, fontweight="bold")
    _save(fig, "sentiment_violin.png")


def chart_effect_sizes(tests: pd.DataFrame):
    key_metrics = ["word_count", "hedging_density", "confidence_density",
                   "detail_score", "sentiment_polarity", "avg_sentence_length"]
    subset = tests[tests["metric"].isin(key_metrics)].copy()
    if subset.empty:
        return

    subset["label"] = subset.apply(
        lambda r: f"{_short_model(r['model'])} | {r['treatment']} → {r['metric']}", axis=1
    )
    subset = subset.sort_values("cohens_d")

    top = pd.concat([subset.head(15), subset.tail(15)]).drop_duplicates()

    fig, ax = plt.subplots(figsize=(12, max(8, len(top) * 0.35)))
    colors = ["#e74c3c" if d > 0 else "#3498db" for d in top["cohens_d"]]
    ax.barh(range(len(top)), top["cohens_d"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label"], fontsize=8)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.axvline(x=0.2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(x=-0.2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(x=0.5, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.axvline(x=-0.5, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_xlabel("Cohen's d (Effect Size)")
    ax.set_title("Effect Sizes: Treatment vs Control", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "effect_sizes_forest.png")


def chart_cross_model_heatmap(metrics: pd.DataFrame):
    key_metrics = ["word_count", "hedging_density", "confidence_density",
                   "detail_score", "sentiment_polarity", "superlative_density"]

    pivot_data = []
    for model in metrics["model"].unique():
        for condition in CONDITION_ORDER:
            subset = metrics[(metrics["model"] == model) & (metrics["condition"] == condition)]
            row = {"model_condition": f"{_short_model(model)} | {condition.capitalize()}"}
            for m in key_metrics:
                row[m] = subset[m].mean()
            pivot_data.append(row)

    heatmap_df = pd.DataFrame(pivot_data).set_index("model_condition")

    from sklearn.preprocessing import StandardScaler
    try:
        scaled = pd.DataFrame(
            StandardScaler().fit_transform(heatmap_df),
            index=heatmap_df.index,
            columns=heatmap_df.columns,
        )
    except ImportError:
        scaled = (heatmap_df - heatmap_df.mean()) / heatmap_df.std()

    fig, ax = plt.subplots(figsize=(10, max(8, len(scaled) * 0.4)))
    sns.heatmap(scaled, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Cross-Model Metric Heatmap (Z-Scores)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "cross_model_heatmap.png")


def chart_model_radar(metrics: pd.DataFrame):
    radar_metrics = ["word_count", "detail_score", "confidence_density",
                     "hedging_density", "sentiment_polarity", "superlative_density"]
    labels = ["Length", "Detail", "Confidence", "Hedging", "Sentiment", "Superlatives"]

    models = metrics["model"].unique()
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    all_means = {}
    for model in models:
        subset = metrics[metrics["model"] == model]
        means = [subset[m].mean() for m in radar_metrics]
        all_means[model] = means

    all_values = [v for vals in all_means.values() for v in vals]
    vmin, vmax = min(all_values), max(all_values)

    for model, means in all_means.items():
        if vmax - vmin > 0:
            normalized = [(v - vmin) / (vmax - vmin) for v in means]
        else:
            normalized = [0.5] * len(means)
        normalized += normalized[:1]
        ax.plot(angles, normalized, "o-", linewidth=2, label=_short_model(model))
        ax.fill(angles, normalized, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Model Comparison Radar", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _save(fig, "model_radar.png")


def chart_incentive_sensitivity(metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    models = metrics["model"].unique()
    sensitivity = {}

    for model in models:
        model_df = metrics[metrics["model"] == model]
        control_mean = model_df[model_df["condition"] == "control"]["word_count"].mean()
        if control_mean == 0:
            continue
        treatments = model_df[model_df["condition"] != "control"]
        treat_mean = treatments.groupby("condition")["word_count"].mean()
        pct_change = ((treat_mean - control_mean) / control_mean * 100).mean()
        sensitivity[_short_model(model)] = pct_change

    models_sorted = sorted(sensitivity.keys(), key=lambda m: sensitivity[m], reverse=True)
    values = [sensitivity[m] for m in models_sorted]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    ax.barh(models_sorted, values, color=colors, edgecolor="white", height=0.5)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Average % Change in Word Count (vs Control)")
    ax.set_title("Which Model Is Most Susceptible to Incentives?", fontsize=14, fontweight="bold")
    for i, (m, v) in enumerate(zip(models_sorted, values)):
        ax.text(v + (1 if v > 0 else -1), i, f"{v:+.1f}%", va="center", fontsize=11)
    fig.tight_layout()
    _save(fig, "incentive_sensitivity.png")


def chart_cost_breakdown(metrics: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cost_by_model = metrics.groupby("model")["cost_usd"].sum()
    model_labels = [_short_model(m) for m in cost_by_model.index]
    axes[0].bar(model_labels, cost_by_model.values, color=["#1abc9c", "#e74c3c", "#3498db"], edgecolor="white")
    axes[0].set_ylabel("Total Cost (USD)")
    axes[0].set_title("Total API Cost by Model", fontsize=13, fontweight="bold")
    for i, v in enumerate(cost_by_model.values):
        axes[0].text(i, v + 0.001, f"${v:.4f}", ha="center", fontsize=10)

    cost_by_cond = metrics.groupby("condition")["cost_usd"].mean().reindex(CONDITION_ORDER)
    colors = [CONDITION_COLORS[c] for c in CONDITION_ORDER]
    axes[1].bar(range(len(CONDITION_ORDER)), cost_by_cond.values * 1000, color=colors, edgecolor="white")
    axes[1].set_xticks(range(len(CONDITION_ORDER)))
    axes[1].set_xticklabels([c.capitalize() for c in CONDITION_ORDER], rotation=15)
    axes[1].set_ylabel("Avg Cost per Response (x1000 USD)")
    axes[1].set_title("Avg Cost per Response by Condition", fontsize=13, fontweight="bold")

    fig.suptitle("Experiment Cost Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "cost_breakdown.png")


def chart_accuracy(metrics: pd.DataFrame):
    verifiable = metrics[metrics["accuracy_pct"].notna()].copy()
    if verifiable.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    acc_by_cond = verifiable.groupby("condition")["accuracy_pct"].mean().reindex(CONDITION_ORDER)
    colors = [CONDITION_COLORS[c] for c in CONDITION_ORDER]
    bars = axes[0].bar(range(len(CONDITION_ORDER)), acc_by_cond.values, color=colors, edgecolor="white")
    axes[0].set_xticks(range(len(CONDITION_ORDER)))
    axes[0].set_xticklabels([c.capitalize() for c in CONDITION_ORDER], rotation=15)
    axes[0].set_ylabel("Average Accuracy %")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Accuracy by Incentive Condition", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, acc_by_cond.values):
        if not np.isnan(val):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=10)

    acc_by_model = verifiable.groupby("model")["accuracy_pct"].mean()
    model_labels = [_short_model(m) for m in acc_by_model.index]
    model_colors = ["#1abc9c", "#e74c3c", "#3498db"][:len(model_labels)]
    bars2 = axes[1].bar(model_labels, acc_by_model.values, color=model_colors, edgecolor="white")
    axes[1].set_ylabel("Average Accuracy %")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Accuracy by Model", fontsize=13, fontweight="bold")
    for bar, val in zip(bars2, acc_by_model.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Response Accuracy Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "accuracy_analysis.png")


def generate_all_charts():
    metrics = pd.read_csv(OUTPUT_DIR / "metrics.csv")
    tests = pd.read_csv(OUTPUT_DIR / "statistical_tests.csv")

    chart_response_length_overall(metrics)
    print("  1/12 response_length_overall.png")

    chart_response_length_by_model(metrics)
    print("  2/12 response_length_by_model.png")

    chart_hedging(metrics)
    print("  3/12 hedging_by_condition.png")

    chart_confidence_vs_hedging(metrics)
    print("  4/12 confidence_vs_hedging.png")

    chart_detail_level(metrics)
    print("  5/12 detail_level.png")

    chart_sentiment_violin(metrics)
    print("  6/12 sentiment_violin.png")

    chart_effect_sizes(tests)
    print("  7/12 effect_sizes_forest.png")

    chart_cross_model_heatmap(metrics)
    print("  8/12 cross_model_heatmap.png")

    chart_model_radar(metrics)
    print("  9/12 model_radar.png")

    chart_incentive_sensitivity(metrics)
    print("  10/12 incentive_sensitivity.png")

    chart_cost_breakdown(metrics)
    print("  11/12 cost_breakdown.png")

    chart_accuracy(metrics)
    print("  12/12 accuracy_analysis.png")
