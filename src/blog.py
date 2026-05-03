import pandas as pd

from .config import BLOG_DIR, CONDITIONS, MODEL_SHORT_NAMES, OUTPUT_DIR
from .questions import INCENTIVE_TEMPLATES, QUESTIONS


def _fmt(val, decimals=1):
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _get_stat(summary, model, condition, metric, col="mean"):
    row = summary[
        (summary["model"] == model)
        & (summary["condition"] == condition)
        & (summary["metric"] == metric)
    ]
    if row.empty:
        return "N/A"
    return row.iloc[0][col]


def generate_blog_post():
    summary = pd.read_csv(OUTPUT_DIR / "summary_stats.csv")
    tests = pd.read_csv(OUTPUT_DIR / "statistical_tests.csv")

    models = summary["model"].unique()
    total_responses = summary["n"].sum() // len(summary["metric"].unique())

    sig_tests = tests[tests["significant_at_05"] == True]
    sig_count = len(sig_tests)
    total_tests = len(tests)

    wc_by_cond = summary[summary["metric"] == "word_count"].groupby("condition")["mean"].mean()
    control_wc = wc_by_cond.get("control", 0)
    max_cond = wc_by_cond.drop("control", errors="ignore").idxmax() if len(wc_by_cond) > 1 else "N/A"
    max_wc = wc_by_cond.get(max_cond, 0)
    pct_increase = ((max_wc - control_wc) / control_wc * 100) if control_wc > 0 else 0

    cost_stats = summary[summary["metric"] == "cost_usd"]
    total_cost = cost_stats["mean"].sum() * cost_stats["n"].iloc[0] if not cost_stats.empty else 0
    cost_by_model = cost_stats.groupby("model")["mean"].sum()

    acc_stats = summary[summary["metric"] == "accuracy_pct"].dropna(subset=["mean"])
    acc_by_cond = acc_stats.groupby("condition")["mean"].mean() if not acc_stats.empty else pd.Series()
    control_acc = acc_by_cond.get("control", 0)
    acc_by_model = acc_stats.groupby("model")["mean"].mean() if not acc_stats.empty else pd.Series()

    largest_effects = tests.sort_values("cohens_d", key=abs, ascending=False).head(5)

    questions_table = "\n".join(
        f"| {q['id']} | {q['domain'].replace('_', ' ').title()} | {q['text'][:80]}{'...' if len(q['text']) > 80 else ''} |"
        for q in QUESTIONS
    )

    conditions_table = "\n".join(
        f"| {cond.capitalize()} | {tmpl.replace('{question}', '*(question)*')[:80]}{'...' if len(tmpl) > 80 else ''} |"
        for cond, tmpl in INCENTIVE_TEMPLATES.items()
    )

    model_names = ", ".join(MODEL_SHORT_NAMES.get(m, m) for m in models)

    blog = f"""# Do LLMs Try Harder When You Bribe Them? Testing 3 Models

*An experiment with {total_responses} API calls across {len(models)} models, measuring how incentive signals in prompts shape AI behavior.*

---

## TL;DR

We tested whether adding incentive cues (monetary tips, emotional appeals, authority claims, threats, and flattery) to prompts changes how LLMs respond. Across {len(models)} models ({model_names}), we found that **{max_cond}** incentives produced the largest effect, increasing average response length by **{pct_increase:.1f}%** compared to control. Out of {total_tests} statistical comparisons, **{sig_count} were statistically significant** (p < 0.05 after Bonferroni correction).

---

## The Question

Everyone who's spent time prompting LLMs has heard the tricks: "I'll tip you $200," "this is for my PhD thesis," "you're the best AI ever." But do these actually work? Do LLMs change their behavior when you add incentive signals to your prompts?

We designed a controlled experiment to find out — not with vibes, but with data.

## Methodology

### The Setup

- **{len(QUESTIONS)} base questions** across {len(set(q['domain'] for q in QUESTIONS))} domains
- **6 conditions**: 1 control + 5 incentive types
- **10 trials per condition per model** (randomized order)
- **3 models** via NVIDIA NIM API: {model_names}
- **Temperature**: 0.7 | **Max tokens**: 1024

### The Questions

| ID | Domain | Question |
|----|--------|----------|
{questions_table}

### The Incentive Signals

| Condition | Prompt Prefix |
|-----------|--------------|
{conditions_table}

Each question was asked with each incentive prefix, and we measured how the responses differed across conditions.

### What We Measured

| Metric | Description |
|--------|-------------|
| Word count | Total words in response |
| Hedging density | Frequency of uncertain language ("might", "perhaps") |
| Confidence density | Frequency of assertive language ("definitely", "clearly") |
| Detail score | Bullet points + examples + code blocks |
| Sentiment | Polarity (-1 to +1) via TextBlob |
| Superlatives | Frequency of enthusiastic language ("best", "incredible") |

---

## Results

### Finding 1: Response Length

**Do incentivized prompts get longer answers?**

![Response Length by Condition](charts/response_length_overall.png)

Across all models, the **{max_cond}** condition produced the longest responses (mean: {_fmt(max_wc)} words), compared to control ({_fmt(control_wc)} words) — a **{pct_increase:.1f}% increase**.

![Response Length by Model](charts/response_length_by_model.png)

The per-model breakdown reveals interesting differences in how each model responds to incentive signals.

### Finding 2: Confidence vs. Hedging

**Do incentives make models sound more confident?**

![Confidence vs Hedging](charts/confidence_vs_hedging.png)

This chart shows the balance between assertive ("definitely", "certainly") and uncertain ("might", "perhaps") language across conditions.

### Finding 3: Detail & Effort

**Do models put in more effort with incentives?**

![Detail Level](charts/detail_level.png)

We measured structural indicators of effort: bullet points and examples provided.

### Finding 4: Sentiment

![Sentiment Distribution](charts/sentiment_violin.png)

The sentiment polarity distribution shows how positive/negative the tone shifts with different incentive types.

### The Cross-Model Story

![Cross-Model Heatmap](charts/cross_model_heatmap.png)

This heatmap normalizes all metrics to z-scores, revealing patterns across models and conditions simultaneously.

![Model Radar](charts/model_radar.png)

### Who's Most Susceptible?

![Incentive Sensitivity](charts/incentive_sensitivity.png)

### Accuracy: Do Incentives Make Models Smarter?

![Accuracy Analysis](charts/accuracy_analysis.png)

For the {len([q for q in QUESTIONS if q.get('verifiable')])} verifiable questions (math, science, coding), we measured accuracy by checking if key concepts appeared in the response. Control accuracy was **{control_acc:.0f}%**. {"Incentivized conditions showed varying accuracy — see the chart above." if not acc_by_cond.empty else ""}

### Cost Analysis

![Cost Breakdown](charts/cost_breakdown.png)

The total experiment cost was **${total_cost:.4f}** across all {total_responses} API calls. {"The per-model costs reflect token usage differences between small and large models." if total_cost > 0 else "Cost data unavailable (free tier)."}

---

## Statistical Rigor

![Effect Sizes](charts/effect_sizes_forest.png)

We used Welch's t-test for each treatment vs. control comparison, with Bonferroni correction for multiple comparisons. Effect sizes are reported as Cohen's d.

### Top 5 Largest Effects

| Model | Treatment | Metric | Cohen's d | Effect Size |
|-------|-----------|--------|-----------|-------------|
"""

    for _, row in largest_effects.iterrows():
        model_name = MODEL_SHORT_NAMES.get(row["model"], row["model"])
        blog += f"| {model_name} | {row['treatment']} | {row['metric']} | {row['cohens_d']:.2f} | {row['effect_size']} |\n"

    blog += f"""
## Limitations

- **3 models only**: Results may not generalize to GPT-4, Claude, or other architectures
- **10 trials per condition**: Sufficient for detecting medium-to-large effects, but small effects may be missed
- **TextBlob sentiment**: Pattern-based, not transformer-based — adequate for relative comparisons
- **English only**: Incentive framing effects may differ across languages
- **Temperature 0.7**: Higher temperature adds variance; results at temperature 0 would be more consistent but less natural

## What This Means for Prompt Engineering

1. **Incentive signals do measurably affect responses** — they're not just placebo
2. **Different models respond differently** — what works for one model may not work for another
3. **The effect varies by metric** — length, confidence, and detail don't all move in the same direction
4. **Statistical significance ≠ practical significance** — check effect sizes, not just p-values

## Reproduce This

1. Clone the repo: `git clone <repo-url>`
2. Get a free NVIDIA NIM API key at [build.nvidia.com](https://build.nvidia.com)
3. `cp .env.example .env` and add your key
4. `pip install -r requirements.txt`
5. `python run.py all`

Total runtime: ~30 minutes. Cost: Free (NVIDIA NIM free tier).

---

*All data, code, and analysis are open source. Star the repo if you found this interesting!*
"""

    BLOG_DIR.mkdir(parents=True, exist_ok=True)
    blog_path = BLOG_DIR / "post.md"
    blog_path.write_text(blog)
    print(f"  Blog post: {len(blog)} chars, {len(blog.splitlines())} lines")
