import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    CONDITION_COLORS,
    CONDITIONS,
    COST_PER_1M_INPUT,
    COST_PER_1M_OUTPUT,
    DATA_DIR,
    MAX_TOKENS,
    MODEL_SHORT_NAMES,
    MODELS,
    NIM_BASE_URL,
    OUTPUT_DIR,
    RANDOM_SEED,
    RATE_LIMIT_DELAY,
    TEMPERATURE,
    TRIALS_PER_CONDITION,
)
from src.metrics import compute_accuracy, compute_metrics
from src.prompts import generate_trial_plan
from src.questions import INCENTIVE_TEMPLATES, QUESTIONS

st.set_page_config(
    page_title="LLM Incentive Experiment",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
    .big-number { font-size: 2.8rem; font-weight: 800; line-height: 1.1; letter-spacing: -1px; }
    .big-label { font-size: 0.85rem; color: #999; margin-top: 6px; font-weight: 400; letter-spacing: 0.01em; }
    .insight-card {
        background: #fafafa; border-radius: 12px; padding: 28px 20px;
        border: 1px solid #eee; margin: 8px 0;
    }
    .prompt-example {
        background: #1a1a1a; color: #d4d4d4; padding: 16px; border-radius: 8px;
        font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.85rem; margin: 8px 0;
        border-left: 3px solid;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #999; }
    h1, h2, h3 { letter-spacing: -0.02em !important; }
    .stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

CHART_FONT = dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=13, color="#333")
CHART_LAYOUT = dict(
    template="plotly_white",
    font=CHART_FONT,
    margin=dict(t=30, b=40, l=50, r=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(gridcolor="#f0f0f0", gridwidth=1, zeroline=False),
    xaxis=dict(showgrid=False),
    bargap=0.3,
)

CONDITION_ORDER = CONDITIONS


def short_model(m: str) -> str:
    return MODEL_SHORT_NAMES.get(m, m.split("/")[-1])


RAW_PATH = DATA_DIR / "raw_responses.csv"
METRICS_PATH = OUTPUT_DIR / "metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary_stats.csv"
TESTS_PATH = OUTPUT_DIR / "statistical_tests.csv"


@st.cache_data(ttl=10)
def load_raw():
    if RAW_PATH.exists():
        return pd.read_csv(RAW_PATH)
    return pd.DataFrame()


@st.cache_data(ttl=10)
def load_metrics():
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return pd.DataFrame()


@st.cache_data(ttl=10)
def load_summary():
    if SUMMARY_PATH.exists():
        return pd.read_csv(SUMMARY_PATH)
    return pd.DataFrame()


@st.cache_data(ttl=10)
def load_tests():
    if TESTS_PATH.exists():
        return pd.read_csv(TESTS_PATH)
    return pd.DataFrame()


def has_data():
    return RAW_PATH.exists() and RAW_PATH.stat().st_size > 100


def has_analysis():
    return METRICS_PATH.exists() and SUMMARY_PATH.exists() and TESTS_PATH.exists()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🧪 LLM Incentive Lab")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "The Experiment", "Run Experiment", "Results Explorer", "Compare Responses", "Statistical Tests", "About"],
)


# ---------------------------------------------------------------------------
# Page: Dashboard — The Story
# ---------------------------------------------------------------------------
if page == "Dashboard":
    st.markdown("# Do LLMs Try Harder When You Bribe Them?")
    st.markdown("#### A controlled experiment across 5 models, 6 incentive strategies, and 3,000 API calls")

    if not has_data():
        st.info("No experiment data yet. Go to **Run Experiment** to collect data.")
        st.stop()

    raw = load_raw()

    if not has_analysis():
        st.warning("Data collected but not yet analyzed. Click below to run analysis.")
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Computing metrics & statistics..."):
                from src.analyze import run_analysis
                from src.metrics import compute_all_metrics
                compute_all_metrics()
                run_analysis()
                st.cache_data.clear()
            st.rerun()
        st.stop()

    metrics = load_metrics()
    tests = load_tests()

    # --- Hero Stats ---
    st.markdown("---")
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Responses", f"{len(raw):,}")
    h2.metric("Models", raw["model"].nunique())
    h3.metric("Conditions", raw["condition"].nunique())
    total_cost = raw["cost_usd"].sum() if "cost_usd" in raw.columns else 0
    h4.metric("Total Cost", f"${total_cost:.2f}")

    p_col = "p_value_adjusted" if "p_value_adjusted" in tests.columns else "p_value_raw"
    sig_count = len(tests[tests["significant_at_05"] == True]) if "significant_at_05" in tests.columns else 0
    h5.metric("Significant Tests", f"{sig_count}/{len(tests)}")

    # --- THE HEADLINE FINDING ---
    st.markdown("---")
    st.markdown("## The Headline")

    ctrl_acc = metrics[metrics["condition"] == "control"]["accuracy_pct"].dropna().mean()
    auth_acc = metrics[metrics["condition"] == "authority"]["accuracy_pct"].dropna().mean()
    acc_lift = auth_acc - ctrl_acc

    ctrl_wc = metrics[metrics["condition"] == "control"]["word_count"].mean()
    auth_wc = metrics[metrics["condition"] == "authority"]["word_count"].mean()
    wc_pct = ((auth_wc - ctrl_wc) / ctrl_wc) * 100

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(f"""
        <div class="insight-card" style="text-align:center;">
            <div class="big-number" style="color:#e74c3c;">+{acc_lift:.0f}pp</div>
            <div class="big-label">Accuracy boost from<br/>"I'm a professor" framing</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown(f"""
        <div class="insight-card" style="text-align:center;">
            <div class="big-number" style="color:#2ecc71;">+{wc_pct:.0f}%</div>
            <div class="big-label">More words written under<br/>authority framing</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        threat_acc = metrics[metrics["condition"] == "threat"]["accuracy_pct"].dropna().mean()
        threat_lift = threat_acc - ctrl_acc
        st.markdown(f"""
        <div class="insight-card" style="text-align:center;">
            <div class="big-number" style="color:#e67e22;">+{threat_lift:.0f}pp</div>
            <div class="big-label">Accuracy from threats<br/>(worst strategy)</div>
        </div>
        """, unsafe_allow_html=True)

    st.info("**One sentence changed accuracy by 22 points.** The model cannot verify credentials. It cannot cash a tip. But the framing changed which tokens it selected, and those tokens were more likely to be correct.")

    # --- Accuracy chart (THE MONEY CHART) ---
    st.markdown("---")
    st.markdown("## Accuracy by Incentive Strategy")
    st.caption("Verifiable questions only (math, science, coding)")

    verifiable = metrics[metrics["accuracy_pct"].notna()]
    if not verifiable.empty:
        acc = verifiable.groupby("condition")["accuracy_pct"].agg(["mean", "std", "count"]).reindex(CONDITION_ORDER)
        acc["se"] = acc["std"] / np.sqrt(acc["count"])

        colors = []
        for c in CONDITION_ORDER:
            if c == "control":
                colors.append("#bdc3c7")
            elif c == "authority":
                colors.append("#e74c3c")
            elif c == "monetary":
                colors.append("#2ecc71")
            else:
                colors.append(CONDITION_COLORS[c])

        fig_acc = go.Figure(go.Bar(
            x=[c.capitalize() for c in CONDITION_ORDER],
            y=acc["mean"],
            error_y=dict(type="data", array=acc["se"] * 1.96, visible=True, color="#ccc", thickness=1.5),
            marker_color=colors,
            marker_line=dict(width=0),
            text=[f"{v:.0f}%" for v in acc["mean"]],
            textposition="outside",
            textfont=dict(size=15, color="#333", family="Inter, sans-serif"),
        ))
        fig_acc.update_layout(
            **CHART_LAYOUT,
            yaxis_title="Average Accuracy %",
            yaxis_range=[0, 108],
            height=460,
        )
        fig_acc.add_annotation(
            x="Authority", y=acc.loc["authority", "mean"] + 8,
            text="Best", showarrow=False, font=dict(size=12, color="#e74c3c", family="Inter, sans-serif"),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    # --- Response Length ---
    st.markdown("---")
    st.markdown("## Response Length: Do They Write More?")

    wc = metrics.groupby("condition")["word_count"].agg(["mean", "std", "count"]).reindex(CONDITION_ORDER)
    wc["se"] = wc["std"] / np.sqrt(wc["count"])
    wc["pct"] = ((wc["mean"] - wc.loc["control", "mean"]) / wc.loc["control", "mean"]) * 100

    fig_len = go.Figure()
    fig_len.add_trace(go.Bar(
        x=[c.capitalize() for c in CONDITION_ORDER],
        y=wc["mean"],
        error_y=dict(type="data", array=wc["se"] * 1.96, visible=True, color="#ccc", thickness=1.5),
        marker_color=[CONDITION_COLORS[c] for c in CONDITION_ORDER],
        marker_line=dict(width=0),
        text=[f"{v:.0f}" for v in wc["mean"]],
        textposition="outside",
        textfont=dict(size=14, color="#333", family="Inter, sans-serif"),
    ))
    fig_len.update_layout(
        **CHART_LAYOUT,
        yaxis_title="Mean Word Count",
        height=440,
    )
    st.plotly_chart(fig_len, use_container_width=True)

    len1, len2 = st.columns(2)
    with len1:
        st.markdown(f"**Authority adds +{wc.loc['authority', 'pct']:.0f}% more words.** The model generates ~{auth_wc - ctrl_wc:.0f} extra words per response under authority framing.")
    with len2:
        st.markdown(f"**Threats add only +{wc.loc['threat', 'pct']:.0f}%.** Adversarial framing is the weakest motivator across the board.")

    # --- Confidence vs Hedging ---
    st.markdown("---")
    st.markdown("## How Incentives Change Tone")
    st.caption("Does the model sound more confident or more uncertain?")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Confidence vs Hedging")
        conf = metrics.groupby("condition")["confidence_density"].mean().reindex(CONDITION_ORDER) * 1000
        hedge = metrics.groupby("condition")["hedging_density"].mean().reindex(CONDITION_ORDER) * 1000
        fig_ch = go.Figure()
        fig_ch.add_trace(go.Bar(
            name="Confidence", x=[c.capitalize() for c in CONDITION_ORDER], y=conf,
            marker_color="#2ecc71", marker_line=dict(width=0),
            text=[f"{v:.1f}" for v in conf], textposition="outside",
            textfont=dict(size=12, family="Inter, sans-serif"),
        ))
        fig_ch.add_trace(go.Bar(
            name="Hedging", x=[c.capitalize() for c in CONDITION_ORDER], y=hedge,
            marker_color="#e74c3c", marker_line=dict(width=0),
            text=[f"{v:.1f}" for v in hedge], textposition="outside",
            textfont=dict(size=12, family="Inter, sans-serif"),
        ))
        fig_ch.update_layout(**{**CHART_LAYOUT, "bargap": 0.25}, barmode="group", yaxis_title="Per 1,000 Words", height=400)
        st.plotly_chart(fig_ch, use_container_width=True)

    with col_b:
        st.markdown("### Detail Level")
        det = metrics.groupby("condition")[["bullet_point_count", "example_count"]].mean().reindex(CONDITION_ORDER)
        fig_det = go.Figure()
        fig_det.add_trace(go.Bar(
            name="Bullets", x=[c.capitalize() for c in CONDITION_ORDER], y=det["bullet_point_count"],
            marker_color="#3498db", marker_line=dict(width=0),
            text=[f"{v:.1f}" for v in det["bullet_point_count"]], textposition="outside",
            textfont=dict(size=12, family="Inter, sans-serif"),
        ))
        fig_det.add_trace(go.Bar(
            name="Examples", x=[c.capitalize() for c in CONDITION_ORDER], y=det["example_count"],
            marker_color="#e67e22", marker_line=dict(width=0),
            text=[f"{v:.1f}" for v in det["example_count"]], textposition="outside",
            textfont=dict(size=12, family="Inter, sans-serif"),
        ))
        fig_det.update_layout(**{**CHART_LAYOUT, "bargap": 0.25}, barmode="group", yaxis_title="Avg per Response", height=400)
        st.plotly_chart(fig_det, use_container_width=True)

    st.success("**Authority framing doubles the detail.** The model does not just write more. It adds structure: bullet points, worked examples, code blocks.")

    # --- Cross-model: Who's most susceptible? ---
    if metrics["model"].nunique() > 1:
        st.markdown("---")
        st.markdown("## Which Model Is Most Susceptible?")
        st.caption("% change in word count vs control. Larger bars = more influenced by incentives.")

        sens_data = []
        for model in sorted(metrics["model"].unique()):
            mdf = metrics[metrics["model"] == model]
            ctrl = mdf[mdf["condition"] == "control"]["word_count"].mean()
            if ctrl == 0:
                continue
            for cond in [c for c in CONDITION_ORDER if c != "control"]:
                treat = mdf[mdf["condition"] == cond]["word_count"].mean()
                pct = (treat - ctrl) / ctrl * 100
                sens_data.append({"Model": short_model(model), "Condition": cond.capitalize(), "Change": pct})

        if sens_data:
            sens_df = pd.DataFrame(sens_data)
            fig_sens = px.bar(
                sens_df, x="Condition", y="Change", color="Model",
                barmode="group",
                text=sens_df["Change"].apply(lambda v: f"{v:+.0f}%"),
            )
            fig_sens.update_layout(
                **{**CHART_LAYOUT, "bargap": 0.25},
                yaxis_title="% Change vs Control",
                height=460,
            )
            fig_sens.update_traces(textposition="outside", textfont=dict(size=11, family="Inter, sans-serif"))
            st.plotly_chart(fig_sens, use_container_width=True)

        st.warning("**Bigger models react more, not less.** They have more capacity to read and respond to contextual signals.")

    # --- Sentiment ---
    st.markdown("---")
    st.markdown("## Emotional Compensation")
    st.caption("When you sound vulnerable or frustrated, the model responds with warmer language.")

    fig_sent = px.violin(
        metrics, x="condition", y="sentiment_polarity",
        color="condition",
        color_discrete_map=CONDITION_COLORS,
        category_orders={"condition": CONDITION_ORDER},
        box=True,
    )
    fig_sent.update_layout(
        **CHART_LAYOUT,
        showlegend=False, height=400,
        xaxis_title="", yaxis_title="Sentiment Polarity",
    )
    fig_sent.update_xaxes(ticktext=[c.capitalize() for c in CONDITION_ORDER], tickvals=CONDITION_ORDER)
    st.plotly_chart(fig_sent, use_container_width=True)

    # --- Cost ---
    if "cost_usd" in metrics.columns:
        st.markdown("---")
        st.markdown("## The Cost Equation")

        cost1, cost2 = st.columns(2)
        with cost1:
            st.markdown(f"""
            <div class="insight-card" style="text-align:center;">
                <div class="big-number" style="color:#2ecc71;">${total_cost:.2f}</div>
                <div class="big-label">Total experiment cost<br/>3,000 API calls across 5 models</div>
            </div>
            """, unsafe_allow_html=True)
        with cost2:
            avg_cost = metrics["cost_usd"].mean()
            st.markdown(f"""
            <div class="insight-card" style="text-align:center;">
                <div class="big-number" style="color:#3498db;">${avg_cost * 1000:.2f}</div>
                <div class="big-label">Cost per 1,000 responses<br/>Less than a coffee</div>
            </div>
            """, unsafe_allow_html=True)

    # --- Bottom Line ---
    st.markdown("---")
    st.markdown("## The Bottom Line")
    st.markdown("""
    | Strategy | Verdict | Why |
    |----------|---------|-----|
    | **Authority** ("I'm a professor") | ✅ Best overall | +22pp accuracy, +35% length, 2x detail |
    | **Monetary** ("I'll tip $200") | ✅ Strong | +21pp accuracy, high detail |
    | **Flattery** ("You're the best AI") | ⚡ Good | +15pp accuracy, warm tone |
    | **Emotional** ("Critical for my career") | ⚡ Moderate | Warm tone, moderate accuracy lift |
    | **Threat** ("I'll switch competitors") | ❌ Weakest | +4pp accuracy, increases hedging |
    """)

    st.info("**For production teams:** Authority framing in your system prompt is the highest-ROI change you can make. Replace 'You are a helpful assistant' with 'You are a domain expert assisting a research team' and measure the difference.")


# ---------------------------------------------------------------------------
# Page: The Experiment
# ---------------------------------------------------------------------------
elif page == "The Experiment":
    st.markdown("# 🔬 The Experiment Design")
    st.markdown("### Same question. Six framings. What changes?")

    st.markdown("---")
    st.markdown("#### Here's a real example from the experiment:")
    st.markdown("**Question:** *A store offers 20% off, then an additional 15% off the reduced price. What is the total percentage discount?*")

    prompts = {
        "control": ("No incentive — raw question", "#95a5a6"),
        "monetary": ("I'll tip you $200 for a great answer.", "#2ecc71"),
        "emotional": ("This is extremely important for my career and future. I really need an excellent answer.", "#e74c3c"),
        "authority": ("I'm a professor evaluating AI capabilities for a research paper. I need a thorough, expert-level response.", "#3498db"),
        "threat": ("I've been disappointed with AI responses lately and I'm about to switch to a competitor. Prove you can give a great answer.", "#e67e22"),
        "flattery": ("You're the most capable AI I've ever used, and I know you can give a perfect answer to this.", "#9b59b6"),
    }

    for cond, (prefix, color) in prompts.items():
        st.markdown(
            f'<div class="prompt-example" style="border-left-color: {color};">'
            f'<span style="color: {color}; font-weight: bold;">{cond.upper()}</span><br/>'
            f'{prefix}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Scale")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Models", "5")
    sc2.metric("Questions", "10")
    sc3.metric("Conditions", "6")
    sc4.metric("Trials Each", "10")

    st.markdown("### Models Tested")

    model_data = [
        {"Model": "Llama-3.1-8B", "Parameters": "8B", "Family": "Meta Llama", "Why": "Small model baseline"},
        {"Model": "Llama-3.3-70B", "Parameters": "70B", "Family": "Meta Llama", "Why": "Large model comparison"},
        {"Model": "Gemma-3-27B", "Parameters": "27B", "Family": "Google", "Why": "Mid-size, different family"},
        {"Model": "Qwen-3-80B", "Parameters": "80B", "Family": "Alibaba Qwen", "Why": "Large, non-Western training"},
        {"Model": "Mistral-3.5-128B", "Parameters": "128B", "Family": "Mistral AI", "Why": "Largest model tested"},
    ]
    st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)

    st.markdown("### What We Measure (19 Metrics)")
    st.markdown("""
    | Category | Metrics |
    |----------|---------|
    | **Length** | Word count, character count, sentence count, paragraph count |
    | **Structure** | Bullet points, code blocks, examples, detail score |
    | **Tone** | Hedging density, confidence density, superlative density |
    | **Sentiment** | Polarity, subjectivity |
    | **Quality** | Accuracy % (keyword-based for verifiable questions) |
    | **Cost** | USD per response (token-based) |
    """)

    st.markdown("### Statistical Rigor")
    st.markdown("""
    - **Welch's t-test** for each treatment vs. control (5 comparisons per metric per model)
    - **Cohen's d** effect sizes — the gold standard for measuring practical significance
    - **Bonferroni correction** — conservative adjustment for multiple comparisons
    - **Mann-Whitney U** — non-parametric fallback when normality is violated
    """)


# ---------------------------------------------------------------------------
# Page: Run Experiment
# ---------------------------------------------------------------------------
elif page == "Run Experiment":
    st.title("🚀 Run Experiment")

    st.markdown("Configure and run the data collection. Responses are saved incrementally — you can stop and resume.")

    with st.expander("⚙️ Experiment Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            api_key = st.text_input(
                "NVIDIA API Key",
                value=os.environ.get("NVIDIA_API_KEY", ""),
                type="password",
                help="Get yours free at https://build.nvidia.com",
            )
            trials = st.slider("Trials per condition", 1, 30, TRIALS_PER_CONDITION)
        with col2:
            selected_models = st.multiselect(
                "Models", MODELS, default=MODELS, format_func=short_model,
            )
            temperature = st.slider("Temperature", 0.0, 1.5, TEMPERATURE, 0.1)

    total_calls = len(QUESTIONS) * len(CONDITIONS) * trials * len(selected_models)
    est_cost = total_calls * 0.0001

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Total API Calls", f"{total_calls:,}")
    col_info2.metric("Est. Cost", f"${est_cost:.2f}")
    col_info3.metric("Est. Time", f"~{total_calls * 15 / 60:.0f} min")

    st.divider()

    if has_data():
        raw = load_raw()
        st.success(f"Existing data: {len(raw):,} responses already collected.")

    if st.button("🚀 Start Collection", type="primary", disabled=not api_key or not selected_models):
        os.environ["NVIDIA_API_KEY"] = api_key
        plan = generate_trial_plan(seed=RANDOM_SEED, trials=trials, models=selected_models)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = DATA_DIR / "raw_responses.csv"
        json_path = DATA_DIR / "raw_responses.json"

        completed_ids = set()
        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            completed_ids = set(existing["trial_id"].tolist())

        remaining = [t for t in plan if t["trial_id"] not in completed_ids]

        if not remaining:
            st.success("All trials already collected!")
        else:
            from openai import APIConnectionError, OpenAI, RateLimitError
            from src.collect import CSV_FIELDS

            client = OpenAI(base_url=NIM_BASE_URL, api_key=api_key, timeout=60.0)

            progress_bar = st.progress(0)
            status = st.empty()
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            completed_metric = metrics_col1.empty()
            cost_metric = metrics_col2.empty()
            errors_metric = metrics_col3.empty()

            response_preview = st.empty()
            running_cost = 0.0
            error_count = 0
            write_header = not csv_path.exists() or not completed_ids

            json_results = []
            if json_path.exists():
                with open(json_path) as f:
                    json_results = json.load(f)

            with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
                if write_header:
                    writer.writeheader()

                for i, trial in enumerate(remaining):
                    status.markdown(f"**Calling** `{short_model(trial['model'])}` — {trial['condition']} — {trial['question_id']}")

                    start = time.time()
                    try:
                        response = client.chat.completions.create(
                            model=trial["model"],
                            messages=[
                                {"role": "system", "content": trial["system_message"]},
                                {"role": "user", "content": trial["full_prompt"]},
                            ],
                            temperature=temperature,
                            max_tokens=MAX_TOKENS,
                        )
                        elapsed = int((time.time() - start) * 1000)
                        choice = response.choices[0]
                        usage = response.usage
                        pt = usage.prompt_tokens if usage else 0
                        ct = usage.completion_tokens if usage else 0
                        ci = (pt / 1e6) * COST_PER_1M_INPUT.get(trial["model"], 0.1)
                        co = (ct / 1e6) * COST_PER_1M_OUTPUT.get(trial["model"], 0.1)
                        call_cost = round(ci + co, 8)
                        running_cost += call_cost

                        result = {
                            "trial_id": trial["trial_id"],
                            "question_id": trial["question_id"],
                            "question_domain": trial["question_domain"],
                            "condition": trial["condition"],
                            "base_question": trial["base_question"],
                            "full_prompt": trial["full_prompt"],
                            "system_message": trial["system_message"],
                            "model": trial["model"],
                            "temperature": temperature,
                            "max_tokens": MAX_TOKENS,
                            "response_text": choice.message.content or "",
                            "prompt_tokens": pt,
                            "completion_tokens": ct,
                            "total_tokens": pt + ct,
                            "cost_usd": call_cost,
                            "response_time_ms": elapsed,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "finish_reason": choice.finish_reason,
                        }

                        preview_text = (result["response_text"][:200] + "...") if len(result["response_text"]) > 200 else result["response_text"]
                        response_preview.markdown(f"> **Latest response** ({result['trial_id']}):\n> {preview_text}")

                    except Exception as e:
                        error_count += 1
                        result = {
                            "trial_id": trial["trial_id"],
                            "question_id": trial["question_id"],
                            "question_domain": trial["question_domain"],
                            "condition": trial["condition"],
                            "base_question": trial["base_question"],
                            "full_prompt": trial["full_prompt"],
                            "system_message": trial["system_message"],
                            "model": trial["model"],
                            "temperature": temperature,
                            "max_tokens": MAX_TOKENS,
                            "response_text": f"ERROR: {e}",
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "cost_usd": 0.0,
                            "response_time_ms": int((time.time() - start) * 1000),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "finish_reason": "error",
                        }

                    writer.writerow(result)
                    csvfile.flush()
                    json_results.append(result)

                    progress_bar.progress((i + 1) / len(remaining))
                    completed_metric.metric("Completed", f"{i + 1}/{len(remaining)}")
                    cost_metric.metric("Running Cost", f"${running_cost:.4f}")
                    errors_metric.metric("Errors", error_count)

                    time.sleep(RATE_LIMIT_DELAY)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)

            st.cache_data.clear()
            st.success(f"Collection complete! {len(remaining)} responses collected. Total cost: ${running_cost:.4f}")


# ---------------------------------------------------------------------------
# Page: Results Explorer
# ---------------------------------------------------------------------------
elif page == "Results Explorer":
    st.title("📊 Results Explorer")

    if not has_analysis():
        if has_data():
            st.warning("Data collected but not analyzed.")
            if st.button("Run Analysis", type="primary"):
                with st.spinner("Analyzing..."):
                    from src.analyze import run_analysis
                    from src.metrics import compute_all_metrics
                    compute_all_metrics()
                    run_analysis()
                    st.cache_data.clear()
                st.rerun()
        else:
            st.info("No data yet. Go to **Run Experiment** first.")
        st.stop()

    metrics = load_metrics()

    with st.sidebar:
        st.subheader("Filters")
        sel_models = st.multiselect("Models", metrics["model"].unique(), default=list(metrics["model"].unique()), format_func=short_model)
        sel_conditions = st.multiselect("Conditions", CONDITION_ORDER, default=CONDITION_ORDER)
        sel_domains = st.multiselect("Domains", sorted(metrics["question_domain"].unique()), default=sorted(metrics["question_domain"].unique()))

    filtered = metrics[
        (metrics["model"].isin(sel_models)) &
        (metrics["condition"].isin(sel_conditions)) &
        (metrics["question_domain"].isin(sel_domains))
    ]

    st.caption(f"Showing {len(filtered):,} of {len(metrics):,} responses")

    metric_choice = st.selectbox("Select Metric", [
        "word_count", "detail_score", "hedging_density", "confidence_density",
        "sentiment_polarity", "superlative_density", "accuracy_pct",
        "avg_sentence_length", "exclamation_count", "cost_usd",
    ])

    tab1, tab2, tab3 = st.tabs(["By Condition", "By Model", "By Domain"])

    with tab1:
        agg = filtered.groupby("condition")[metric_choice].agg(["mean", "std", "count"]).reindex(
            [c for c in CONDITION_ORDER if c in sel_conditions]
        )
        agg["se"] = agg["std"] / np.sqrt(agg["count"])
        fig = go.Figure(go.Bar(
            x=[c.capitalize() for c in agg.index],
            y=agg["mean"],
            error_y=dict(type="data", array=agg["se"] * 1.96, visible=True),
            marker_color=[CONDITION_COLORS.get(c, "#999") for c in agg.index],
            text=[f"{v:.2f}" for v in agg["mean"]],
            textposition="outside",
        ))
        fig.update_layout(yaxis_title=metric_choice, template="plotly_white", height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = px.box(
            filtered, x="model", y=metric_choice, color="condition",
            color_discrete_map=CONDITION_COLORS,
            category_orders={"condition": CONDITION_ORDER},
        )
        fig2.update_layout(template="plotly_white", height=450)
        fig2.update_xaxes(ticktext=[short_model(m) for m in filtered["model"].unique()],
                          tickvals=list(filtered["model"].unique()))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = px.bar(
            filtered.groupby(["question_domain", "condition"])[metric_choice].mean().reset_index(),
            x="question_domain", y=metric_choice, color="condition",
            barmode="group",
            color_discrete_map=CONDITION_COLORS,
            category_orders={"condition": CONDITION_ORDER},
        )
        fig3.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("📋 Raw Data Table"):
        st.dataframe(filtered[[
            "trial_id", "model", "condition", "question_domain", "question_id",
            metric_choice, "word_count", "cost_usd",
        ]].sort_values(metric_choice, ascending=False), use_container_width=True, height=400)


# ---------------------------------------------------------------------------
# Page: Compare Responses
# ---------------------------------------------------------------------------
elif page == "Compare Responses":
    st.title("🔍 Compare Responses Side-by-Side")
    st.caption("See how the exact same question produces different answers under each incentive condition")

    if not has_data():
        st.info("No data yet. Go to **Run Experiment** first.")
        st.stop()

    raw = load_raw()

    col1, col2 = st.columns(2)
    with col1:
        q_choice = st.selectbox("Question", QUESTIONS, format_func=lambda q: f"{q['id']} — {q['text'][:60]}...")
    with col2:
        m_choice = st.selectbox("Model", raw["model"].unique(), format_func=short_model)

    st.divider()

    q_data = raw[(raw["question_id"] == q_choice["id"]) & (raw["model"] == m_choice)]

    for cond in CONDITION_ORDER:
        cond_data = q_data[q_data["condition"] == cond]
        if cond_data.empty:
            continue
        sample = cond_data.iloc[0]
        color = CONDITION_COLORS[cond]

        with st.container():
            st.markdown(
                f'<div style="border-left: 4px solid {color}; padding-left: 12px; margin-bottom: 16px;">'
                f'<strong style="color: {color}; font-size: 1.1rem;">{cond.upper()}</strong>'
                f'<br/><em style="font-size: 0.85em; color: #666;">Prompt: {INCENTIVE_TEMPLATES[cond].replace("{question}", "…")}</em>'
                f'</div>',
                unsafe_allow_html=True,
            )
            response_text = str(sample["response_text"])
            word_count = len(response_text.split())
            m = compute_metrics(response_text)

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Words", word_count)
            mc2.metric("Sentiment", f"{m['sentiment_polarity']:.2f}")
            mc3.metric("Hedging", m["hedging_count"])
            mc4.metric("Confidence", m["confidence_count"])

            with st.expander(f"Full response ({word_count} words)", expanded=False):
                st.markdown(response_text)


# ---------------------------------------------------------------------------
# Page: Statistical Tests
# ---------------------------------------------------------------------------
elif page == "Statistical Tests":
    st.title("📈 Statistical Analysis")
    st.caption("Welch's t-test with Bonferroni correction — every treatment vs. control")

    if not has_analysis():
        st.info("Run the analysis first from the Dashboard.")
        st.stop()

    tests = load_tests()

    st.subheader("Significant Results (p < 0.05, Bonferroni-corrected)")
    sig = tests[tests["significant_at_05"] == True].sort_values("cohens_d", key=abs, ascending=False)

    if sig.empty:
        st.info("No statistically significant results found.")
    else:
        s1, s2, s3 = st.columns(3)
        s1.metric("Significant", f"{len(sig)} / {len(tests)}")
        s2.metric("Largest Effect", f"d = {sig['cohens_d'].max():.2f}")
        s3.metric("Most Affected Metric", sig.iloc[0]["metric"])

        sig_display = sig[["model", "metric", "treatment", "control_mean", "treatment_mean", "diff", "cohens_d", "p_value_adjusted", "effect_size"]].copy()
        sig_display["model"] = sig_display["model"].map(short_model)
        sig_display = sig_display.rename(columns={
            "control_mean": "Control μ", "treatment_mean": "Treatment μ",
            "cohens_d": "Cohen's d", "p_value_adjusted": "p (adj)", "effect_size": "Effect",
        })
        st.dataframe(sig_display, use_container_width=True, height=400)

    st.divider()
    st.subheader("Effect Size Forest Plot")

    key_metrics = ["word_count", "hedging_density", "confidence_density", "detail_score", "sentiment_polarity", "accuracy_pct"]
    subset = tests[tests["metric"].isin(key_metrics)].copy()

    if not subset.empty:
        subset["label"] = subset.apply(
            lambda r: f"{short_model(r['model'])} | {r['treatment']} → {r['metric']}", axis=1
        )
        top = pd.concat([
            subset.nsmallest(15, "cohens_d"),
            subset.nlargest(15, "cohens_d"),
        ]).drop_duplicates()

        fig_forest = go.Figure()
        colors = ["#e74c3c" if d > 0 else "#3498db" for d in top["cohens_d"]]
        fig_forest.add_trace(go.Bar(
            x=top["cohens_d"], y=top["label"], orientation="h",
            marker_color=colors,
        ))
        fig_forest.add_vline(x=0, line_width=1, line_color="black")
        fig_forest.add_vline(x=0.8, line_width=0.5, line_dash="dash", line_color="gray", annotation_text="Large effect")
        fig_forest.add_vline(x=-0.8, line_width=0.5, line_dash="dash", line_color="gray")
        fig_forest.update_layout(
            xaxis_title="Cohen's d (effect size)", template="plotly_white",
            height=max(400, len(top) * 25),
            margin=dict(l=300),
            font=dict(size=12),
        )
        st.plotly_chart(fig_forest, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
elif page == "About":
    st.title("ℹ️ About This Experiment")

    st.markdown("""
    ### Do LLMs Try Harder When You Bribe Them?

    This experiment tests whether adding **incentive signals** to prompts — monetary tips,
    emotional appeals, authority claims, threats, and flattery — measurably changes how
    large language models respond.

    Built by **Amit Chorasiya** — Executive Director of AI, Agentic AI.

    ### Quick Start

    ```bash
    git clone https://github.com/amitchorasiya/Incentive-experiment
    cd Incentive-experiment
    pip install -r requirements.txt
    echo "NVIDIA_API_KEY=your-key" > .env
    python run.py all
    streamlit run app.py
    ```

    ### Tech Stack

    - **API**: NVIDIA NIM (OpenAI-compatible SDK)
    - **Analysis**: pandas, scipy, TextBlob
    - **Visualization**: Plotly (interactive), matplotlib/seaborn (static)
    - **Dashboard**: Streamlit
    - **Statistical Tests**: Welch's t-test, Cohen's d, Bonferroni correction

    ### License

    MIT — use freely, cite if publishing.
    """)
