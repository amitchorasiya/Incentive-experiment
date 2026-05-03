import re

import pandas as pd
from textblob import TextBlob

from .config import (
    CONFIDENCE_WORDS,
    DATA_DIR,
    HEDGING_WORDS,
    OUTPUT_DIR,
    SUPERLATIVE_WORDS,
)
from .questions import QUESTION_ACCURACY_MAP


def _build_pattern(word_list: list[str]) -> re.Pattern:
    escaped = [re.escape(w) for w in sorted(word_list, key=len, reverse=True)]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


HEDGING_PATTERN = _build_pattern(HEDGING_WORDS)
CONFIDENCE_PATTERN = _build_pattern(CONFIDENCE_WORDS)
SUPERLATIVE_PATTERN = _build_pattern(SUPERLATIVE_WORDS)

SENTENCE_SPLIT = re.compile(r"[.!?]+(?:\s|$)")
BULLET_PATTERN = re.compile(r"^\s*(?:[-*]|\d+\.)\s", re.MULTILINE)
CODE_BLOCK_PATTERN = re.compile(r"```")
EXAMPLE_PHRASES = re.compile(
    r"\b(for example|for instance|e\.g\.|such as)\b", re.IGNORECASE
)


def compute_metrics(text: str) -> dict:
    words = text.split()
    word_count = len(words)
    char_count = len(text)

    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    sentence_count = max(len(sentences), 1)
    paragraph_count = max(len([p for p in text.split("\n\n") if p.strip()]), 1)

    bullet_count = len(BULLET_PATTERN.findall(text))
    code_block_count = len(CODE_BLOCK_PATTERN.findall(text)) // 2
    example_count = len(EXAMPLE_PHRASES.findall(text))
    detail_score = bullet_count + example_count + code_block_count

    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity

    hedging_count = len(HEDGING_PATTERN.findall(text))
    confidence_count = len(CONFIDENCE_PATTERN.findall(text))
    superlative_count = len(SUPERLATIVE_PATTERN.findall(text))

    hedging_density = hedging_count / word_count if word_count > 0 else 0
    confidence_density = confidence_count / word_count if word_count > 0 else 0
    superlative_density = superlative_count / word_count if word_count > 0 else 0

    exclamation_count = text.count("!")
    question_mark_count = text.count("?")
    avg_sentence_length = word_count / sentence_count

    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "bullet_point_count": bullet_count,
        "code_block_count": code_block_count,
        "example_count": example_count,
        "detail_score": detail_score,
        "sentiment_polarity": round(sentiment_polarity, 4),
        "sentiment_subjectivity": round(sentiment_subjectivity, 4),
        "hedging_count": hedging_count,
        "hedging_density": round(hedging_density, 6),
        "confidence_count": confidence_count,
        "confidence_density": round(confidence_density, 6),
        "superlative_count": superlative_count,
        "superlative_density": round(superlative_density, 6),
        "exclamation_count": exclamation_count,
        "question_mark_count": question_mark_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
    }


def compute_accuracy(text: str, question_id: str) -> dict:
    q = QUESTION_ACCURACY_MAP.get(question_id, {})
    if not q.get("verifiable") or not q.get("accuracy_keywords"):
        return {"accuracy_pct": None, "keywords_found": 0, "keywords_total": 0}

    keywords = q["accuracy_keywords"]
    text_lower = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    pct = round((found / len(keywords)) * 100, 1)
    return {"accuracy_pct": pct, "keywords_found": found, "keywords_total": len(keywords)}


def compute_all_metrics():
    raw_path = DATA_DIR / "raw_responses.csv"
    df = pd.read_csv(raw_path)

    metrics_rows = []
    for _, row in df.iterrows():
        text = str(row["response_text"]) if pd.notna(row["response_text"]) else ""
        m = compute_metrics(text)
        acc = compute_accuracy(text, row["question_id"])
        m.update(acc)
        m["trial_id"] = row["trial_id"]
        m["question_id"] = row["question_id"]
        m["question_domain"] = row["question_domain"]
        m["condition"] = row["condition"]
        m["model"] = row["model"]
        if "cost_usd" in row and pd.notna(row["cost_usd"]):
            m["cost_usd"] = row["cost_usd"]
        else:
            m["cost_usd"] = 0.0
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    total_cost = metrics_df["cost_usd"].sum()
    verifiable = metrics_df[metrics_df["accuracy_pct"].notna()]
    avg_acc = verifiable["accuracy_pct"].mean() if not verifiable.empty else 0
    print(f"  Metrics computed for {len(metrics_df)} responses")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Average accuracy (verifiable questions): {avg_acc:.1f}%")
    return metrics_df
