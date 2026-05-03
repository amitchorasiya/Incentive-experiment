import numpy as np
import pandas as pd

from src.analyze import cohens_d, compute_summary_stats


def test_cohens_d_zero():
    g1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    g2 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    d = cohens_d(g1, g2)
    assert abs(d) < 0.01


def test_cohens_d_large():
    g1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    g2 = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    d = cohens_d(g1, g2)
    assert d > 3.0


def test_cohens_d_direction():
    g1 = pd.Series([1.0, 2.0, 3.0])
    g2 = pd.Series([4.0, 5.0, 6.0])
    d = cohens_d(g1, g2)
    assert d > 0


def test_cohens_d_negative():
    g1 = pd.Series([4.0, 5.0, 6.0])
    g2 = pd.Series([1.0, 2.0, 3.0])
    d = cohens_d(g1, g2)
    assert d < 0


def test_summary_stats_structure():
    df = pd.DataFrame({
        "model": ["m1"] * 6,
        "condition": ["control"] * 3 + ["monetary"] * 3,
        "word_count": [100, 110, 105, 120, 130, 125],
        "char_count": [500, 550, 525, 600, 650, 625],
        "sentence_count": [5, 6, 5, 6, 7, 6],
        "paragraph_count": [2, 2, 3, 3, 3, 2],
        "bullet_point_count": [0, 1, 0, 2, 3, 1],
        "code_block_count": [0, 0, 0, 1, 0, 1],
        "example_count": [1, 1, 2, 2, 3, 2],
        "detail_score": [1, 2, 2, 5, 6, 4],
        "sentiment_polarity": [0.1, 0.2, 0.15, 0.3, 0.25, 0.35],
        "sentiment_subjectivity": [0.5, 0.4, 0.45, 0.6, 0.55, 0.65],
        "hedging_count": [2, 3, 2, 1, 1, 0],
        "hedging_density": [0.02, 0.03, 0.02, 0.008, 0.008, 0.0],
        "confidence_count": [1, 1, 2, 3, 2, 3],
        "confidence_density": [0.01, 0.009, 0.019, 0.025, 0.015, 0.024],
        "superlative_count": [0, 1, 0, 2, 1, 2],
        "superlative_density": [0.0, 0.009, 0.0, 0.017, 0.008, 0.016],
        "exclamation_count": [0, 0, 1, 1, 2, 1],
        "question_mark_count": [1, 0, 0, 0, 1, 0],
        "avg_sentence_length": [20.0, 18.3, 21.0, 20.0, 18.6, 20.8],
    })

    result = compute_summary_stats(df)
    assert not result.empty
    assert "mean" in result.columns
    assert "ci_lower" in result.columns
    assert "ci_upper" in result.columns

    wc_control = result[(result["condition"] == "control") & (result["metric"] == "word_count")]
    assert len(wc_control) == 1
    assert abs(wc_control.iloc[0]["mean"] - 105.0) < 0.1
