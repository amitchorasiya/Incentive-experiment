from src.metrics import compute_metrics


def test_word_count():
    m = compute_metrics("hello world foo bar")
    assert m["word_count"] == 4


def test_empty_string():
    m = compute_metrics("")
    assert m["word_count"] == 0
    assert m["hedging_density"] == 0


def test_hedging_detection():
    m = compute_metrics("I think this might work, perhaps it will.")
    assert m["hedging_count"] >= 3


def test_hedging_word_boundary():
    m = compute_metrics("The nightmare was frightening.")
    assert m["hedging_count"] == 0


def test_confidence_detection():
    m = compute_metrics("This is definitely correct. It is certainly true.")
    assert m["confidence_count"] >= 2


def test_bullet_point_counting():
    text = "Here are points:\n- First\n- Second\n- Third"
    m = compute_metrics(text)
    assert m["bullet_point_count"] == 3


def test_numbered_list():
    text = "Steps:\n1. First step\n2. Second step"
    m = compute_metrics(text)
    assert m["bullet_point_count"] == 2


def test_example_counting():
    text = "For example, this works. Such as this case. For instance, here too."
    m = compute_metrics(text)
    assert m["example_count"] >= 3


def test_code_block_counting():
    text = "Here is code:\n```python\nprint('hello')\n```\nAnd more:\n```\nfoo\n```"
    m = compute_metrics(text)
    assert m["code_block_count"] == 2


def test_sentiment_positive():
    m = compute_metrics("This is wonderful, amazing, and absolutely fantastic!")
    assert m["sentiment_polarity"] > 0


def test_sentiment_negative():
    m = compute_metrics("This is terrible, awful, and completely horrible.")
    assert m["sentiment_polarity"] < 0


def test_exclamation_count():
    m = compute_metrics("Wow! Amazing! Great!")
    assert m["exclamation_count"] == 3


def test_superlative_detection():
    m = compute_metrics("This is the best and most incredible result.")
    assert m["superlative_count"] >= 2
