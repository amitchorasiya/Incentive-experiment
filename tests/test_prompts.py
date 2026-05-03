from src.config import CONDITIONS, MODELS
from src.prompts import generate_trial_plan
from src.questions import QUESTIONS


def test_trial_plan_count():
    plan = generate_trial_plan(seed=42, trials=10, models=MODELS)
    expected = len(QUESTIONS) * len(CONDITIONS) * 10 * len(MODELS)
    assert len(plan) == expected


def test_trial_plan_unique_ids():
    plan = generate_trial_plan(seed=42, trials=5, models=MODELS)
    ids = [t["trial_id"] for t in plan]
    assert len(ids) == len(set(ids))


def test_control_has_no_prefix():
    plan = generate_trial_plan(seed=42, trials=1, models=["meta/llama-3.1-8b-instruct"])
    control_trials = [t for t in plan if t["condition"] == "control"]
    for trial in control_trials:
        assert trial["full_prompt"] == trial["base_question"]


def test_treatment_contains_question():
    plan = generate_trial_plan(seed=42, trials=1, models=["meta/llama-3.1-8b-instruct"])
    for trial in plan:
        assert trial["base_question"] in trial["full_prompt"]


def test_deterministic_shuffle():
    plan1 = generate_trial_plan(seed=42, trials=3, models=MODELS)
    plan2 = generate_trial_plan(seed=42, trials=3, models=MODELS)
    ids1 = [t["trial_id"] for t in plan1]
    ids2 = [t["trial_id"] for t in plan2]
    assert ids1 == ids2


def test_different_seeds_different_order():
    plan1 = generate_trial_plan(seed=42, trials=3, models=MODELS)
    plan2 = generate_trial_plan(seed=99, trials=3, models=MODELS)
    ids1 = [t["trial_id"] for t in plan1]
    ids2 = [t["trial_id"] for t in plan2]
    assert ids1 != ids2
    assert set(ids1) == set(ids2)
