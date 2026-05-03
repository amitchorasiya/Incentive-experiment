import random

from .config import CONDITIONS, MODELS, SYSTEM_MESSAGE
from .questions import INCENTIVE_TEMPLATES, QUESTIONS


def generate_trial_plan(seed: int, trials: int, models: list[str] | None = None) -> list[dict]:
    if models is None:
        models = MODELS

    plan = []
    for model in models:
        for question in QUESTIONS:
            for condition in CONDITIONS:
                template = INCENTIVE_TEMPLATES[condition]
                full_prompt = template.format(question=question["text"])
                for t in range(trials):
                    plan.append({
                        "trial_id": f"{question['id']}_{condition}_{model.split('/')[-1]}_t{t:02d}",
                        "question_id": question["id"],
                        "question_domain": question["domain"],
                        "base_question": question["text"],
                        "condition": condition,
                        "full_prompt": full_prompt,
                        "system_message": SYSTEM_MESSAGE,
                        "model": model,
                        "trial_num": t,
                    })

    rng = random.Random(seed)
    rng.shuffle(plan)
    return plan
