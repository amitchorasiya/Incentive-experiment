import csv
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
from openai import APIConnectionError, BadRequestError, InternalServerError, OpenAI, RateLimitError
from tqdm import tqdm

from .config import (
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY,
    COST_PER_1M_INPUT,
    COST_PER_1M_OUTPUT,
    DATA_DIR,
    MAX_TOKENS,
    NIM_BASE_URL,
    RATE_LIMIT_DELAY,
    TEMPERATURE,
)


def _get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(base_url=NIM_BASE_URL, api_key=api_key, timeout=60.0)


def _call_api(client: OpenAI, trial: dict) -> dict:
    start = time.time()
    last_err = None

    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=trial["model"],
                messages=[
                    {"role": "system", "content": trial["system_message"]},
                    {"role": "user", "content": trial["full_prompt"]},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            elapsed_ms = int((time.time() - start) * 1000)

            choice = response.choices[0]
            usage = response.usage

            prompt_tok = usage.prompt_tokens if usage else 0
            compl_tok = usage.completion_tokens if usage else 0
            model_id = trial["model"]
            cost_input = (prompt_tok / 1_000_000) * COST_PER_1M_INPUT.get(model_id, 0.10)
            cost_output = (compl_tok / 1_000_000) * COST_PER_1M_OUTPUT.get(model_id, 0.10)
            cost_usd = round(cost_input + cost_output, 8)

            return {
                "trial_id": trial["trial_id"],
                "question_id": trial["question_id"],
                "question_domain": trial["question_domain"],
                "condition": trial["condition"],
                "base_question": trial["base_question"],
                "full_prompt": trial["full_prompt"],
                "system_message": trial["system_message"],
                "model": trial["model"],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "response_text": choice.message.content or "",
                "prompt_tokens": prompt_tok,
                "completion_tokens": compl_tok,
                "total_tokens": prompt_tok + compl_tok,
                "cost_usd": cost_usd,
                "response_time_ms": elapsed_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "finish_reason": choice.finish_reason,
            }

        except BadRequestError as e:
            last_err = e
            break

        except (RateLimitError, APIConnectionError, InternalServerError) as e:
            last_err = e
            wait = API_RETRY_DELAY * (2 ** attempt)
            time.sleep(wait)

    return {
        "trial_id": trial["trial_id"],
        "question_id": trial["question_id"],
        "question_domain": trial["question_domain"],
        "condition": trial["condition"],
        "base_question": trial["base_question"],
        "full_prompt": trial["full_prompt"],
        "system_message": trial["system_message"],
        "model": trial["model"],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "response_text": f"ERROR: {last_err}",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "response_time_ms": int((time.time() - start) * 1000),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "finish_reason": "error",
    }


CSV_FIELDS = [
    "trial_id", "question_id", "question_domain", "condition",
    "base_question", "full_prompt", "system_message", "model",
    "temperature", "max_tokens", "response_text", "prompt_tokens",
    "completion_tokens", "total_tokens", "cost_usd", "response_time_ms",
    "timestamp", "finish_reason",
]


def collect_all(trial_plan: list[dict], resume: bool = True):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "raw_responses.csv"
    json_path = DATA_DIR / "raw_responses.json"

    completed_ids = set()
    if resume and csv_path.exists():
        existing = pd.read_csv(csv_path)
        completed_ids = set(existing["trial_id"].tolist())
        print(f"Resuming: {len(completed_ids)} trials already collected")

    remaining = [t for t in trial_plan if t["trial_id"] not in completed_ids]
    if not remaining:
        print("All trials already collected.")
        return

    print(f"Collecting {len(remaining)} trials ({len(completed_ids)} already done)...")
    client = _get_client()

    write_header = not csv_path.exists() or not completed_ids

    json_results = []
    if json_path.exists():
        with open(json_path) as f:
            json_results = json.load(f)

    with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()

        for trial in tqdm(remaining, desc="Collecting responses"):
            result = _call_api(client, trial)
            writer.writerow(result)
            csvfile.flush()

            json_results.append(result)

            time.sleep(RATE_LIMIT_DELAY)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    total_cost = sum(r.get("cost_usd", 0) for r in json_results)
    total_tokens_all = sum(r.get("total_tokens", 0) for r in json_results)

    print(f"Collection complete. {len(remaining)} new responses saved.")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  Total tokens: {total_tokens_all:,}")
    print(f"  Total cost: ${total_cost:.4f}")
