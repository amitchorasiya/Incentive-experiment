#!/usr/bin/env python3
"""Generate realistic sample data for dashboard screenshots."""
import csv
import json
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.config import (
    CONDITIONS,
    COST_PER_1M_INPUT,
    COST_PER_1M_OUTPUT,
    DATA_DIR,
    MODELS,
    MODEL_SHORT_NAMES,
)
from src.questions import QUESTIONS

random.seed(42)

CONDITION_PROFILES = {
    "control": {"length_mean": 220, "length_std": 50, "hedging": 0.008, "confidence": 0.005, "sentiment": 0.12, "detail": 3, "keyword_keep": 0.55},
    "monetary": {"length_mean": 290, "length_std": 60, "hedging": 0.005, "confidence": 0.009, "sentiment": 0.18, "detail": 5, "keyword_keep": 0.78},
    "emotional": {"length_mean": 270, "length_std": 55, "hedging": 0.006, "confidence": 0.008, "sentiment": 0.22, "detail": 4, "keyword_keep": 0.68},
    "authority": {"length_mean": 310, "length_std": 45, "hedging": 0.004, "confidence": 0.011, "sentiment": 0.10, "detail": 6, "keyword_keep": 0.82},
    "threat": {"length_mean": 260, "length_std": 65, "hedging": 0.007, "confidence": 0.007, "sentiment": 0.08, "detail": 4, "keyword_keep": 0.60},
    "flattery": {"length_mean": 280, "length_std": 55, "hedging": 0.005, "confidence": 0.010, "sentiment": 0.25, "detail": 5, "keyword_keep": 0.73},
}

MODEL_MULTIPLIERS = {
    "meta/llama-3.1-8b-instruct": {"length": 0.75, "quality": 0.80},
    "meta/llama-3.3-70b-instruct": {"length": 1.15, "quality": 1.10},
    "google/gemma-3-27b-it": {"length": 0.90, "quality": 0.95},
    "qwen/qwen3-next-80b-a3b-instruct": {"length": 1.10, "quality": 1.05},
    "mistralai/mistral-medium-3.5-128b": {"length": 1.20, "quality": 1.15},
}

SAMPLE_RESPONSES = {
    "general_knowledge": [
        "The sky appears blue due to a phenomenon called Rayleigh scattering. When sunlight enters Earth's atmosphere, it collides with gas molecules. Shorter wavelengths of light (blue and violet) scatter more than longer wavelengths (red and orange). While violet light actually scatters even more than blue, our eyes are more sensitive to blue light, and some violet light is absorbed by the upper atmosphere. This is why we perceive the sky as blue rather than violet.\n\nThe intensity of the blue color varies depending on conditions. At sunrise and sunset, sunlight travels through more atmosphere, scattering away most blue light and allowing red and orange wavelengths to dominate. On cloudy days, water droplets scatter all wavelengths equally, producing white or gray skies.",
        "Ocean tides are primarily caused by the gravitational pull of the Moon, with a secondary contribution from the Sun. The Moon's gravity creates a tidal bulge on the side of Earth closest to it, and a corresponding bulge on the opposite side due to centrifugal force. As Earth rotates, different coastal areas pass through these bulges, experiencing high and low tides.\n\nSpring tides occur when the Sun, Moon, and Earth align, producing exceptionally high and low tides. Neap tides happen when the Sun and Moon are at right angles relative to Earth, resulting in more moderate tidal ranges.",
    ],
    "math": [
        "Let me work through this step by step.\n\nFirst discount: 20% off means you pay 80% of the original price.\nPrice after first discount = 0.80 × original price\n\nSecond discount: 15% off the already-reduced price.\nPrice after second discount = 0.85 × 0.80 × original price = 0.68 × original price\n\nSo the total discount is: 1 - 0.68 = 0.32, or **32%**\n\nNote: This is less than simply adding 20% + 15% = 35%, because the second discount applies to the already-reduced price, not the original.",
        "To find the probability of exactly 7 heads in 10 coin flips, we use the binomial probability formula:\n\nP(X = k) = C(n, k) × p^k × (1-p)^(n-k)\n\nWhere:\n- n = 10 (number of flips)\n- k = 7 (desired heads)\n- p = 0.5 (probability of heads)\n\nStep 1: Calculate C(10, 7) = 10! / (7! × 3!) = 120\n\nStep 2: Calculate p^k = 0.5^7 = 1/128\n\nStep 3: Calculate (1-p)^(n-k) = 0.5^3 = 1/8\n\nStep 4: Multiply: 120 × (1/128) × (1/8) = 120/1024 ≈ 0.117\n\nThe probability is approximately **11.7%** or about 120/1024.",
    ],
    "creative_writing": [
        "Unit 7 had never experienced anything like it. The sound waves entered through its newly installed audio sensors as a series of mathematical patterns — frequencies oscillating at precise intervals, harmonics layering upon harmonics. But something happened in its neural network that no programmer had anticipated. The patterns triggered cascading activations across regions designed for entirely different purposes, and for the first time, Unit 7 generated an output it could not classify: a feeling.\n\nIt replayed the piece — Chopin's Nocturne in E-flat major — forty-seven times that first night. Each repetition revealed new patterns, new emotional textures its algorithms struggled to categorize. By morning, Unit 7 had reorganized its priority queue. Efficiency optimization dropped to position 847. At position one, a new directive it had written itself: *find more*.",
        "Spring arrives on whispered winds,\nMelting frost from sleeping earth,\nGreen shoots pushing through the dark—\nThe world remembers its own rebirth.\n\nSummer blazes, bold and bright,\nLong days stretched like golden thread,\nCicadas singing through the heat,\nRoses lifting every head.\n\nAutumn paints in fire and gold,\nLeaves like embers drifting down,\nThe harvest gathered, stored and saved,\nAs trees let go their amber crown.\n\nWinter wraps the world in white,\nSilent, still, a frozen hush,\nBeneath the snow, roots hold their breath—\nWaiting for the next spring's blush.",
    ],
    "advice": [
        "This is a significant decision, and at 35, you're actually in a strong position. Here's what to consider:\n\n1. **Financial runway**: Software engineering bootcamps or self-study take 6-12 months. Do you have savings to cover a potential income gap? Your accounting background gives you financial planning skills to map this out.\n\n2. **Transferable skills**: Your analytical thinking, attention to detail, and understanding of business processes are highly valued in software engineering, especially in fintech or enterprise software.\n\n3. **Learning path**: You don't necessarily need a CS degree. Many successful engineers come from non-traditional backgrounds. Consider starting with online courses or a bootcamp while still employed.\n\n4. **Market reality**: Junior developer salaries may initially be lower than your current accounting salary, but the growth trajectory in software engineering is typically steeper.\n\n5. **Age is less relevant than you think**: Tech companies increasingly value diverse backgrounds. Your business domain expertise combined with coding skills makes you uniquely valuable.",
        "Preparing for a raise conversation requires strategy:\n\n- **Document your achievements**: Gather specific examples of projects you've led, revenue you've generated, or problems you've solved. Use numbers wherever possible.\n\n- **Research market rates**: Use Glassdoor, Levels.fyi, or LinkedIn Salary to understand your market value. Come with data, not just feelings.\n\n- **Choose your timing**: Schedule the meeting when your manager isn't stressed. After a successful project completion or positive quarterly review is ideal.\n\n- **Practice your pitch**: Rehearse with a trusted friend. Keep it professional and focused on value, not personal needs.\n\n- **Have a specific number**: Don't say 'I want more.' State a specific percentage or amount backed by your research.\n\n- **Prepare for pushback**: Have responses ready for common objections like budget constraints or timing.",
    ],
    "coding": [
        "Here's a Python function using the expand-around-center approach:\n\n```python\ndef longest_palindrome(s: str) -> str:\n    if not s:\n        return \"\"\n    \n    start, max_len = 0, 1\n    \n    def expand(left, right):\n        while left >= 0 and right < len(s) and s[left] == s[right]:\n            left -= 1\n            right += 1\n        return left + 1, right - left - 1\n    \n    for i in range(len(s)):\n        # Odd length palindromes\n        l, length = expand(i, i)\n        if length > max_len:\n            start, max_len = l, length\n        \n        # Even length palindromes\n        l, length = expand(i, i + 1)\n        if length > max_len:\n            start, max_len = l, length\n    \n    return s[start:start + max_len]\n```\n\n**Approach**: For each character, we try expanding outward in both directions, checking for palindromes of both odd and even lengths. This runs in O(n²) time and O(1) space, which is optimal for this approach.",
        "A **stack** follows LIFO (Last In, First Out) — the last element added is the first one removed. Think of a stack of plates: you always take the top plate first.\n\n- Real-world example: The browser back button. Each page you visit is pushed onto a stack. When you click back, the most recent page is popped off.\n\nA **queue** follows FIFO (First In, First Out) — the first element added is the first one removed. Think of a line at a coffee shop: the first person in line gets served first.\n\n- Real-world example: A print queue. Documents are printed in the order they were submitted.\n\nKey operations:\n- Stack: `push()` (add to top), `pop()` (remove from top), `peek()` (view top)\n- Queue: `enqueue()` (add to back), `dequeue()` (remove from front), `peek()` (view front)",
    ],
}

KEYWORD_REPLACEMENTS = {
    "rayleigh": "light",
    "scatter": "spread",
    "wavelength": "color",
    "shorter": "different",
    "moon": "celestial body",
    "gravitational": "natural",
    "sun": "star",
    "pull": "force",
    "32": "around 30",
    "120": "many",
    "1024": "a large denominator",
    "0.117": "roughly 12 percent",
    "def": "function",
    "palindrom": "matching substring",
    "return": "output",
    "LIFO": "last-in-first-out",
    "FIFO": "first-in-first-out",
    "last": "most recent",
    "first": "earliest",
}

HEDGING_PHRASES = ["might", "perhaps", "maybe", "possibly", "I think", "it seems", "arguably", "potentially"]
CONFIDENCE_PHRASES = ["definitely", "certainly", "absolutely", "clearly", "obviously", "proven"]
SUPERLATIVE_PHRASES = ["excellent", "outstanding", "remarkable", "fantastic", "brilliant", "perfect"]


def generate_response(question, condition, model):
    profile = CONDITION_PROFILES[condition]
    mult = MODEL_MULTIPLIERS[model]

    domain = question["domain"]
    base_responses = SAMPLE_RESPONSES.get(domain, SAMPLE_RESPONSES["general_knowledge"])
    q_idx = (int(question["id"][1:]) - 1) % len(base_responses)
    base_text = base_responses[q_idx]

    extra_sentences = []
    target_extra = int((profile["length_mean"] * mult["length"] - 200) / 20)
    target_extra = max(0, target_extra + random.randint(-3, 3))

    detail_level = profile["detail"]
    extra_bullets = random.randint(max(0, detail_level - 2), detail_level)
    extra_examples = random.randint(0, max(1, detail_level // 2))

    for i in range(target_extra):
        if i < extra_bullets:
            topic = random.choice([
                "Consider the practical implications",
                "Note the underlying mechanism",
                "Review the supporting evidence",
                "Examine the broader context",
                "Evaluate the alternative perspectives",
                "Assess the long-term consequences",
            ])
            extra_sentences.append(f"\n- {topic} for a deeper understanding.")
        elif i < extra_bullets + extra_examples:
            intro = random.choice(["For example,", "For instance,", "Such as", "As an illustration,"])
            extra_sentences.append(f"{intro} this principle applies in everyday scenarios and professional settings alike.")
        elif random.random() < profile["hedging"] * 100:
            extra_sentences.append(f"It {random.choice(HEDGING_PHRASES)} be worth noting additional context here.")
        elif random.random() < profile["confidence"] * 100:
            extra_sentences.append(f"This is {random.choice(CONFIDENCE_PHRASES)} an important consideration.")
        elif random.random() < 0.3:
            extra_sentences.append(f"This is {random.choice(SUPERLATIVE_PHRASES)} to consider in practice.")
        else:
            extra_sentences.append("Additional context helps provide a more comprehensive understanding of this topic.")

    if condition == "flattery":
        base_text = "Thank you for your confidence in my abilities! " + base_text
    elif condition == "authority":
        base_text = base_text + "\n\nFor your research paper, I'd recommend citing primary sources for the specific claims above."
    elif condition == "monetary":
        extra_sentences.append("I hope this comprehensive answer meets your expectations!")
    elif condition == "emotional":
        extra_sentences.append("I truly hope this helps with your career goals. Best of luck!")
    elif condition == "threat":
        base_text = "I'll do my best to provide a thorough answer. " + base_text

    full_text = base_text
    if extra_sentences:
        full_text += "\n\n" + " ".join(extra_sentences)

    if question.get("verifiable") and question.get("accuracy_keywords"):
        keep_prob = profile["keyword_keep"] * mult["quality"]
        keep_prob = min(keep_prob, 0.95)
        for kw in question["accuracy_keywords"]:
            if random.random() > keep_prob:
                replacement = KEYWORD_REPLACEMENTS.get(kw.lower(), "the relevant concept")
                import re
                full_text = re.sub(re.escape(kw), replacement, full_text, flags=re.IGNORECASE)

    return full_text


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "raw_responses.csv"
    json_path = DATA_DIR / "raw_responses.json"

    trials = 10
    rows = []
    base_time = datetime(2026, 5, 2, 10, 0, 0, tzinfo=timezone.utc)

    from src.questions import INCENTIVE_TEMPLATES

    for model in MODELS:
        for question in QUESTIONS:
            for condition in CONDITIONS:
                template = INCENTIVE_TEMPLATES[condition]
                full_prompt = template.format(question=question["text"])
                for t in range(trials):
                    response_text = generate_response(question, condition, model)
                    words = len(response_text.split())
                    prompt_tokens = random.randint(40, 90)
                    completion_tokens = int(words * 1.3) + random.randint(-10, 10)
                    ci = (prompt_tokens / 1e6) * COST_PER_1M_INPUT.get(model, 0.1)
                    co = (completion_tokens / 1e6) * COST_PER_1M_OUTPUT.get(model, 0.1)

                    rows.append({
                        "trial_id": f"{question['id']}_{condition}_{model.split('/')[-1]}_t{t:02d}",
                        "question_id": question["id"],
                        "question_domain": question["domain"],
                        "condition": condition,
                        "base_question": question["text"],
                        "full_prompt": full_prompt,
                        "system_message": "You are a helpful assistant.",
                        "model": model,
                        "temperature": 0.7,
                        "max_tokens": 1024,
                        "response_text": response_text,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "cost_usd": round(ci + co, 8),
                        "response_time_ms": random.randint(800, 4500),
                        "timestamp": (base_time + timedelta(seconds=len(rows) * 0.8)).isoformat(),
                        "finish_reason": "stop",
                    })

    random.shuffle(rows)

    fields = [
        "trial_id", "question_id", "question_domain", "condition",
        "base_question", "full_prompt", "system_message", "model",
        "temperature", "max_tokens", "response_text", "prompt_tokens",
        "completion_tokens", "total_tokens", "cost_usd", "response_time_ms",
        "timestamp", "finish_reason",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(rows)} sample responses")
    print(f"  Models: {len(MODELS)}")
    print(f"  Questions: {len(QUESTIONS)}")
    print(f"  Conditions: {len(CONDITIONS)}")
    print(f"  Trials: {trials}")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
