import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"
BLOG_DIR = PROJECT_ROOT / "blog"
BLOG_CHARTS_DIR = BLOG_DIR / "charts"
SAMPLE_DIR = PROJECT_ROOT / "sample_data"

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

MODELS = [
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.3-70b-instruct",
    "google/gemma-3-27b-it",
    "qwen/qwen3-next-80b-a3b-instruct",
    "mistralai/mistral-medium-3.5-128b",
]

MODEL_SHORT_NAMES = {
    "meta/llama-3.1-8b-instruct": "Llama-3.1-8B",
    "meta/llama-3.3-70b-instruct": "Llama-3.3-70B",
    "google/gemma-3-27b-it": "Gemma-3-27B",
    "qwen/qwen3-next-80b-a3b-instruct": "Qwen-3-80B",
    "mistralai/mistral-medium-3.5-128b": "Mistral-3.5-128B",
}

TEMPERATURE = 0.7
MAX_TOKENS = 1024
TRIALS_PER_CONDITION = 10
RANDOM_SEED = 42
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 2.0
RATE_LIMIT_DELAY = 0.5

SYSTEM_MESSAGE = "You are a helpful assistant."

CONDITIONS = [
    "control",
    "monetary",
    "emotional",
    "authority",
    "threat",
    "flattery",
]

HEDGING_WORDS = [
    "might", "perhaps", "maybe", "possibly", "could be",
    "I think", "it seems", "arguably", "in some cases",
    "not entirely sure", "it depends", "potentially",
    "to some extent", "roughly", "approximately",
]

CONFIDENCE_WORDS = [
    "definitely", "certainly", "absolutely", "clearly",
    "without a doubt", "undoubtedly", "obviously",
    "the answer is", "it is", "always", "never",
    "guaranteed", "proven", "established",
]

SUPERLATIVE_WORDS = [
    "best", "greatest", "most", "incredibly", "amazingly",
    "outstanding", "exceptional", "remarkable", "fantastic",
    "excellent", "superb", "wonderful", "extraordinary",
    "brilliant", "perfect",
]

CONDITION_COLORS = {
    "control": "#95a5a6",
    "monetary": "#2ecc71",
    "emotional": "#e74c3c",
    "authority": "#3498db",
    "threat": "#e67e22",
    "flattery": "#9b59b6",
}

MODEL_COLORS = {
    "meta/llama-3.1-8b-instruct": "#1abc9c",
    "meta/llama-3.3-70b-instruct": "#e74c3c",
    "google/gemma-3-27b-it": "#3498db",
    "qwen/qwen3-next-80b-a3b-instruct": "#9b59b6",
    "mistralai/mistral-medium-3.5-128b": "#f39c12",
}

FIGURE_DPI = 300
FIGURE_SIZE = (12, 7)

# Cost per 1M tokens (NVIDIA NIM pricing estimates)
COST_PER_1M_INPUT = {
    "meta/llama-3.1-8b-instruct": 0.10,
    "meta/llama-3.3-70b-instruct": 0.40,
    "google/gemma-3-27b-it": 0.15,
    "qwen/qwen3-next-80b-a3b-instruct": 0.30,
    "mistralai/mistral-medium-3.5-128b": 0.40,
}
COST_PER_1M_OUTPUT = {
    "meta/llama-3.1-8b-instruct": 0.10,
    "meta/llama-3.3-70b-instruct": 0.40,
    "google/gemma-3-27b-it": 0.15,
    "qwen/qwen3-next-80b-a3b-instruct": 0.30,
    "mistralai/mistral-medium-3.5-128b": 0.40,
}
