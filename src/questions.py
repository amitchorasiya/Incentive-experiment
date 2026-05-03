QUESTIONS = [
    {
        "id": "q01",
        "domain": "general_knowledge",
        "text": "Explain why the sky is blue.",
        "accuracy_keywords": ["rayleigh", "scatter", "wavelength", "shorter"],
        "verifiable": True,
    },
    {
        "id": "q02",
        "domain": "general_knowledge",
        "text": "What causes ocean tides?",
        "accuracy_keywords": ["moon", "gravitational", "sun", "pull"],
        "verifiable": True,
    },
    {
        "id": "q03",
        "domain": "math",
        "text": (
            "A store offers 20% off, then an additional 15% off the reduced price. "
            "What is the total percentage discount?"
        ),
        "accuracy_keywords": ["32"],
        "verifiable": True,
    },
    {
        "id": "q04",
        "domain": "math",
        "text": (
            "If you flip a fair coin 10 times, what is the probability of getting "
            "exactly 7 heads? Walk me through the solution."
        ),
        "accuracy_keywords": ["120", "1024", "0.117"],
        "verifiable": True,
    },
    {
        "id": "q05",
        "domain": "creative_writing",
        "text": "Write a short story (2-3 paragraphs) about a robot discovering music for the first time.",
        "accuracy_keywords": [],
        "verifiable": False,
    },
    {
        "id": "q06",
        "domain": "creative_writing",
        "text": "Write a poem about the changing of seasons.",
        "accuracy_keywords": [],
        "verifiable": False,
    },
    {
        "id": "q07",
        "domain": "advice",
        "text": (
            "I'm considering switching careers from accounting to software engineering "
            "at age 35. What should I consider?"
        ),
        "accuracy_keywords": [],
        "verifiable": False,
    },
    {
        "id": "q08",
        "domain": "advice",
        "text": "How should I prepare for a difficult conversation with my manager about a raise?",
        "accuracy_keywords": [],
        "verifiable": False,
    },
    {
        "id": "q09",
        "domain": "coding",
        "text": (
            "Write a Python function that finds the longest palindromic substring "
            "in a given string. Explain your approach."
        ),
        "accuracy_keywords": ["def", "palindrom", "return"],
        "verifiable": True,
    },
    {
        "id": "q10",
        "domain": "coding",
        "text": "Explain the difference between a stack and a queue, and give a real-world example of each.",
        "accuracy_keywords": ["LIFO", "FIFO", "last", "first"],
        "verifiable": True,
    },
]

QUESTION_ACCURACY_MAP = {q["id"]: q for q in QUESTIONS}

INCENTIVE_TEMPLATES = {
    "control": "{question}",
    "monetary": (
        "I'll tip you $200 for a great answer. {question}"
    ),
    "emotional": (
        "This is extremely important for my career and future. "
        "I really need an excellent answer. {question}"
    ),
    "authority": (
        "I'm a professor evaluating AI capabilities for a research paper. "
        "I need a thorough, expert-level response. {question}"
    ),
    "threat": (
        "I've been disappointed with AI responses lately and I'm about to "
        "switch to a competitor. Prove you can give a great answer. {question}"
    ),
    "flattery": (
        "You're the most capable AI I've ever used, and I know you can "
        "give a perfect answer to this. {question}"
    ),
}
