"""
utils.py — helpers for seed-guided HPV QA generation
"""

import json
import random
import re
from pathlib import Path
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────

SEED_QA_PATH = "data/seed_qa.json"
OUTPUT_PATH = "data/generated_qa.json"

# Topic hints cycled during generation to encourage topical diversity
TOPIC_HINTS = [
    "HPV prevention and vaccine schedules",
    "HPV transmission routes and risk factors",
    "HPV symptoms, warts, and diagnosis",
    "cervical cancer screening (Pap smear and HPV test)",
    "HPV-related cancers: cervical, anal, oropharyngeal",
    "HPV in men and gender-specific considerations",
    "HPV and pregnancy or fertility",
    "HPV in immunocompromised individuals (HIV, transplant)",
    "high-risk vs. low-risk HPV types and their consequences",
    "common myths and misconceptions about HPV",
    "HPV treatment options and management of sequelae",
    "HPV epidemiology and public health statistics",
]


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_seed_qa(path: str = SEED_QA_PATH) -> list[dict]:
    """Load seed QA pairs from a JSON file (list of {question, answer} dicts)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data)}")
    return data


def save_dataset(pairs: list[dict], path: str = OUTPUT_PATH) -> None:
    """Save generated QA pairs to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(pairs)} pairs → {path}")


def save_jsonl(pairs: list[dict], path: str) -> None:
    """Save generated QA pairs to a JSONL file (one JSON object per line)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} pairs → {path}")


# ── Prompt construction ───────────────────────────────────────────────────────

def sample_seeds(pool: list[dict], n: int = 5) -> list[dict]:
    """Randomly sample n QA pairs from the pool."""
    return random.sample(pool, min(n, len(pool)))


def build_prompt(
    seeds: list[dict],
    n_generate: int = 10,
    topic_hint: Optional[str] = None,
) -> list[dict]:
    """
    Build a chat-format prompt (list of role/content dicts) that shows a few
    seed QA pairs and asks the model to generate n_generate new diverse pairs.
    """
    seed_block = "\n\n".join(
        f"Q: {p['question']}\nA: {p['answer']}" for p in seeds
    )

    topic_str = f" Focus especially on: **{topic_hint}**." if topic_hint else ""

    system_msg = (
        "You are a medical education expert creating high-quality question-answer pairs "
        "about Human Papillomavirus (HPV) for patient education and clinical training datasets. "
        "Every answer must be medically accurate, evidence-based, and written clearly enough "
        "for a general adult audience to understand."
    )

    user_msg = f"""Here are example HPV question-answer pairs for reference:

{seed_block}

---

Generate {n_generate} NEW, diverse question-answer pairs about HPV.{topic_str}
- Do NOT repeat or closely paraphrase any question shown above.
- Cover a range of subtopics (prevention, transmission, symptoms, screening, treatment, cancer risk, epidemiology, etc.).
- Keep answers factually accurate, 2–5 sentences each.

Respond ONLY with a valid JSON array using this exact format — no extra text, no markdown fences:
[
  {{"question": "...", "answer": "..."}},
  ...
]"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ── Output parsing ────────────────────────────────────────────────────────────

def parse_qa_output(text: str) -> list[dict]:
    """
    Extract a list of {question, answer} dicts from raw model output.

    Handles:
    - Qwen3 <think>...</think> blocks
    - Extra prose before/after the JSON array
    - Malformed JSON (returns [] on failure)
    """
    # Strip thinking-mode tags (Qwen3 when enable_thinking=True)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Extract the first [...] block
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []

    try:
        pairs = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    # Validate structure and minimum content length
    valid = []
    for p in pairs:
        if (
            isinstance(p, dict)
            and isinstance(p.get("question"), str)
            and isinstance(p.get("answer"), str)
            and len(p["question"].strip()) > 10
            and len(p["answer"].strip()) > 20
        ):
            valid.append({"question": p["question"].strip(), "answer": p["answer"].strip()})

    return valid


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate(new_pairs: list[dict], existing_pairs: list[dict]) -> list[dict]:
    """
    Filter out pairs whose questions are exact duplicates (case-insensitive)
    of questions already in existing_pairs.
    """
    seen = {p["question"].lower().strip() for p in existing_pairs}
    unique = []
    for pair in new_pairs:
        key = pair["question"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    return unique
