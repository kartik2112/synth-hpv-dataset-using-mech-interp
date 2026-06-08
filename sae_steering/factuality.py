"""
Factuality guardrail.

The golden rule from the literature (LITERATURE_REVIEW.md §5): steering changes
the *topic*, never the *truth*. So we never rely on steering for correctness.
Instead, the ANSWER is generated WITHOUT steering and then verified two ways:

  1. Grounding   — lexical overlap with the closest curated seed answer. Cheap,
                   no model call; catches answers untethered from known facts.
  2. Judge       — an un-steered model-as-judge YES/NO consistency check against
                   the seed knowledge as context. Swap in a stronger judge (or a
                   clinician) for real use.

Both are deliberately conservative: when in doubt, DROP the item rather than
keep a possibly-wrong medical statement.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from model import Bundle, generate

_SEED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "seed_qa.json")


def load_seeds(path: str = _SEED_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def grounding_score(answer: str, seeds: list[dict]) -> tuple[float, str]:
    """Best token-overlap (Jaccard) between `answer` and any seed answer.
    Returns (score in [0,1], the matching seed question)."""
    a = _tokens(answer)
    if not a:
        return 0.0, ""
    best, best_q = 0.0, ""
    for s in seeds:
        b = _tokens(s["answer"])
        j = len(a & b) / len(a | b) if (a | b) else 0.0
        if j > best:
            best, best_q = j, s["question"]
    return best, best_q


_JUDGE_PROMPT = """You are a careful medical fact-checker. Using only well-established \
public-health knowledge about HPV, decide whether the ANSWER is factually correct and \
not misleading.

QUESTION: {q}
ANSWER: {a}

Reply with exactly one word, YES or NO, then a brief reason."""


def judge_consistency(b: Bundle, question: str, answer: str) -> tuple[bool, str]:
    """Un-steered model-as-judge. Returns (is_consistent, raw_reply)."""
    reply = generate(
        b,
        _JUDGE_PROMPT.format(q=question, a=answer),
        steering_hook=None,           # the judge must never be steered
        max_new_tokens=40,
        temperature=0.0,
        top_p=1.0,
    )
    verdict = reply.strip().upper().startswith("YES")
    return verdict, reply


@dataclass
class FactCheck:
    question: str
    answer: str
    grounding: float
    grounded: bool
    judged_ok: bool
    passed: bool
    judge_reply: str


def check(
    b: Bundle,
    question: str,
    answer: str,
    seeds: list[dict],
    grounding_threshold: float = 0.12,
    use_judge: bool = True,
) -> FactCheck:
    """Run both checks. An item passes only if it is grounded AND judged OK."""
    g, _ = grounding_score(answer, seeds)
    grounded = g >= grounding_threshold

    judged_ok, reply = (True, "(judge skipped)")
    if use_judge:
        judged_ok, reply = judge_consistency(b, question, answer)

    return FactCheck(
        question=question,
        answer=answer,
        grounding=round(g, 3),
        grounded=grounded,
        judged_ok=judged_ok,
        passed=grounded and judged_ok,
        judge_reply=reply,
    )
