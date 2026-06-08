"""
Run the topic-steering experiment end to end.

For one HPV domain it:
  1. finds the domain's SAE features (the topic vector),
  2. generates questions at several steering coefficients (the sweep),
  3. measures the two axes that matter — DIVERSITY (did we move on-topic?) and
     FACTUALITY (did the grounded+verified answer survive?),
  4. prints a table and saves a JSON report.

The point of the sweep is to find the operating point: the largest coefficient
that lifts on-target rate before factuality/fluency roll off (LITERATURE_REVIEW.md §8).

Usage (run from inside the sae_steering/ folder):
    python run_experiment.py --domain immunocompromised
    python run_experiment.py --domain screening --coeffs 0 4 8 12 --n 8
    python run_experiment.py --domain men --method caa        # SAE-free baseline
    python run_experiment.py --domain pregnancy --no-judge    # faster, grounding only
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict

import config
import model as M
import features as F
import steering as S
import factuality as FC
from domains import DOMAINS, ELICITATION_PROMPT


# ── A tiny lexical-diversity metric ───────────────────────────────────────────
def distinct_n(texts: list[str], n: int = 2) -> float:
    """Fraction of distinct n-grams across `texts`. Higher = more diverse."""
    grams: Counter = Counter()
    for t in texts:
        toks = t.lower().split()
        grams.update(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    total = sum(grams.values())
    return (len(grams) / total) if total else 0.0


def answer_unsteered(b: M.Bundle, question: str) -> str:
    """Answers are ALWAYS generated without steering (LITERATURE_REVIEW.md §5)."""
    prompt = f"Answer this patient question about HPV accurately and concisely:\n{question}"
    return M.generate(b, prompt, steering_hook=None, temperature=0.3, top_p=0.9)


def run_domain(
    b: M.Bundle,
    domain: str,
    coeffs: list[int],
    n: int,
    method: str,
    use_judge: bool,
) -> dict:
    seeds = FC.load_seeds()

    # 1. Topic vector for this domain.
    df = F.find_for_domain(b, domain)
    print(f"\n[run] domain='{domain}' ({df.label})")
    print(f"[run] top features: {df.indices[:8]}")
    print(f"[run] verify feature #0 → {df.neuronpedia[0]}")

    if method == "sae":
        vector = F.steering_vector(b, df)
    else:  # caa baseline
        vector = S.caa_vector(b, domain)

    rows = []
    for c in coeffs:
        gen = S.make_steered_generator(b, vector, c)
        questions = [gen(ELICITATION_PROMPT) for _ in range(n)]

        # DIVERSITY axis: did steering move generations onto the domain?
        on_target = F.on_target_score(b, questions, df)
        d2 = distinct_n(questions, 2)

        # FACTUALITY axis: answer (un-steered) + verify, on a small subset to save time.
        checks = []
        for q in questions[: min(n, 4)]:
            ans = answer_unsteered(b, q)
            checks.append(FC.check(b, q, ans, seeds, use_judge=use_judge))
        pass_rate = sum(ch.passed for ch in checks) / len(checks) if checks else 0.0

        rows.append({
            "coeff": c,
            "on_target": round(on_target, 4),
            "distinct2": round(d2, 3),
            "factuality_pass_rate": round(pass_rate, 3),
            "questions": questions,
            "checks": [asdict(ch) for ch in checks],
        })
        print(f"  coeff={c:>3}  on_target={on_target:7.4f}  distinct2={d2:5.3f}  "
              f"factuality={pass_rate:4.0%}")

    return {"domain": domain, "label": df.label, "method": method,
            "features": asdict(df), "sweep": rows}


def main() -> None:
    p = argparse.ArgumentParser(description="SAE topic-steering experiment for HPV QA.")
    p.add_argument("--domain", default="immunocompromised", choices=list(DOMAINS) + ["all"])
    p.add_argument("--coeffs", type=int, nargs="+", default=config.COEFF_SWEEP)
    p.add_argument("--n", type=int, default=6, help="questions generated per coefficient")
    p.add_argument("--layer", type=int, default=config.SAE_LAYER)
    p.add_argument("--method", choices=["sae", "caa"], default="sae")
    p.add_argument("--no-judge", action="store_true", help="skip the model-judge (faster)")
    args = p.parse_args()

    b = M.load(layer=args.layer)

    domains = list(DOMAINS) if args.domain == "all" else [args.domain]
    report = [run_domain(b, d, args.coeffs, args.n, args.method, not args.no_judge)
              for d in domains]

    os.makedirs(os.path.dirname(config.REPORT_PATH), exist_ok=True)
    with open(config.REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] report → {config.REPORT_PATH}")
    print("[hint] read the table: pick the largest coeff where on_target rises but "
          "factuality stays flat — that's your operating point.")


if __name__ == "__main__":
    main()
