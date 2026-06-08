# SAE Topic Steering for Diverse, Factual HPV QA

Use **sparse autoencoders** to find the internal "topic vectors" a model uses for
each HPV subtopic, then **steer activations** along those vectors to over-sample
under-represented domains — *without* letting steering touch factual correctness.

This is the mechanistic-interpretability companion to the prompt-based generator
in the repo root. Start with **[`LITERATURE_REVIEW.md`](./LITERATURE_REVIEW.md)** —
it explains the approach, the method survey, the tooling choices, and the
experiment design. This README is just how to run it.

## The idea in three lines

1. **Find** the SAE features that fire for a domain (e.g. *immunocompromised*) →
   their decoder directions are the topic vector. (`features.py`)
2. **Steer** generation along that vector so the model *asks about* that domain.
   (`steering.py`)
3. **Verify** factuality separately: the *answer* is generated un-steered and
   checked against the curated seeds + a model judge. Steering never adds facts.
   (`factuality.py`)

## Why Gemma, not the project's Qwen3-4B?

No pretrained SAE exists for Qwen3-4B (Qwen-Scope starts at 8B), so the SAE work
uses **Gemma-2-2B + Gemma Scope** — the best-tooled, MPS-friendly path with
pre-labeled features. The **generator was unified onto the same Gemma-2-2B**
(see the root README), so the topic vectors and operating point found here apply
to it directly. A **CAA baseline** (`--method caa`) remains as an SAE-free
comparison. Full reasoning in the literature review, §7.

## Setup

Gemma-2-2b-it is **gated**. Once:

```bash
# 1. Accept the license at https://huggingface.co/google/gemma-2-2b-it
# 2. Authenticate
export HF_TOKEN=hf_xxx            # or: huggingface-cli login
# 3. Install the extra deps into the project venv
uv pip install -r sae_steering/requirements.txt
```

## Run

```bash
cd sae_steering

# Sweep steering strength on an under-covered domain (the core experiment):
python run_experiment.py --domain immunocompromised

# A few more:
python run_experiment.py --domain screening --coeffs 0 4 8 12 --n 8
python run_experiment.py --domain men --method caa      # SAE-free baseline
python run_experiment.py --domain pregnancy --no-judge  # grounding only, faster
python run_experiment.py --domain all --n 4             # every domain (slow)
```

## Reading the output

Each row of the sweep prints three numbers:

| Column | Axis | What you want |
|---|---|---|
| `on_target` | **diversity** — mean activation of the domain's SAE features in the generated questions | to **rise** with the coefficient |
| `distinct2` | lexical diversity (distinct bigrams) | to stay healthy (not collapse to one phrase) |
| `factuality` | pass-rate of grounded + judged answers | to stay **flat** |

**Operating point** = the largest coefficient where `on_target` is clearly up but
`factuality` hasn't dropped. Just below the knee. That coefficient is the result
you carry over to bulk generation.

Artifacts land in `results/`: `domain_features.json` (the topic vectors, with
Neuronpedia links to verify each feature's meaning) and `steering_report.json`
(the full sweep, generated questions, and per-item fact-checks).

## File map

| File | Role |
|---|---|
| `config.py` | all knobs: model, SAE release/layer, device, coefficient sweep |
| `domains.py` | the 12 HPV domains + probe sentences (mirrors `../utils.py` topic hints) |
| `model.py` | load model + SAE; capture activations; generate (optionally steered) |
| `features.py` | find domain features (topic vectors); build steering vectors; score on-target |
| `steering.py` | the additive steering hook + CAA baseline direction |
| `factuality.py` | grounding-vs-seeds + model-as-judge verification |
| `run_experiment.py` | orchestrate the sweep, compute metrics, write the report |

## Caveats (read §9 of the literature review)

Steering is **fragile** and can break behavior at high strength — that's why we
sweep, steer at a mid layer, and never trust steering for correctness. This is a
**medical** dataset: every fact must come from grounding + verification, and ideally
clinician review, before any downstream use. The code is conservative by design —
it **drops** doubtful items rather than guessing.
