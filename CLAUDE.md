# CLAUDE.md

Guidance for AI agents (and humans) working in this repo.

## What this repo is

Two related pieces for building a **synthetic HPV question–answer dataset** for
patient education / clinical training:

1. **Prompt-based generator** (repo root) — seed-QA guided sampling from an LLM.
2. **SAE topic-steering experiments** (`sae_steering/`) — mechanistic-interpretability
   approach that finds per-domain "topic vectors" with sparse autoencoders and
   steers activations to produce domain-diverse, factually-checked questions.

Both halves run the **same model — `google/gemma-2-2b-it`** (default) — so steering
directions found in `sae_steering/` apply to the generator directly. Unsloth 4-bit
Qwen models still work in the generator via `--model` on CUDA.

## Layout

```
generate.py            # main generation loop (prompt-based)
utils.py               # prompt building, parsing, dedup, I/O; TOPIC_HINTS (12 HPV subtopics)
data/seed_qa.json      # 20 curated HPV seed QA pairs (source of truth for facts)
data/generated_qa.*    # output (created at runtime)
pyproject.toml         # uv project, Python 3.13, platform-marked deps
sae_steering/          # SAE + activation-steering subproject (see its README)
  LITERATURE_REVIEW.md #   deep analysis, method survey, tooling, experiment design
  config.py domains.py model.py features.py steering.py factuality.py run_experiment.py
```

## Running

Prompt-based generator (existing):

```bash
uv sync
export HF_TOKEN=...                       # gemma-2-2b-it is gated
uv run generate.py                        # 100 pairs, Gemma-2-2B-it
uv run generate.py --model unsloth/Qwen3-4B-bnb-4bit --thinking   # Qwen fast path (CUDA)
```

SAE steering (new — see `sae_steering/README.md` for detail):

```bash
export HF_TOKEN=...                       # gemma-2-2b-it is gated
uv pip install -r sae_steering/requirements.txt
cd sae_steering && python run_experiment.py --domain immunocompromised
```

## Conventions & constraints

- **Package manager: uv.** `package = false` (script project, no entry point).
- **Platform-marked deps.** `unsloth` + `bitsandbytes` are Linux/CUDA-only;
  macOS uses plain `transformers` + MPS. Keep `sys_platform` markers intact when
  editing `pyproject.toml`.
- **Devices / dtype.** Generator default Gemma-2-2B-it via `transformers` in
  **bfloat16** on MPS/CUDA, float32 on CPU. **Never float16 for Gemma-2** — its
  attention soft-capping overflows fp16 → NaNs. The Unsloth 4-bit path is used
  only for `unsloth/*` or `-bnb-4bit` models on CUDA. SAE subproject: Gemma-2-2B
  on MPS/CUDA/CPU (auto-detected, float32).
- **Seeds are the factual ground truth.** Treat `data/seed_qa.json` as the
  authority; new content is validated against it, not invented.

## SAE subproject — the one rule to remember

**Steering changes the TOPIC, never the TRUTH.** Steer only the *question's*
subtopic; generate the *answer* un-steered and verify it (grounding vs seeds +
model-as-judge). Never use activation steering to inject or "fix" facts —
empirically it doesn't reliably improve factuality and is fragile at high
strength. Prefer mid-layer steering and always sweep the coefficient. Full
reasoning and references in `sae_steering/LITERATURE_REVIEW.md`.

## Backbone note

There is **no pretrained SAE for Qwen3-4B** (Qwen-Scope starts at 8B), so SAE work
uses **Gemma-2-2B + Gemma Scope** — and the generator was unified onto the same
Gemma-2-2B so steering applies directly (no cross-model transfer needed). The
**CAA baseline** (`--method caa`) remains as an SAE-free comparison and a recipe
that ports to any model if the generator is ever swapped (e.g. back to Qwen).
