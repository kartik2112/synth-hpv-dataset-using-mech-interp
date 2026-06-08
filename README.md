# Synthetic HPV QA Dataset Generator

Generates synthetic question-answer pairs about HPV using a **seed-QA guided**
approach. A small set of curated seed pairs is shown to the model on each call to
anchor style and accuracy, while topic hints rotate to ensure topical diversity
across the generated pairs.

The default model is **`google/gemma-2-2b-it`** — the same model used by the
mechanistic-interpretability subproject in [`sae_steering/`](./sae_steering/), so
the topic vectors and steering operating points discovered there apply to this
generator directly. Unsloth 4-bit **Qwen** models still work on CUDA via `--model`.

## File structure

```
.
├── generate.py            # Main generation script
├── utils.py               # Helpers: prompt building, parsing, I/O, deduplication
├── data/
│   ├── seed_qa.json       # 20 curated seed QA pairs (input)
│   └── generated_qa.json  # Generated output (created after running)
├── sae_steering/          # SAE topic-vector + activation-steering experiments
└── README.md
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). Runs on **Apple Silicon (MPS)**, **CUDA**,
or **CPU** — no GPU strictly required (CPU is slow, for testing only).

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

**Gemma is gated.** Before the first run, accept the license at
<https://huggingface.co/google/gemma-2-2b-it> and authenticate:

```bash
export HF_TOKEN=hf_xxx        # or: huggingface-cli login
```

> **Apple Silicon (macOS):** `uv sync` picks the MPS-enabled PyPI torch wheel
> automatically. The model loads via `transformers` in **bfloat16** — note that
> Gemma-2 must *not* run in float16 (its attention soft-capping overflows and
> produces NaNs), which is why the script uses bfloat16/float32 only.
>
> **CUDA (Linux):** `pyproject.toml` defaults to CUDA 12.4 (`cu124`); edit the
> index URL for a different version. `unsloth` + `bitsandbytes` install on Linux
> only and power the 4-bit fast path used for Unsloth/Qwen models.

## Usage

**Generate 100 pairs (default, Gemma-2-2B-it):**
```bash
uv run generate.py
```

**Custom options:**
```bash
uv run generate.py \
  --target 200 \
  --output data/my_dataset.json \
  --temperature 0.9
```

**Use a different model** (e.g. the Unsloth 4-bit Qwen fast path on CUDA):
```bash
uv run generate.py --model unsloth/Qwen3-4B-bnb-4bit --thinking
```

`--thinking` enables Qwen3's extended reasoning; it is Qwen-only and silently
ignored by Gemma.

## How it works

Each LLM call follows this pattern:

1. **Sample seeds** — 5 pairs are drawn from the seed pool (which grows to include
   already-generated pairs, enabling self-seeding).
2. **Topic hint** — one of 12 predefined hints is cycled in to diversify coverage.
3. **Prompt** — the model receives the seed examples and is asked for 10 new
   JSON-formatted QA pairs on the given topic. (A `system` instruction is folded
   into the user turn automatically for templates like Gemma's that reject it.)
4. **Parse & deduplicate** — output is parsed as JSON; pairs duplicating existing
   questions are discarded.
5. **Repeat** until the target count is reached.

## Output format

`generated_qa.json` — JSON array of objects:
```json
[
  {
    "question": "Can HPV be detected through a routine blood test?",
    "answer": "No. There is currently no approved blood test for HPV. ...",
    "id": "synth_0000",
    "source": "synthetic",
    "model": "google/gemma-2-2b-it",
    "device": "mps"
  }
]
```

`generated_qa.jsonl` — same data, one JSON object per line (convenient for
training pipelines).

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--model` | `google/gemma-2-2b-it` | HF model name (Unsloth 4-bit → CUDA fast path) |
| `--seed-path` | `data/seed_qa.json` | Seed QA JSON file |
| `--output` | `data/generated_qa.json` | Output path |
| `--target` | `100` | Number of pairs to generate |
| `--temperature` | `0.8` | Sampling temperature |
| `--top-p` | `0.95` | Top-p nucleus sampling |
| `--max-new-tokens` | `1536` | Max tokens per LLM call |
| `--thinking` | off | Qwen3 extended thinking (Qwen-only) |
| `--seed` | `42` | Random seed |

## Mechanistic-interpretability subproject

[`sae_steering/`](./sae_steering/) uses sparse autoencoders to find the model's
internal "topic vectors" for each HPV subtopic and steers activations to
over-sample under-represented domains — without letting steering touch factual
correctness. Because the generator now runs the same Gemma-2-2B model, those
steering directions can be folded straight into generation. Start with
[`sae_steering/LITERATURE_REVIEW.md`](./sae_steering/LITERATURE_REVIEW.md).
