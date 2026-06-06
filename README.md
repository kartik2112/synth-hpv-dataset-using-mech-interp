# Synthetic HPV QA Dataset Generator

Generates synthetic question-answer pairs about HPV using a **seed-QA guided** approach with Unsloth's Qwen3 4B (4-bit quantized). A small set of curated seed pairs is shown to the model each call to anchor style and accuracy, while topic hints rotate to ensure topical diversity across the 100 generated pairs.

## File structure

```
.
├── generate.py          # Main generation script
├── utils.py             # Helpers: prompt building, parsing, I/O, deduplication
├── data/
│   ├── seed_qa.json     # 20 curated seed QA pairs (input)
│   └── generated_qa.json   # Generated output (created after running)
└── README.md
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a CUDA-capable GPU (~6 GB VRAM, e.g. RTX 3060 or T4).

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (downloads torch + unsloth automatically)
uv sync
```

> **CUDA version (Linux):** `pyproject.toml` defaults to CUDA 12.4 (`cu124`). If your system uses a different version, edit the index URL before running `uv sync` (e.g. `cu121` for CUDA 12.1).
>
> **Apple Silicon (macOS):** `uv sync` picks the PyPI torch wheel automatically (MPS-enabled). `unsloth` and `bitsandbytes` are Linux-only and are skipped. The script loads the base model via `transformers` in float16.

## Usage

**Generate 100 pairs (default):**
```bash
uv run generate.py
```

**Custom options:**
```bash
uv run generate.py \
  --target 200 \
  --output data/my_dataset.json \
  --temperature 0.9 \
  --thinking          # enable Qwen3 thinking mode (slower, often better)
```

**Use a different model:**
```bash
uv run generate.py --model unsloth/Qwen2.5-4B-bnb-4bit
```

All available Unsloth models: https://huggingface.co/unsloth

## How it works

Each LLM call follows this pattern:

1. **Sample seeds** — 5 pairs are randomly drawn from the seed pool (which grows to include already-generated pairs, enabling self-seeding).
2. **Topic hint** — a topic (e.g., *"HPV in immunocompromised individuals"*) is cycled through 12 predefined hints to diversify coverage.
3. **Prompt** — the model receives the seed examples and is asked to produce 10 new JSON-formatted QA pairs on the given topic.
4. **Parse & deduplicate** — output is parsed as JSON; pairs too similar to existing ones are discarded.
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
    "model": "unsloth/Qwen3-4B-bnb-4bit"
  },
  ...
]
```

`generated_qa.jsonl` — same data, one JSON object per line (convenient for training pipelines).

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--model` | `unsloth/Qwen3-4B-bnb-4bit` | Unsloth model name |
| `--seed-path` | `data/seed_qa.json` | Seed QA JSON file |
| `--output` | `data/generated_qa.json` | Output path |
| `--target` | `100` | Number of pairs to generate |
| `--temperature` | `0.8` | Sampling temperature |
| `--top-p` | `0.95` | Top-p nucleus sampling |
| `--max-new-tokens` | `1536` | Max tokens per LLM call |
| `--thinking` | off | Enable Qwen3 extended thinking |
| `--seed` | `42` | Random seed |
