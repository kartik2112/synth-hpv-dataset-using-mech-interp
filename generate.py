"""
generate.py — seed-guided synthetic HPV QA dataset generation

Backends (selected automatically by available hardware):
  CUDA  → Unsloth + Qwen3-4B-bnb-4bit  (4-bit quantized, fastest)
  MPS   → transformers + Qwen3-4B       (float16, Apple Silicon)
  CPU   → transformers + Qwen3-4B       (float32, slow — for testing only)

Usage:
    uv run generate.py
    uv run generate.py --target 200 --output data/my_dataset.json
    uv run generate.py --model Qwen/Qwen3-4B --thinking
"""

import argparse
import re
import random
import sys
from pathlib import Path

import torch

from utils import (
    SEED_QA_PATH,
    TOPIC_HINTS,
    build_prompt,
    deduplicate,
    load_seed_qa,
    parse_qa_output,
    sample_seeds,
    save_dataset,
    save_jsonl,
)

# ── Default config ─────────────────────────────────────────────────────────────

# The default model is the CUDA 4-bit variant. On MPS/CPU the -bnb-4bit suffix
# is stripped automatically and the base weights are loaded instead.
MODEL_NAME = "unsloth/Qwen3-4B-bnb-4bit"
MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 1536
TEMPERATURE = 0.8
TOP_P = 0.95
N_SEEDS_PER_CALL = 5   # seed examples shown to the model each call
BATCH_SIZE = 10         # QA pairs requested per LLM call
DEFAULT_TARGET = 100


# ── Device detection ───────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_model_name(model_name: str, device: str) -> str:
    """
    On non-CUDA devices, strip the -bnb-4bit suffix so we load full-precision
    weights that transformers can handle. Prints a notice when the name changes.
    """
    if device != "cuda" and model_name.endswith("-bnb-4bit"):
        base = re.sub(r"-bnb-4bit$", "", model_name)
        print(
            f"[device={device}] Swapping quantized model name:\n"
            f"  {model_name!r}  →  {base!r}\n"
            f"  To use a different model, pass --model explicitly.\n"
        )
        return base
    return model_name


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_cuda(model_name: str, max_seq_length: int):
    """Load via Unsloth with 4-bit quantization (CUDA only)."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print(
            "ERROR: unsloth is not installed (CUDA device detected).\n"
            "       Run: uv sync",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Backend : Unsloth (4-bit)  |  model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,  # auto (bfloat16 on Ampere+, float16 otherwise)
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_model_mps_cpu(model_name: str, device: str):
    """Load via HuggingFace transformers for MPS or CPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"Backend : transformers ({dtype})  |  device: {device}  |  model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    return model, tokenizer


def load_model(model_name: str, max_seq_length: int, device: str):
    model_name = resolve_model_name(model_name, device)
    if device == "cuda":
        model, tokenizer = load_model_cuda(model_name, max_seq_length)
    else:
        model, tokenizer = load_model_mps_cpu(model_name, device)
    print("Model ready.\n")
    return model, tokenizer


# ── Inference ──────────────────────────────────────────────────────────────────

def apply_chat_template(tokenizer, messages: list[dict], enable_thinking: bool) -> str:
    """
    Apply the model's chat template. Qwen3 supports `enable_thinking`;
    older tokenizers (Qwen2.x) do not — falls back gracefully.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_batch(
    model,
    tokenizer,
    messages: list[dict],
    device: str,
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Run one generation call and return the decoded output text."""
    prompt_text = apply_chat_template(tokenizer, messages, enable_thinking)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ── Main generation loop ───────────────────────────────────────────────────────

def generate(args) -> list[dict]:
    device = get_device()
    print(f"Device  : {device}\n")

    seed_qa = load_seed_qa(args.seed_path)
    print(f"Seeds   : {len(seed_qa)} pairs from {args.seed_path}")

    model, tokenizer = load_model(args.model, args.max_seq_length, device)

    generated: list[dict] = []
    attempts = 0
    max_attempts = args.target * 4

    topics = TOPIC_HINTS.copy()
    random.shuffle(topics)

    print(f"Target  : {args.target} pairs  |  batch: {BATCH_SIZE}/call  |  thinking: {args.thinking}\n")
    print(f"{'Attempt':>7}  {'Topic':<40}  {'New':>4}  {'Total':>6}")
    print("-" * 65)

    while len(generated) < args.target and attempts < max_attempts:
        topic = topics[attempts % len(topics)]
        pool = seed_qa + generated
        seeds = sample_seeds(pool, N_SEEDS_PER_CALL)
        messages = build_prompt(seeds, n_generate=BATCH_SIZE, topic_hint=topic)

        raw = generate_batch(
            model, tokenizer, messages,
            device=device,
            enable_thinking=args.thinking,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        pairs = parse_qa_output(raw)
        unique = deduplicate(pairs, pool)
        generated.extend(unique)
        attempts += 1

        print(
            f"{attempts:>7}  {topic:<40}  {len(unique):>4}  "
            f"{min(len(generated), args.target):>6}/{args.target}"
        )

    generated = generated[: args.target]
    print(f"\nDone: {len(generated)} pairs in {attempts} LLM call(s).")
    return generated


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic HPV QA pairs (CUDA / MPS / CPU)"
    )
    parser.add_argument("--model", default=MODEL_NAME,
                        help="Model name (default: %(default)s). "
                             "On MPS/CPU the -bnb-4bit suffix is stripped automatically.")
    parser.add_argument("--seed-path", default=SEED_QA_PATH,
                        help="Path to seed QA JSON (default: %(default)s)")
    parser.add_argument("--output", default="data/generated_qa.json",
                        help="Output JSON path (default: %(default)s)")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help="Number of QA pairs to generate (default: %(default)s)")
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH,
                        help="Model max sequence length (default: %(default)s)")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                        help="Max new tokens per call (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help="Sampling temperature (default: %(default)s)")
    parser.add_argument("--top-p", type=float, default=TOP_P,
                        help="Top-p nucleus sampling (default: %(default)s)")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable Qwen3 thinking mode (slower, often better quality)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: %(default)s)")
    args = parser.parse_args()

    random.seed(args.seed)

    generated = generate(args)

    device = get_device()
    for i, pair in enumerate(generated):
        pair["id"] = f"synth_{i:04d}"
        pair["source"] = "synthetic"
        pair["model"] = args.model
        pair["device"] = device

    save_dataset(generated, args.output)
    save_jsonl(generated, str(Path(args.output).with_suffix(".jsonl")))


if __name__ == "__main__":
    main()
