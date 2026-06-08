"""
Central configuration for the SAE topic-steering experiments.

Everything that you might want to tweak lives here so the other modules stay
short and readable. See LITERATURE_REVIEW.md (section 7) for why these defaults
were chosen.
"""

from __future__ import annotations

import os
import torch


# ── Device ────────────────────────────────────────────────────────────────────
def pick_device() -> str:
    """Prefer Apple-Silicon GPU (MPS), then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = pick_device()

# float32 is the safe choice on MPS; switch to bfloat16 to halve memory if your
# Mac is tight on RAM (2B in fp32 ≈ 10 GB, in bf16 ≈ 5 GB).
DTYPE = torch.float32


# ── Model + SAE (see LITERATURE_REVIEW.md §7) ─────────────────────────────────
# We do the real SAE work on Gemma-2-2B because Gemma Scope is the best-tooled,
# best-labeled pretrained-SAE suite and it runs on a Mac. (No SAE exists for the
# project's Qwen3-4B; Qwen-Scope starts at 8B.)
# TransformerLens alias for the same weights the generator uses
# (HF name: google/gemma-2-2b-it). Instruct model → follows "ask a question" prompts.
MODEL_NAME = "gemma-2-2b-it"

# Gemma Scope canonical residual-stream SAEs (JumpReLU, 16k features per layer).
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_LAYER = 12                         # mid-depth (Gemma-2-2B has 26 layers)
SAE_WIDTH = "16k"                      # 16k features is plenty and cheapest to load

def sae_id_for_layer(layer: int = SAE_LAYER, width: str = SAE_WIDTH) -> str:
    """SAELens id for a Gemma Scope canonical residual SAE."""
    return f"layer_{layer}/width_{width}/canonical"


def hook_name_for_layer(layer: int = SAE_LAYER) -> str:
    """TransformerLens hook for the residual stream after a given block."""
    return f"blocks.{layer}.hook_resid_post"


# Gemma-2-2b-it is a gated model. Accept the license on HuggingFace and set a
# token (export HF_TOKEN=...) before first run. We read it here for convenience.
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


# ── Feature search (features.py) ──────────────────────────────────────────────
TOP_FEATURES_PER_DOMAIN = 12           # how many SAE features define a "topic vector"


# ── Steering + generation (steering.py) ───────────────────────────────────────
# The coefficient sweep IS the experiment (see LITERATURE_REVIEW.md §8). Find the
# largest coefficient that raises on-target rate before fluency/factuality drop.
COEFF_SWEEP = [0, 2, 4, 8, 12, 16]
DEFAULT_COEFF = 8

GEN_MAX_NEW_TOKENS = 48
GEN_TEMPERATURE = 0.9                   # we WANT diversity in the questions
GEN_TOP_P = 0.95


# ── Output ────────────────────────────────────────────────────────────────────
REPORT_PATH = os.path.join(os.path.dirname(__file__), "results", "steering_report.json")
FEATURES_CACHE_PATH = os.path.join(os.path.dirname(__file__), "results", "domain_features.json")

NEURONPEDIA_BASE = "https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-16k/{index}"


def neuronpedia_url(index: int, layer: int = SAE_LAYER) -> str:
    """Link to a feature's auto-interpreted dashboard for human verification."""
    return NEURONPEDIA_BASE.format(layer=layer, index=index)
