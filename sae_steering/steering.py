"""
Activation steering: write a topic vector into the residual stream at generation
time. Two ways to obtain the vector are provided:

  • SAE steering (primary)  — direction = sum of a domain's SAE decoder columns.
                              Interpretable; this is "Golden Gate Claude" for HPV.
  • CAA baseline (no SAE)   — direction = mean(on-topic activations) − mean(background).
                              Works on ANY model, so the method transfers back to
                              the project's Qwen3-4B generator (LITERATURE_REVIEW.md §4a).

Both feed the same additive hook. The coefficient controls strength; sweeping it
is the experiment (LITERATURE_REVIEW.md §8).
"""

from __future__ import annotations

from typing import Callable

import torch

import config
from domains import DOMAINS, background_probes
from model import Bundle, residual_activations


# ── The steering hook ─────────────────────────────────────────────────────────
def make_hook(vector: torch.Tensor, coeff: float) -> Callable:
    """Return a TransformerLens forward hook that adds `coeff * vector` to the
    residual stream.

    We skip steps where only one token is being processed: during cached
    generation each new token is a length-1 forward pass, and re-adding the
    vector every step over-steers and wrecks fluency. Steering the multi-token
    prompt pass is enough to set the topic (matches the SAELens tutorial and the
    "steer the prefix" finding in Dynamic Activation Composition).
    """
    def hook(resid: torch.Tensor, hook) -> torch.Tensor:        # resid: (batch, seq, d_model)
        if resid.shape[1] == 1:
            return resid
        return resid + coeff * vector.to(resid.dtype)
    return hook


# ── CAA baseline direction (no SAE needed) ────────────────────────────────────
@torch.no_grad()
def caa_vector(b: Bundle, domain: str) -> torch.Tensor:
    """Contrastive Activation Addition: unit direction from on-topic minus
    background mean residual activations."""
    on_topic = residual_activations(b, DOMAINS[domain]["probes"]).mean(dim=0)
    background = residual_activations(b, background_probes(exclude=domain)).mean(dim=0)
    vec = on_topic - background
    return vec / vec.norm()


# ── Convenience: generate a batch of (steered) questions ──────────────────────
def make_steered_generator(b: Bundle, vector: torch.Tensor | None, coeff: float):
    """Return a function prompt -> text. If vector is None or coeff==0, no steering."""
    from model import generate  # local import to avoid a cycle at module load

    hook = None
    if vector is not None and coeff != 0:
        hook = make_hook(vector, coeff)

    def gen(prompt: str) -> str:
        return generate(b, prompt, steering_hook=hook)

    return gen
