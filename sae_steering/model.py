"""
Load the model and its sparse autoencoder, and provide two small helpers:
capturing residual-stream activations and generating text (optionally steered).

We use TransformerLens (`HookedTransformer`) because it exposes every internal
activation as a named hook — the substrate that makes steering a two-line change.
SAEs come from SAELens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

import config


@dataclass
class Bundle:
    """Everything the rest of the code needs, loaded once."""
    model: HookedTransformer
    sae: SAE
    hook_name: str
    device: str


def load(layer: int = config.SAE_LAYER) -> Bundle:
    """Load the instruct model and the Gemma Scope SAE for `layer`."""
    device = config.DEVICE
    print(f"[model] loading {config.MODEL_NAME} on {device} ({config.DTYPE}) ...")
    model = HookedTransformer.from_pretrained(
        config.MODEL_NAME,
        device=device,
        dtype=config.DTYPE,
    )
    model.eval()

    print(f"[model] loading SAE {config.SAE_RELEASE} / {config.sae_id_for_layer(layer)} ...")
    loaded = SAE.from_pretrained(
        release=config.SAE_RELEASE,
        sae_id=config.sae_id_for_layer(layer),
        device=device,
    )
    # SAELens has returned either an SAE or a (sae, cfg, sparsity) tuple across
    # versions — handle both so the code doesn't break on upgrade.
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    sae = sae.to(device)

    # Prefer the hook name the SAE was trained on; fall back to the convention.
    try:
        hook_name = sae.cfg.metadata.hook_name
    except AttributeError:
        hook_name = config.hook_name_for_layer(layer)

    print(f"[model] ready. d_model={model.cfg.d_model}, d_sae={sae.W_dec.shape[0]}, "
          f"hook={hook_name}")
    return Bundle(model=model, sae=sae, hook_name=hook_name, device=device)


# ── Activation capture (used to find domain features) ─────────────────────────
@torch.no_grad()
def residual_activations(b: Bundle, texts: list[str]) -> torch.Tensor:
    """Return the mean residual-stream activation per text at the hook layer.

    Output shape: (len(texts), d_model). We average over tokens so each text is a
    single point in residual space — enough to rank features by domain.
    """
    means = []
    for text in texts:
        tokens = b.model.to_tokens(text)                       # (1, seq)
        _, cache = b.model.run_with_cache(tokens, names_filter=b.hook_name)
        acts = cache[b.hook_name][0]                            # (seq, d_model)
        means.append(acts.mean(dim=0))
    return torch.stack(means)                                  # (n_texts, d_model)


# ── Text generation, with an optional steering hook ───────────────────────────
def _to_chat_tokens(b: Bundle, user_prompt: str) -> torch.Tensor:
    """Apply the gemma chat template and tokenize (the template already adds BOS)."""
    text = b.model.tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return b.model.to_tokens(text, prepend_bos=False)


@torch.no_grad()
def generate(
    b: Bundle,
    user_prompt: str,
    steering_hook: Callable | None = None,
    max_new_tokens: int = config.GEN_MAX_NEW_TOKENS,
    temperature: float = config.GEN_TEMPERATURE,
    top_p: float = config.GEN_TOP_P,
) -> str:
    """Generate a reply. If `steering_hook` is given, it edits the residual stream
    during the forward pass (see steering.py)."""
    tokens = _to_chat_tokens(b, user_prompt)
    n_prompt = tokens.shape[1]

    fwd_hooks = [(b.hook_name, steering_hook)] if steering_hook is not None else []
    with b.model.hooks(fwd_hooks=fwd_hooks):
        out = b.model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            stop_at_eos=False,     # avoids a known MPS generation bug
            verbose=False,
        )
    new_tokens = out[0, n_prompt:]
    return b.model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
