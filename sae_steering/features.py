"""
Find the SAE features (the "topic vectors") for each HPV domain.

Method (training-free, contrastive — see LITERATURE_REVIEW.md §3):
  1. Encode on-topic probe sentences and background (other-domain) sentences
     into SAE feature space.
  2. Rank features by  mean_activation_on_topic − mean_activation_background.
  3. The top features are the domain's signature; their decoder columns,
     summed and normalized, form the steering vector.

We also print a Neuronpedia link per feature so a human can confirm the label
("is feature 4711 really *vaccination*?") before trusting it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import torch

import config
from domains import DOMAINS, background_probes
from model import Bundle, residual_activations


@dataclass
class DomainFeatures:
    domain: str
    label: str
    indices: list[int]          # SAE feature ids, strongest first
    scores: list[float]         # on-topic minus background separation
    neuronpedia: list[str]      # one dashboard URL per feature


@torch.no_grad()
def _feature_means(b: Bundle, texts: list[str]) -> torch.Tensor:
    """Mean SAE feature activation across `texts`. Shape: (d_sae,)."""
    resid = residual_activations(b, texts)          # (n_texts, d_model)
    feats = b.sae.encode(resid)                      # (n_texts, d_sae)
    return feats.mean(dim=0)                          # (d_sae,)


def find_for_domain(b: Bundle, domain: str, top_k: int = config.TOP_FEATURES_PER_DOMAIN) -> DomainFeatures:
    """Rank SAE features that are specific to `domain`."""
    on_topic = DOMAINS[domain]["probes"]
    background = background_probes(exclude=domain)

    sep = _feature_means(b, on_topic) - _feature_means(b, background)   # (d_sae,)
    top = torch.topk(sep, k=top_k)
    indices = top.indices.tolist()
    scores = top.values.tolist()

    return DomainFeatures(
        domain=domain,
        label=DOMAINS[domain]["label"],
        indices=indices,
        scores=scores,
        neuronpedia=[config.neuronpedia_url(i, config.SAE_LAYER) for i in indices],
    )


def find_all(b: Bundle) -> dict[str, DomainFeatures]:
    """Find features for every domain and cache the result to JSON."""
    out: dict[str, DomainFeatures] = {}
    for domain in DOMAINS:
        df = find_for_domain(b, domain)
        out[domain] = df
        print(f"[features] {domain:16s} → {df.indices[:5]} ... (top sep={df.scores[0]:.3f})")

    os.makedirs(os.path.dirname(config.FEATURES_CACHE_PATH), exist_ok=True)
    with open(config.FEATURES_CACHE_PATH, "w") as f:
        json.dump({k: asdict(v) for k, v in out.items()}, f, indent=2)
    print(f"[features] cached → {config.FEATURES_CACHE_PATH}")
    return out


def steering_vector(b: Bundle, df: DomainFeatures, n_features: int | None = None) -> torch.Tensor:
    """Build a unit steering vector from a domain's decoder columns.

    Summing the decoder directions of the domain's features gives the direction
    that writes "this domain" into the residual stream. We normalize so the
    coefficient alone controls strength (LITERATURE_REVIEW.md §4c).
    """
    idx = df.indices if n_features is None else df.indices[:n_features]
    directions = b.sae.W_dec[idx]                    # (n_features, d_model)
    vec = directions.sum(dim=0)                       # (d_model,)
    return vec / vec.norm()


@torch.no_grad()
def on_target_score(b: Bundle, texts: list[str], df: DomainFeatures) -> float:
    """How strongly do `texts` activate the domain's features? Used to MEASURE
    whether steering worked — the same SAE that steers also evaluates."""
    if not texts:
        return 0.0
    feats = _feature_means(b, texts)                  # (d_sae,)
    return feats[df.indices].mean().item()
