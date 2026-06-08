# Topic Vectors via Sparse Autoencoders + Activation Steering for Diverse, Factual HPV QA Generation

**A literature review, method survey, and experiment design.**
Compiled June 2026 for the `synth-hpv-dataset-using-mech-interp` project.

---

## 1. The problem, stated precisely

The existing pipeline generates synthetic HPV question–answer pairs by few-shot prompting an LLM and rotating through 12 hand-written "topic hints." Prompt-level topic hints are a blunt instrument: the model drifts back to a few high-frequency subtopics (what is HPV, the vaccine), under-covers the long tail (immunocompromised patients, pregnancy, oropharyngeal cancer, epidemiology), and gives us no measurable handle on *how much* of each domain we actually produced.

This project asks a sharper question:

> Can we (a) locate the internal directions a model uses to represent each HPV subtopic ("topic vectors"), (b) push generation along those directions to deliberately over-sample under-represented domains, and (c) do so **without degrading factual correctness**?

The first two are squarely mechanistic-interpretability problems and are what sparse autoencoders (SAEs) and activation steering were built for. The third is the hard part, and most of the design effort below is spent making diversity-steering and factuality *not* fight each other.

---

## 2. Conceptual background

**Superposition and the linear representation hypothesis.** Transformer residual streams pack far more concepts than they have dimensions by representing features as *almost-orthogonal directions* that are active sparsely — "superposition." A large body of evidence supports a **linear representation hypothesis**: many human-interpretable concepts correspond to (roughly) linear directions in activation space, so concepts can be read off by a dot product and written by vector addition. This is the foundation that makes both "find a topic vector" and "add a topic vector" coherent operations.

**Why raw neurons don't work.** Individual MLP neurons are *polysemantic* — one neuron fires for many unrelated concepts — so you cannot just grab "the HPV neuron." SAEs solve exactly this.

**Sparse autoencoders.** An SAE is a wide one-hidden-layer autoencoder trained on a model's activations (typically the residual stream at one layer). It learns an over-complete dictionary: encode the d_model-dimensional activation into a much larger (16k–1M) but *sparse* feature vector, then decode back. Because only a handful of features are active at once, the learned features tend to be **monosemantic** — each corresponds to a single, often nameable concept (a named bridge, a programming token, "medical/clinical language," "vaccination"). Each feature `i` has:

- an **encoder row** (what activation pattern turns it on), used for *detection* / measurement, and
- a **decoder column** `W_dec[i]` (the direction it writes back into the residual stream), used for *steering*.

The decoder columns are the "topic vectors" we want.

Key references: Anthropic's *Towards Monosemanticity* and *Scaling Monosemanticity* (the "Golden Gate Claude" demonstration that clamping one feature reshapes all outputs); the survey *A Survey on Sparse Autoencoders* ([2503.05613](https://hf.co/papers/2503.05613)); and OpenAI's *Scaling and evaluating sparse autoencoders* introducing TopK SAEs ([2406.04093](https://hf.co/papers/2406.04093)).

**SAE architecture variants you'll meet.**

- **ReLU SAEs** — original; sparsity via an L1 penalty.
- **TopK SAEs** — keep exactly *k* features per token (OpenAI, [2406.04093](https://hf.co/papers/2406.04093)); this is what **Qwen-Scope** uses (k=50).
- **JumpReLU SAEs** — a learned activation threshold; this is what **Gemma Scope** uses, and gives the best reconstruction/sparsity trade-off at the 16k width we'll use.

---

## 3. Finding "topic vectors" with an SAE

There are three practical ways to identify the SAE features that represent an HPV subtopic, in increasing order of rigor. The shipped code uses the first two and points at the third.

1. **Activation ranking on probe text (contrastive).** Run a small set of on-topic sentences (e.g., the seed answers for "vaccination") through the model, capture SAE feature activations at the hook layer, and average over tokens. Subtract the average activations on a *background* corpus (generic or other-domain text). Features with the largest positive difference are the domain's signature. This is a tiny, training-free version of the "feature as a probe" idea and is robust enough to pick out "clinical/medical" and subtopic features. (Cf. *Less is Enough: Synthesizing Diverse Data in Feature Space* [2602.10388](https://hf.co/papers/2602.10388), which uses SAE *Feature Activation Coverage* as a diversity coordinate system — directly analogous to what we do for evaluation.)

2. **Human labels via Neuronpedia.** Gemma Scope features are auto-interpreted and browsable on [Neuronpedia](https://www.neuronpedia.org/gemma-2-2b/gemmascope-res-16k). After ranking, we print the Neuronpedia URL for each candidate feature so a human can confirm "yes, feature 12persuasion-something is *vaccination*, not *generic syringe*." This guards against the classic failure mode catalogued in *When the Coffee Feature Activates on Coffins* ([2601.03047](https://hf.co/papers/2601.03047)): features look clean until they fire on a near-synonym you didn't intend.

3. **Supervised probing in the sparse basis (most rigorous).** Train a linear classifier on SAE features to separate domain vs not-domain, and steer along the probe direction restricted to its top sparse coordinates. This is **SAE-SSV** ([2505.16188](https://hf.co/papers/2505.16188)) and the supervised half of **SAE-TS / Feature-Guided Activation Additions** ([2501.09929](https://hf.co/papers/2501.09929)). Recommended as a follow-up once the cheap method is working.

---

## 4. Activation steering: a taxonomy of how to *write* a topic vector

Steering = modify activations at inference time so outputs acquire a target property, with no weight updates. The methods differ in **where the direction comes from** and **how it is injected**.

### 4a. Where the direction comes from

- **Prompt-contrastive mean difference — ActAdd / CAA.** Take pairs of prompts that differ only in the property ("Talk about vaccines" vs "Talk about weather"), and use the *mean difference of activations* as the steering vector. **Contrastive Activation Addition** ([2312.06681](https://hf.co/papers/2312.06681)) is the canonical version and a strong, SAE-free baseline. We keep it as an SAE-free comparison; it also ports to any model unchanged (handy if the generator is ever swapped off Gemma).
- **SAE decoder direction (our primary method).** Use `W_dec[i]` for a chosen feature, or a sum over a domain's feature set. Interpretable by construction and the basis for Neuronpedia steering and "Golden Gate Claude."
- **Probe / concept-activation-vector — ITI, GCAV.** **Inference-Time Intervention** ([Li et al. 2023](https://arxiv.org/abs/2306.03341)) finds "truthful" attention heads with linear probes and nudges along them; **GCAV** ([2501.05764](https://hf.co/papers/2501.05764)) builds concept activation vectors for topic/sentiment/toxicity control.
- **Optimization-based — SAE-TS / FGAA.** Solve for the residual edit that maximizes the *target* feature while minimizing collateral feature changes ([2501.09929](https://hf.co/papers/2501.09929)). Cleaner, more selective steering; more code.

### 4b. How it is injected

- **Additive steering:** `resid += coeff · v̂` at one layer, applied to the prompt tokens (the standard recipe in the SAELens tutorial). Simple, what we use.
- **Feature clamping:** force feature `i`'s activation to a fixed value via the SAE (encode → set coordinate → decode), as Anthropic did for Golden Gate Claude and as Neuronpedia exposes. Slightly more faithful to "turn this concept up to 8"; included as an alternative.
- **Directional ablation (the opposite):** *remove* a direction to suppress a concept (used by *LangFIR* [2604.03532](https://hf.co/papers/2604.03532) for language steering and detox work) — useful here to *suppress* the over-represented "what is HPV / vaccine basics" features and force the long tail.

### 4c. The single most important empirical fact: the **strength/quality trade-off**

Every survey converges on the same curve. Quoting the consensus from the activation-steering literature ([CAA](https://hf.co/papers/2312.06681); *Multi-property Steering / Dynamic Activation Composition* [2406.17563](https://hf.co/papers/2406.17563); the [emergentmind activation-steering survey](https://www.emergentmind.com/topics/activation-steering-method)):

> Increasing concept incorporation degrades instruction-following and fluency. Excessive strength — **especially at early layers** — wrecks the output; **mid-to-late-layer** and **dynamic** interventions give the best trade-off.

Practical consequences baked into the design: steer at a **mid layer** (≈ 0.4–0.6 depth; layer 12 of Gemma-2-2B's 26), **sweep the coefficient** rather than guessing, and use a **dynamic / conditional** schedule (steer the topic-setting prefix, not every token) à la *Dynamic Activation Composition* ([2406.17563](https://hf.co/papers/2406.17563)) and *Steering When Necessary / FASB* ([2508.17621](https://hf.co/papers/2508.17621)).

---

## 5. Factuality: why steering must not be the thing that "adds facts"

This is the crux and the most common way projects like this go wrong.

**Steering moves *topic*, not *truth*.** A vaccination feature makes the model *talk about* vaccines; it does not make the vaccine schedule it states *correct*. The literature is explicit and sobering:

- Steering vectors **do not reliably hill-climb factuality** on capable models — even ITI's truthfulness gains are fragile and shrink on stronger models (emergentmind survey; ITI follow-ups).
- SAE steering is **fragile and unreliable for safety-critical control** — *When the Coffee Feature Activates on Coffins* ([2601.03047](https://hf.co/papers/2601.03047)).
- Steering can have **destructive side effects** — *The Rogue Scalpel: Activation Steering Compromises LLM Safety* ([2509.22067](https://hf.co/papers/2509.22067)) shows random/over-strong feature combinations break alignment. Treat strong steering as a scalpel that can slip.

**The design response — decouple diversity from facts.** We therefore use a two-channel pipeline:

1. **Diversity channel (steered):** steering only sets the *subtopic of the question*. We push the model to *ask about* the under-covered domain.
2. **Factuality channel (grounded + verified, *un*-steered):** the *answer* is generated normally and then checked against the curated seed knowledge base, plus a model-as-judge consistency pass. Questions whose answers can't be grounded are dropped, not "fixed" by more steering.

This mirrors retrieval-grounded factuality work (*Explicit Working Memory* [2412.18069](https://hf.co/papers/2412.18069); document-grounded factuality control [2210.17418](https://hf.co/papers/2210.17418)) and the controllable-generation survey's separation of *attribute control* from *content faithfulness* ([2408.12599](https://hf.co/papers/2408.12599)). The SAE buys us **measurable, targeted diversity**; grounding + verification buys us **factuality**. Neither tries to do the other's job.

---

## 6. Libraries and pre-trained assets (the tooling map)

| Tool | Role | Notes for this project |
|---|---|---|
| **TransformerLens** | Hooked transformer with named activation hooks; the substrate for steering. | `HookedTransformer.from_pretrained("gemma-2-2b-it")`. Hook names like `blocks.{L}.hook_resid_post`. The legacy `from_pretrained` still works; the new **TransformerBridge** path (50+ archs, incl. Qwen3) is the 3.0 direction. ([repo](https://github.com/TransformerLensOrg/TransformerLens)) |
| **SAELens** | Load/train SAEs; `HookedSAETransformer`. | `SAE.from_pretrained(release, sae_id)` returns a ready SAE with `.encode`, `.decode`, `.W_dec`, and `.cfg.metadata.hook_name`. ([docs](https://decoderesearch.github.io/SAELens/dev/)) |
| **Neuronpedia** | Browse 64M+ auto-interpreted features; hosted steering API. | Human-readable labels + dashboards for every Gemma Scope feature; `Gemma-2-2B`, `Gemma-2B-IT`, `GPT2-Small` steerable via API. ([steering docs](https://docs.neuronpedia.org/steering)) |
| **nnsight** | Alternative hooking on raw HF models. | Good for larger models / Qwen path; heavier than TransformerLens for a 2B demo. |

**Pre-trained SAE suites (so we don't train our own):**

- **Gemma Scope** — JumpReLU SAEs for *every* layer/sublayer of Gemma-2-2B/9B (+27B), 16k→1M widths, 400+ SAEs / 30M+ features. The most mature, best-tooled, best-labeled suite. Gemma Scope 2 extends to Gemma 3. ([DeepMind](https://deepmind.google/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/); [paper 2408.05147](https://hf.co/papers/2408.05147)) **← chosen.**
- **Llama Scope** — millions of features for Llama-3.1-8B ([2410.20526](https://arxiv.org/abs/2410.20526)).
- **Qwen-Scope** (Apr–May 2026) — TopK SAEs (k=50, 64k width, residual stream, all layers) for **Qwen3-8B-Base** and Qwen3.5-9B/27B. Raw `.pt` dicts (`W_enc/W_dec/b_enc/b_dec`), **no Qwen3-4B**, **no auto-interp labels**. ([Qwen-Scope collection](https://huggingface.co/collections/Qwen/qwen-scope); [model card](https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50))

---

## 7. Model & SAE decision for this project

The project targets **Apple Silicon (MPS)** and originally ran a **Qwen3-4B** generator. Two facts forced the SAE backbone choice:

1. **No SAE exists for Qwen3-4B.** Qwen-Scope covers 8B and up only. Using it would mean switching to the heavier 8B base model *and* doing manual feature labeling.
2. **Gemma-2-2B + Gemma Scope is the turnkey path:** native SAELens + Neuronpedia integration, pre-labeled features, JumpReLU 16k SAEs at every layer, and it runs comfortably in bf16 on a Mac.

**Decision:** Do the real SAE topic-vector work on **`gemma-2-2b-it`** with **`gemma-scope-2b-pt-res-canonical`** at **layer 12** (16k width). Use the base-trained SAE on the instruct model — standard practice and exactly what Neuronpedia's Gemma-2B-IT steering does (base SAE directions transfer to the IT residual stream; the architecture and basis are identical). The generator was subsequently **unified onto the same `gemma-2-2b-it`** (see the repo root README), so these steering directions apply to it directly — no cross-model transfer needed. A **CAA-style baseline** (`--method caa`) is kept as an SAE-free comparison and a portable recipe.

---

## 8. The right way to run the experiment

**Pipeline (per HPV domain `d`):**

1. **Build probes.** From the seed QA, collect on-topic sentences for `d` and a background set (other domains + generic text).
2. **Find topic features.** Encode probes through the SAE at layer 12; rank features by `mean_on_topic − mean_background`. Keep the top *m* (≈ 8–16). Emit Neuronpedia links for human confirmation.
3. **Form the steering vector.** `v_d = normalize(Σ W_dec[i])` over the kept features (or a single confirmed feature for the cleanest effect).
4. **Generate.** For a neutral elicitation prompt ("Write a patient's question about HPV:"), generate N questions **with** steering at coefficient `c` and N **without** (baseline).
5. **Answer + verify (un-steered).** Generate an answer per question, then run the factuality checker: (a) lexical/semantic grounding against the seed knowledge base, (b) a YES/NO model-as-judge consistency prompt. Drop ungrounded items.

**Metrics — measure both axes, always:**

- **Domain coverage / on-target rate (diversity win):** re-encode each generated question through the SAE and score the domain feature set. Steering "worked" if the steered batch has materially higher in-domain activation and a higher fraction of questions a human/judge labels as domain-`d`. *The SAE that steers is also the classifier that evaluates* — a clean, self-consistent loop (the *Feature Activation Coverage* idea, [2602.10388](https://hf.co/papers/2602.10388)).
- **Lexical diversity:** distinct-1/2 and self-BLEU within and across domains (rising distinct-n, falling cross-domain overlap = real diversification, not paraphrase).
- **Factuality (the guardrail):** pass-rate of the verifier; this must **not** drop materially versus baseline.
- **Fluency:** mean token log-prob / perplexity of generations under the model; the early-warning signal that the coefficient is too high.

**The coefficient sweep is the experiment.** Plot on-target rate and factuality/fluency vs `c ∈ {0, 2, 4, 8, 12, 16}`. The deliverable is the **operating point**: the largest `c` that lifts coverage before factuality or fluency rolls off. Expect a knee; pick just below it.

**Ablations worth running:** layer (8/12/16/20); single-feature vs feature-set steering; additive vs clamp vs directional-ablation-of-overrepresented-features; SAE steering vs CAA baseline; steer-prefix-only vs steer-all-tokens.

---

## 9. Risks, limitations, and honest caveats

- **Steering is fragile** ([2601.03047](https://hf.co/papers/2601.03047)) and can break behavior at high strength ([2509.22067](https://hf.co/papers/2509.22067)) — hence the sweep, the mid-layer choice, and never trusting steering for correctness.
- **Base-SAE-on-IT-model** transfer is well-precedented but imperfect; confirm features fire as expected on the IT model before trusting them.
- **Auto-interp labels can mislead** — always sanity-check a feature's top activations, don't trust the name.
- **This is a medical dataset.** Diversity steering is an *acceptable* use of interpretability tooling; injecting clinical claims via steering is **not**. Every fact must come from grounding + verification, and ideally clinician review before any downstream use. The code treats the seed set as the source of truth and is conservative (drop, don't guess).
- **Generalization:** findings on Gemma-2-2B inform, but do not guarantee, behavior of the Qwen3-4B generator — which is exactly why the CAA baseline exists.

---

## 10. Reference list

**SAEs & interpretability foundations**
- Towards / Scaling Monosemanticity, Anthropic (Golden Gate Claude).
- Scaling and evaluating sparse autoencoders (TopK) — [2406.04093](https://hf.co/papers/2406.04093)
- A Survey on Sparse Autoencoders — [2503.05613](https://hf.co/papers/2503.05613)
- Interpreting LMs Through Concept Descriptions (survey) — [2510.01048](https://hf.co/papers/2510.01048)

**Steering methods**
- Contrastive Activation Addition (CAA) — [2312.06681](https://hf.co/papers/2312.06681)
- Inference-Time Intervention (ITI) — [arxiv 2306.03341](https://arxiv.org/abs/2306.03341)
- Feature-Guided Activation Additions / SAE-TS — [2501.09929](https://hf.co/papers/2501.09929)
- SAE-SSV: Supervised Steering in Sparse Spaces — [2505.16188](https://hf.co/papers/2505.16188)
- Concept Activation Vectors (GCAV) — [2501.05764](https://hf.co/papers/2501.05764)
- Multi-property / Dynamic Activation Composition — [2406.17563](https://hf.co/papers/2406.17563)
- Flexible Steering with Backtracking (FASB) — [2508.17621](https://hf.co/papers/2508.17621)
- LF-Steering (SAE semantic consistency) — [2501.11036](https://hf.co/papers/2501.11036)
- Analyze Feature Flow (cross-layer steering) — [2502.03032](https://hf.co/papers/2502.03032)

**Diversity & synthetic data**
- Less is Enough: Diverse Data in Feature Space (SAE Feature Activation Coverage) — [2602.10388](https://hf.co/papers/2602.10388)
- On the Diversity of Synthetic Data — [2410.15226](https://hf.co/papers/2410.15226)
- MetaSynth — [2504.12563](https://hf.co/papers/2504.12563); CorrSynth — [2411.08553](https://hf.co/papers/2411.08553)

**Factuality**
- Explicit Working Memory — [2412.18069](https://hf.co/papers/2412.18069)
- Controllable Text Generation survey — [2408.12599](https://hf.co/papers/2408.12599)

**Cautions**
- When the Coffee Feature Activates on Coffins — [2601.03047](https://hf.co/papers/2601.03047)
- The Rogue Scalpel — [2509.22067](https://hf.co/papers/2509.22067)

**Tooling & pre-trained SAEs**
- TransformerLens — https://github.com/TransformerLensOrg/TransformerLens
- SAELens — https://decoderesearch.github.io/SAELens/dev/
- Neuronpedia steering — https://docs.neuronpedia.org/steering
- Gemma Scope — [paper 2408.05147](https://hf.co/papers/2408.05147) · [DeepMind](https://deepmind.google/models/gemma/gemma-scope/)
- Llama Scope — [2410.20526](https://arxiv.org/abs/2410.20526)
- Qwen-Scope — [collection](https://huggingface.co/collections/Qwen/qwen-scope) · [model card](https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50)
