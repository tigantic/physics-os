# FluidVision: Roadmap to Infinite-Context Language Modeling

**Document Status:** ACTIVE  
**Created:** January 12, 2026  
**Predecessor:** [FluidElite.md](FluidElite.md) (Implementation Complete ✅)

---

## CONSTITUTION OF FLUIDELITE DEVELOPMENT

### Preamble

We, the builders of FluidElite, establish this Constitution to guide the development of a fundamentally new approach to language modeling—one that treats context not as a finite window but as a compressed, infinite stream. These Articles are binding.

---

### Article I — The Core Thesis

**I.1** FluidElite posits that language modeling does not require explicit storage of all past tokens.

**I.2** A bounded-rank Matrix Product State (MPS) can serve as a sufficient compressed representation of unbounded context.

**I.3** The trade-off between compression ratio (χ) and information retention is the fundamental lever of the architecture.

**I.4** This thesis must be validated empirically, not assumed.

---

### Article II — Scientific Rigor

**II.1** Every claim must have a corresponding benchmark.

**II.2** Comparisons to baselines (Transformer, Mamba, RWKV) must be fair:
  - II.2.1 — Same parameter count (±10%)
  - II.2.2 — Same training data
  - II.2.3 — Same compute budget

**II.3** Negative results are results. Document what doesn't work.

**II.4** Perplexity is necessary but not sufficient. Downstream tasks validate real-world utility.

---

### Article III — Engineering Standards

**III.1** All code must pass tests before merge. Minimum 80% coverage on core modules.

**III.2** Performance claims require profiling evidence (memory, throughput, latency).

**III.3** CUDA/Triton kernels must have reference Python implementations for verification.

**III.4** Numerical stability is non-negotiable. SVD failures must be handled gracefully.

**III.5** Reproducibility: all experiments must log seeds, hyperparameters, and git commits.

---

### Article IV — Intellectual Honesty

**IV.1** Acknowledge prior art. Cite:
  - Tensor networks in ML (Stoudenmire, Schwab, et al.)
  - State space models (Mamba, S4)
  - Linear attention variants (RWKV, RetNet)

**IV.2** State limitations clearly. FluidElite forgets—this is a feature and a bug.

**IV.3** Do not overclaim. "Infinite context" means unbounded, not omniscient.

---

### Article V — Publication Standards

**V.1** Preprint before press. Arxiv precedes any public announcement.

**V.2** Code accompanies paper. No "code available upon request."

**V.3** Ablations are mandatory. Show what each component contributes.

**V.4** Failure modes section required in all publications.

---

### Article VI — Definition of Success

**VI.1** Phase success requires ALL checkboxes marked complete.

**VI.2** A phase is not complete until results are documented.

**VI.3** "Working" means passes tests AND produces expected behavior on real data.

**VI.4** The ultimate success criterion:

> FluidElite achieves ≤1.2× perplexity of a same-parameter Transformer baseline while processing 10× longer sequences with bounded memory.

---

### Article VII — Governance

**VII.1** This Constitution may be amended with documented rationale.

**VII.2** Breaking changes require deprecation period or major version bump.

**VII.3** All decisions affecting architecture must be documented in ADR format.

---

## THE VISION

### What We're Building

A language model architecture where:

```
Context Window = ∞
Memory Usage = O(L × χ²)  [χ bounded]
Per-Token Cost = O(χ³)    [independent of history length]
```

### Why It Matters

| Current Paradigm | FluidElite Paradigm |
|------------------|---------------------|
| Context is a buffer | Context is a state |
| Forget by truncation | Forget by compression |
| Quadratic attention | Linear contraction |
| Fixed window (4K-128K) | Unbounded stream |

### The Bet

We bet that the low singular values discarded during MPS truncation correspond to **irrelevant details**, not **semantic content**. If true, FluidElite can:

1. Process infinite documents
2. Run on edge devices
3. Enable real-time streaming applications
4. Fundamentally change how we think about context

---

## PHASED EXECUTION ROADMAP

### Phase 0: Foundation (COMPLETE ✅)

**Deliverable:** Working FluidElite prototype with tests

| Task | Status | Artifact |
|------|--------|----------|
| MPS implementation | ✅ | `fluidelite/core/mps.py` |
| MPO implementation | ✅ | `fluidelite/core/mpo.py` |
| Decompositions (SafeSVD) | ✅ | `fluidelite/core/decompositions.py` |
| FluidElite model | ✅ | `fluidelite/llm/fluid_elite.py` |
| RiemannianAdam optimizer | ✅ | `fluidelite/optim/riemannian.py` |
| Training scripts | ✅ | `fluidelite/scripts/` |
| Test suite (92 tests, 91%) | ✅ | `fluidelite/tests/` |

---

### Phase 1: Empirical Validation (Weeks 1-4)

**Goal:** Answer "Does this work at all on real language?"

#### 1.1 Perplexity Benchmarks
| Task | Status | File |
|------|--------|------|
| ☐ WikiText-2 evaluation loop | Not Started | `benchmarks/wikitext.py` |
| ☐ WikiText-103 full training | Not Started | `benchmarks/wikitext103_train.py` |
| ☐ Perplexity vs χ sweep | Not Started | `experiments/chi_sweep.py` |
| ☐ Perplexity vs L sweep | Not Started | `experiments/length_sweep.py` |
| ☐ Comparison table vs GPT-2 | Not Started | `results/perplexity.md` |

**Success Criterion:** PPL ≤ 1.5× GPT-2 Small at L=1024

#### 1.2 Memory Profiling
| Task | Status | File |
|------|--------|------|
| ☐ Peak VRAM measurement | Not Started | `benchmarks/memory_profile.py` |
| ☐ Memory vs L curve | Not Started | `experiments/memory_scaling.py` |
| ☐ Memory vs χ curve | Not Started | `experiments/chi_memory.py` |
| ☐ Comparison plot vs Transformer | Not Started | `results/memory_curves.png` |

**Success Criterion:** Constant memory at L > 8192

#### 1.3 Throughput Analysis
| Task | Status | File |
|------|--------|------|
| ☐ Tokens/sec vs L | Not Started | `benchmarks/throughput.py` |
| ☐ Tokens/sec vs χ | Not Started | `benchmarks/throughput_chi.py` |
| ☐ Latency per token histogram | Not Started | `results/latency.png` |
| ☐ Crossover point calculation | Not Started | `results/crossover_analysis.md` |

**Success Criterion:** Faster than Transformer at L > 4096

---

### Phase 2: Kernel Optimization (Weeks 5-8)

**Goal:** Make it fast enough to train at scale

#### 2.1 Triton Kernels
| Task | Status | File |
|------|--------|------|
| ☐ Fused MPO-MPS contraction | Not Started | `kernels/triton/mpo_apply.py` |
| ☐ Batched SVD wrapper | Not Started | `kernels/triton/batched_svd.py` |
| ☐ Fused truncation kernel | Not Started | `kernels/triton/truncate.py` |
| ☐ Benchmark vs PyTorch baseline | Not Started | `benchmarks/kernel_speedup.py` |

**Success Criterion:** ≥3× speedup over pure PyTorch

#### 2.2 CUDA Kernels (Optional)
| Task | Status | File |
|------|--------|------|
| ☐ cuSOLVER batched SVD integration | Not Started | `kernels/cuda/svd.cu` |
| ☐ Tensor contraction kernel | Not Started | `kernels/cuda/contract.cu` |
| ☐ Python bindings | Not Started | `kernels/cuda/bindings.cpp` |

**Success Criterion:** ≥5× speedup for SVD-heavy operations

#### 2.3 Mixed Precision
| Task | Status | File |
|------|--------|------|
| ☐ FP16 MPS tensors | Not Started | `core/mps_fp16.py` |
| ☐ FP32 accumulation for SVD | Not Started | `core/decompositions_mixed.py` |
| ☐ Numerical stability tests | Not Started | `tests/test_mixed_precision.py` |
| ☐ Memory savings measurement | Not Started | `results/fp16_memory.md` |

**Success Criterion:** 2× memory reduction, <1% PPL degradation

---

### Phase 3: Architecture Research (Weeks 9-12)

**Goal:** Explore the design space

#### 3.1 Rank Dynamics
| Task | Status | File |
|------|--------|------|
| ☐ Adaptive χ based on entropy | Not Started | `research/adaptive_rank.py` |
| ☐ Per-site rank variation | Not Started | `research/variable_rank.py` |
| ☐ Rank growth analysis | Not Started | `experiments/rank_dynamics.py` |

#### 3.2 Structural Variants
| Task | Status | File |
|------|--------|------|
| ☐ Hierarchical MPS (MERA-inspired) | Not Started | `research/hierarchical_mps.py` |
| ☐ Bidirectional MPS | Not Started | `research/bidirectional.py` |
| ☐ Local attention + MPS hybrid | Not Started | `research/hybrid_attention.py` |
| ☐ Tree tensor network context | Not Started | `research/ttn_context.py` |

#### 3.3 Training Dynamics
| Task | Status | File |
|------|--------|------|
| ☐ Gradient flow through truncation | Not Started | `research/truncation_grad.py` |
| ☐ Straight-through estimator | Not Started | `research/ste_truncation.py` |
| ☐ Riemannian optimization variants | Not Started | `research/riemannian_variants.py` |

---

### Phase 4: Downstream Evaluation (Weeks 13-16)

**Goal:** Prove it works on real tasks

#### 4.1 Long-Range Benchmarks
| Task | Status | File |
|------|--------|------|
| ☐ LAMBADA (last word prediction) | Not Started | `eval/lambada.py` |
| ☐ Passkey retrieval (10K, 50K, 100K) | Not Started | `eval/passkey.py` |
| ☐ PG-19 book completion | Not Started | `eval/pg19.py` |
| ☐ Scrolls summarization | Not Started | `eval/scrolls.py` |

**Success Criterion:** Non-trivial performance on 100K context passkey

#### 4.2 Standard Benchmarks
| Task | Status | File |
|------|--------|------|
| ☐ HellaSwag | Not Started | `eval/hellaswag.py` |
| ☐ PIQA | Not Started | `eval/piqa.py` |
| ☐ ARC-Easy | Not Started | `eval/arc.py` |
| ☐ lm-eval-harness integration | Not Started | `eval/harness_config.yaml` |

#### 4.3 Streaming Applications
| Task | Status | File |
|------|--------|------|
| ☐ Live transcription demo | Not Started | `demos/streaming_transcribe.py` |
| ☐ Infinite conversation | Not Started | `demos/infinite_chat.py` |
| ☐ Log analysis pipeline | Not Started | `demos/log_analyzer.py` |

---

### Phase 5: Publication & Release (Weeks 17-20)

**Goal:** Share with the world

#### 5.1 Paper
| Task | Status | File |
|------|--------|------|
| ☐ Paper outline | Not Started | `paper/outline.md` |
| ☐ Introduction + motivation | Not Started | `paper/sections/intro.tex` |
| ☐ Method section | Not Started | `paper/sections/method.tex` |
| ☐ Experiments section | Not Started | `paper/sections/experiments.tex` |
| ☐ Related work | Not Started | `paper/sections/related.tex` |
| ☐ Conclusion + limitations | Not Started | `paper/sections/conclusion.tex` |
| ☐ Arxiv submission | Not Started | `paper/arxiv/` |

**Target Venue:** NeurIPS 2026 / ICML 2027

#### 5.2 Open Source Release
| Task | Status | File |
|------|--------|------|
| ☐ Clean API documentation | Not Started | `docs/api.md` |
| ☐ Tutorial notebooks | Not Started | `notebooks/tutorial.ipynb` |
| ☐ pip package setup | Not Started | `setup.py`, `pyproject.toml` |
| ☐ HuggingFace model cards | Not Started | `hf/model_card.md` |
| ☐ Pretrained checkpoints | Not Started | `checkpoints/` |

#### 5.3 Community
| Task | Status | File |
|------|--------|------|
| ☐ README with quickstart | Not Started | `README.md` |
| ☐ Contributing guide | Not Started | `CONTRIBUTING.md` |
| ☐ Discord/GitHub discussions | Not Started | — |
| ☐ Blog post announcement | Not Started | `blog/announcement.md` |

---

## MILESTONES

| Milestone | Target Date | Gate Criterion |
|-----------|-------------|----------------|
| **M1: First PPL Number** | Week 2 | WikiText-2 PPL computed |
| **M2: Scaling Validated** | Week 4 | Memory/throughput curves complete |
| **M3: Kernels Shipped** | Week 8 | 3× speedup achieved |
| **M4: Architecture Locked** | Week 12 | Best variant selected |
| **M5: Benchmarks Complete** | Week 16 | All eval tasks run |
| **M6: Arxiv Submitted** | Week 18 | Paper public |
| **M7: v1.0 Released** | Week 20 | pip installable, checkpoints available |

---

## RISK REGISTER

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| MPS truncation loses critical information | Medium | Critical | Ablate χ, compare retrieval tasks |
| Training unstable at scale | Medium | High | Gradient clipping, learning rate warmup |
| Kernels don't provide expected speedup | Low | Medium | Profile first, optimize bottlenecks |
| Perplexity uncompetitive | Medium | Critical | Hybrid architectures as fallback |
| Reviewer skepticism (novelty) | High | Medium | Strong baselines, clear ablations |

---

## RESOURCE REQUIREMENTS

| Resource | Specification | Purpose |
|----------|---------------|---------|
| **GPU Compute** | 8× A100 80GB (or equivalent) | Full-scale training |
| **Local Dev** | RTX 5070 8GB | Prototyping, small experiments |
| **Storage** | 1TB SSD | Datasets, checkpoints |
| **Time** | 20 weeks (1 FTE) | Full roadmap execution |

---

## DECISION LOG

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-12 | Use isolated `fluidelite/` module | Clean separation, easy removal |
| 2026-01-12 | Select RiemannianAdam without QR | Simpler, QR can be added later |
| 2026-01-12 | Target NeurIPS 2026 | Reasonable timeline for validation |

---

## APPENDIX A: Key Equations

### MPS State Update
$$|\psi_{t+1}\rangle = \hat{W}_{x_t} |\psi_t\rangle$$

Where $\hat{W}_{x_t}$ is the MPO slice for token $x_t$.

### Truncation
After each update:
$$|\psi\rangle \approx \sum_{i=1}^{\chi} \sigma_i |L_i\rangle \otimes |R_i\rangle$$

Keep top $\chi$ singular values.

### Perplexity
$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{t=1}^{N} \log p(x_t | x_{<t})\right)$$

---

## APPENDIX B: Reference Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FluidElite Model                        │
├─────────────────────────────────────────────────────────────┤
│  Input: token x_t                                           │
│     ↓                                                       │
│  ┌─────────────┐                                            │
│  │ Embedding   │  vocab → d (physical dimension)            │
│  └─────────────┘                                            │
│     ↓                                                       │
│  ┌─────────────┐     ┌─────────────┐                        │
│  │ EliteLinear │ ←── │   MPO W     │  d_in → d_out          │
│  └─────────────┘     └─────────────┘                        │
│     ↓                                                       │
│  ┌─────────────┐                                            │
│  │  GELU Act   │  (via cross approximation)                 │
│  └─────────────┘                                            │
│     ↓                                                       │
│  ┌─────────────┐     ┌─────────────┐                        │
│  │ EliteLinear │ ←── │   MPO W     │  d → d                 │
│  └─────────────┘     └─────────────┘                        │
│     ↓                                                       │
│  ┌─────────────┐                                            │
│  │   Output    │  MPS → logits (vocab)                      │
│  └─────────────┘                                            │
│     ↓                                                       │
│  Output: p(x_{t+1} | context)                               │
└─────────────────────────────────────────────────────────────┘
```

---

**End of Document**

*"The context is not a window. The context is a wave."*
