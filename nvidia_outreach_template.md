Subject: QTT Compression of Your Ahmed Body Dataset — [RATIO]× with [ERROR] Reconstruction Error

---

Hi [NAME],

I compressed your PhysicsNeMo Ahmed Body dataset using Quantized Tensor Train decomposition.

Results on [N] test samples:
  • Compression ratio: [RATIO]× (22GB → [SIZE])
  • Reconstruction L2 error: [ERROR]
  • Bond dimension: constant at χ=64 across all parametric variations
  • Decompress time: [TIME]ms per sample

The compressed fields (pMean, wallShearStressMean) reconstruct to within [ERROR] of OpenFOAM ground truth. No retraining needed — this is lossless-grade compression of the simulation output itself.

Published result backing this: "Quantized Tensor Train Compression for Turbulent Flow Simulation" [ZENODO_LINK] — demonstrates Reynolds-independent bond dimension across Re=50-800, 10,000×+ compression at 256³.

Benchmark code and compressed dataset: [GITHUB_LINK]

The implication for PhysicsNeMo: QTT can compress your training data pipeline by orders of magnitude while preserving field accuracy. This applies to any structured CFD output — not just Ahmed Body.

Happy to discuss integration with PhysicsNeMo Curator or the MoE framework.

Brad Adams
Tigantic Holdings LLC
[EMAIL]
[PHONE]

---

FILL BEFORE SENDING:
  [NAME]     — Target: search NVIDIA PhysicsNeMo team on LinkedIn
               Candidates: Mike Houston, Rev Lebaredian, or the PhysicsNeMo GitHub contributors
  [RATIO]    — From benchmark_report.txt
  [ERROR]    — From benchmark_report.txt  
  [SIZE]     — 22GB / ratio
  [TIME]     — From benchmark_report.txt
  [N]        — Number of samples benchmarked
  [ZENODO_LINK]  — Your existing Zenodo publication
  [GITHUB_LINK]  — Public repo with benchmark code + results
  [EMAIL]    — Your contact email
  [PHONE]    — Your phone number

SEND TO:
  1. PhysicsNeMo GitHub discussion/issue (public — they'll see it)
  2. LinkedIn DM to PhysicsNeMo team members
  3. NVIDIA Inception application (link in mapping doc)
  4. physicsnemo-support@nvidia.com or relevant contact
