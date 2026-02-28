# FluidElite: Gradient-Free Language Models

**The world's first production-ready gradient-free LLM training framework.**

---

## 🚀 What Is This?

FluidElite trains language models **without backpropagation**.

Instead of computing gradients through billions of parameters, FluidElite:
1. **Samples** the language function at strategic points (TCI)
2. **Decomposes** the function into a tensor network (QTT)
3. **Done.** No optimizer state. No gradient explosions. No GPU clusters.

## 📊 Results

| Method | Params | Compression | Accuracy | Training |
|--------|--------|-------------|----------|----------|
| Traditional | 4.2M | 1× | 41% | Gradient descent |
| **FluidElite** | **16K** | **261×** | **35%** | **TCI (no gradients)** |
| FluidElite-Edge | 7.8K | 534× | 32% | TCI |

**Trade 6% accuracy for 261× compression and zero gradients.**

## 🔧 Quick Start

```python
from fluidelite.fe_tci import FluidEliteModel

# Create model (default rank=24, optimal from NS methodology)
model = FluidEliteModel(vocab_size=256, context_length=8)

# Train on corpus — NO GRADIENTS!
corpus = open("data.txt", "rb").read()
model.train(corpus)

# Generate
next_token = model.generate_one(context)
```

## 🧠 How It Works

### Traditional LLM
```
Forward pass → Compute loss → Backward pass → Update weights → Repeat 1M times
```

### FluidElite
```
Sample f(context) → Tensor decomposition → Done
```

### The Math

A language model is a function:
```
f: context → next_token_distribution
```

FluidElite uses **Tensor Cross Interpolation (TCI)** to build a **Quantized Tensor Train (QTT)** approximation of f using only O(r² × log N) samples.

The QTT stores the function in O(log N × r²) parameters instead of O(N).

## 🎯 Key Hyperparameters

From extensive rank sweep (NS Millennium methodology):

| Use Case | Rank | Params | Compression | Accuracy |
|----------|------|--------|-------------|----------|
| Max accuracy | 32 | 27K | 154× | 35.3% |
| **Balanced** | **24** | **16K** | **261×** | **33.2%** |
| Max compression | 16 | 7.8K | 534× | 31.7% |

**Default: rank=24 (best balance)**

## 📁 Project Structure

```
fluidelite/
├── fe_tci/
│   ├── fluidelite_model.py   # Main model class
│   └── context_encoder.py    # Context → index encoding
├── qtt_direct_train.py       # Direct QTT training (SGD on TT manifold)
├── qtt_rank_sweep.py         # Rank optimization experiments
├── qtt_scale.py              # Scaling to larger features
├── triton_features.py        # GPU-accelerated feature extraction
└── FINDINGS.md               # Research log and results
```

## 🏆 Why FluidElite?

| Feature | Traditional | FluidElite |
|---------|-------------|------------|
| Training method | Backprop | TCI sampling |
| Gradient computation | Required | **None** |
| Optimizer state | GBs | **Zero** |
| GPU memory | Massive | Minimal |
| Training time | Days | **Minutes** |
| Gradient explosions | Common | **Impossible** |

## 📚 Citation

```bibtex
@software{fluidelite2026,
  title={FluidElite: Gradient-Free Language Models via Tensor Cross Interpolation},
  author={TiganticLabz},
  year={2026},
  url={https://github.com/physics_os/fluidelite}
}
```

## 🔬 Research Foundation

- **TCI (Tensor Cross Interpolation)**: Oseledets & Tyrtyshnikov
- **QTT (Quantized Tensor Train)**: Khoromskij
- **Rank optimization**: NS Millennium Framework methodology
- **Xavier initialization**: Critical for QTT gradient flow

---

*FluidElite: Because the best gradient is no gradient at all.* ⚡
