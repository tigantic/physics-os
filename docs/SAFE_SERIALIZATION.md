# Safe Serialization Patterns

This document describes safe serialization practices for HyperTensor to prevent deserialization vulnerabilities.

---

## ⚠️ Security Background

Pickle deserialization (`pickle.load()`, `np.load(allow_pickle=True)`) is **unsafe** because:
- Pickle can execute arbitrary code during deserialization
- A malicious pickle file can compromise the entire system
- This is a well-known attack vector (CWE-502: Deserialization of Untrusted Data)

---

## ✅ Approved Patterns

### 1. JSON for Configuration and State

Use JSON for configuration, metadata, and simple state:

```python
import json

# GOOD: JSON for configuration
def save_config(config: dict, path: Path):
    with open(path, "w") as f:
        json.dump(config, f, indent=2)

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)
```

### 2. PyTorch save/load for Tensors and Models

Use PyTorch's save/load with `weights_only=True`:

```python
import torch

# GOOD: PyTorch save/load for tensors
def save_model(model: nn.Module, path: Path):
    torch.save(model.state_dict(), path)

def load_model(model: nn.Module, path: Path):
    # SAFE: weights_only=True prevents arbitrary code execution
    state = torch.load(path, weights_only=True)
    model.load_state_dict(state)
```

### 3. NumPy .npz for Arrays (without pickle)

Use NumPy's `.npz` format with `allow_pickle=False`:

```python
import numpy as np

# GOOD: NumPy without pickle
def save_arrays(path: Path, **arrays):
    np.savez_compressed(path, **arrays)

def load_arrays(path: Path) -> dict:
    # SAFE: allow_pickle=False by default
    data = np.load(path, allow_pickle=False)
    return dict(data)
```

### 4. Dataclass Serialization

Use dataclasses with JSON for structured state:

```python
from dataclasses import dataclass, asdict
import json

@dataclass
class TrainingState:
    epoch: int
    loss: float
    learning_rate: float
    
    def to_json(self, path: Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f)
    
    @classmethod
    def from_json(cls, path: Path) -> "TrainingState":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
```

---

## ❌ Forbidden Patterns

### Never use `pickle.load()` on untrusted data

```python
# BAD: Arbitrary code execution risk
import pickle
with open(path, "rb") as f:
    data = pickle.load(f)  # VULNERABLE
```

### Never use `allow_pickle=True` with untrusted data

```python
# BAD: NumPy pickle allows arbitrary code
import numpy as np
data = np.load(path, allow_pickle=True)  # VULNERABLE
```

### Never use `weights_only=False` with untrusted data

```python
# BAD: PyTorch without weights_only allows code execution
import torch
model = torch.load(path)  # VULNERABLE (weights_only defaults to False in older versions)
```

---

## 🔄 Migration Guide

### From pickle.load to JSON

Before:
```python
import pickle
with open("state.pkl", "rb") as f:
    state = pickle.load(f)
```

After:
```python
import json
with open("state.json", "r") as f:
    state = json.load(f)
```

### From pickle to torch.save (for tensors)

Before:
```python
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model.state_dict(), f)
```

After:
```python
import torch
torch.save(model.state_dict(), "model.pt")
# On load:
state = torch.load("model.pt", weights_only=True)
```

### From np.load(allow_pickle=True) to .npz

Before:
```python
data = np.load("data.npy", allow_pickle=True)
```

After:
```python
data = np.load("data.npz", allow_pickle=False)
arrays = {k: data[k] for k in data.files}
```

---

## 🔍 Linting Rules

Ruff is configured to flag unsafe serialization:

- `S301`: pickle usage (flake8-bandit)
- `S302`: marshal usage
- `S506`: unsafe YAML load

Check configuration in `pyproject.toml`:
```toml
[tool.ruff.lint]
select = ["S"]  # flake8-bandit security rules
```

---

## 📋 Checklist for New Serialization Code

- [ ] Is pickle avoided?
- [ ] Is `allow_pickle=False` used for NumPy?
- [ ] Is `weights_only=True` used for PyTorch?
- [ ] Is input validated after loading?
- [ ] Are file paths validated (no path traversal)?
- [ ] Is the serialization format documented?

---

## See Also

- [OWASP: Deserialization Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Deserialization_Cheat_Sheet.html)
- [CWE-502: Deserialization of Untrusted Data](https://cwe.mitre.org/data/definitions/502.html)
- [NumPy Security](https://numpy.org/doc/stable/reference/generated/numpy.load.html)
