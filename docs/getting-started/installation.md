# Installation

## Python Package

```bash
# Core engine only
pip install tensornet

# With domain packs
pip install tensornet[cfd]
pip install tensornet[quantum]
pip install tensornet[all]
```

## Development Setup

```bash
git clone https://github.com/tigantic/HyperTensor-VM.git
cd HyperTensor-VM

# Install with dev dependencies
pip install -e ".[dev,docs]"

# Install Rust workspace
cargo build --workspace

# Run quality gates
make check
```

## Rust Crates

The Rust performance substrate builds as a Cargo workspace:

```bash
cargo build --workspace --release
cargo test --workspace
cargo clippy --workspace
```

## Verify Installation

```python
import ontic
print(tensornet.__version__)
```
