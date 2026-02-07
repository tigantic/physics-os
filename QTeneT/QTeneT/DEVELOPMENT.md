# Development

This environment may enforce **PEP 668** (externally-managed Python), which blocks `pip install` into the system interpreter.

## Recommended
Use a virtual environment:

```bash
cd QTeneT
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
pytest -q
```

## Alternative (not recommended)
Override system guard:

```bash
pip install -e '.[dev]' --break-system-packages
```
