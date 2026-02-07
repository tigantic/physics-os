# CLI Contract: `qtenet`

This document defines a stable, enterprise-grade CLI contract. Commands may remain thin wrappers initially.

## Principles
- Machine-readable outputs by default (`--json`).
- Deterministic exit codes.
- Explicit escape hatches for densification.

## Commands

### `qtenet inventory`
Print the location of the repo index inventory.

- Output: path string
- Exit codes: 0 success

### `qtenet doctor`
Validate environment + capability wiring (enterprise support command).

- Checks: torch import, CUDA visibility, inventory presence
- Output: human summary + optional `--json` payload
- Exit codes: 0 ok, 2 degraded, 3 failed

### `qtenet inspect <artifact>`
Inspect a QTT artifact/container.

- `--json` emits metadata.

### `qtenet compress ...`
Facade entrypoint for compression.

Notes:
- The_Compressor remains a separate product; this command may delegate to it or to other codecs.

### `qtenet query <artifact> <index>`
Point query.

### `qtenet reconstruct <artifact>`
Dense escape hatch.

- MUST require explicit confirmation flag: `--allow-dense`.

### `qtenet benchmark`
Run benchmark suites.

## Output schema (`--json`)
All commands that emit JSON should follow:

```json
{
  "tool": "qtenet",
  "command": "query",
  "status": "ok",
  "meta": {},
  "result": {}
}
```
