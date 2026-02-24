# Versioning Policy

**Effective**: v4.0.0 baseline
**Scope**: All artifacts on `release/v4.0.x` branch

---

## Version Scheme

All version numbers follow **MAJOR.MINOR.PATCH** (semver-compatible).

| Component        | Current | Bumped When                                        |
|------------------|---------|----------------------------------------------------|
| API Version      | 1.0.0   | Breaking schema/endpoint change                    |
| Schema Version   | 1.0.0   | Envelope or certificate schema change              |
| Runtime Version  | 3.1.0   | Internal runtime change affecting outputs          |
| Package Version  | 4.0.x   | Any release from the hardening branch              |

---

## Version Bump Rules

### PATCH (v4.0.0 → v4.0.1)

Allowed changes:
- Bug fixes
- Security patches
- Test additions
- Documentation improvements
- Log message changes
- Performance improvements with no output change
- Error message clarifications

NOT allowed:
- New endpoints
- New request/response fields
- Schema changes
- New error codes
- Behavioral changes to existing endpoints

### MINOR (pre-release beta tags: v4.1.0-beta.1)

Used for pilot candidate releases.  Allowed additional changes:
- Additive response fields (new optional fields only)
- New optional request parameters with defaults matching current behavior
- New error codes (existing codes unchanged)
- New health/introspection endpoints

NOT allowed:
- Removal of any field or endpoint
- Type changes to existing fields
- Semantic changes to existing behavior

### MAJOR (v5.0.0)

Reserved for breaking changes after alpha.  Not used during this cycle.

---

## Tagging Convention

| Tag Format          | Purpose                                    | Example        |
|---------------------|--------------------------------------------|----------------|
| `v4.0.X`            | Hardening patch                            | `v4.0.1`       |
| `v4.X.0-beta.N`     | Pilot candidate                            | `v4.1.0-beta.1`|
| `v4.X.0-alpha.N`    | Internal testing                           | `v4.0.1-alpha.1`|

---

## Branch Policy

| Branch              | Purpose                      | Merge Rules                         |
|---------------------|------------------------------|-------------------------------------|
| `main`              | Development trunk            | Feature work, experiments           |
| `release/v4.0.x`    | Hardening branch (this)      | Patches only; launch gate required  |

### Release Branch Rules

1. All changes to `release/v4.0.x` must pass the launch gate checklist
   defined in `LAUNCH_READINESS.md`.
2. No direct edits — all work happens in topic branches merged via
   reviewed commits.
3. Every merge to `release/v4.0.x` gets a patch tag.
4. The pilot candidate is tagged from this branch.

---

## Compatibility Guarantees

### Within a PATCH version

- Request schemas: identical
- Response schemas: identical
- Error codes: identical
- Envelope format: identical
- Certificate format: identical
- HTTP status codes: identical

### Within a MINOR version

- All guarantees of PATCH, plus:
- New optional response fields may appear (additive only)
- Clients MUST ignore unknown fields
- Existing fields never change type or semantics

### API Version Header

Every response includes version metadata:

```json
{
  "api_version": "1.0.0",
  "schema_version": "1.0.0",
  "runtime_version": "3.1.0"
}
```

Clients should check `schema_version` to detect contract changes.

---

## Pre-Release Disclosure

During private alpha:
- API may have unannounced downtime
- No uptime SLA
- Data retention not guaranteed beyond session
- Version bumps may be more frequent
- Alpha users are notified of breaking changes 48h in advance
