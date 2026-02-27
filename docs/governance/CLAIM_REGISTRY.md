# Claim Registry

**Baseline**: v4.0.0
**Source**: `hypertensor/core/evidence.py` — `generate_claims()`
**Status**: FROZEN — new tags require MINOR version bump

---

## Purpose

Every trust certificate contains a `claims` array.  Each claim has a
`tag` field that identifies the type of assertion.  This registry is
the **exhaustive allowlist** of valid claim tags.

Any claim tag NOT in this registry is a bug.  Certificate verifiers
MAY reject certificates containing unregistered tags.

---

## Registered Claim Tags

### CONSERVATION

| Property          | Value                                                         |
|-------------------|---------------------------------------------------------------|
| **Tag**           | `CONSERVATION`                                                |
| **Assertion**     | A named physical quantity is preserved across the simulation   |
| **Witness fields**| `initial`, `final`, `relative_error`, `threshold`             |
| **Satisfied when**| `relative_error < threshold` (default threshold: `1e-4`)      |
| **Domains**       | All domains with invariant quantities                         |

**Witness schema:**

```json
{
  "initial": 1.0,
  "final": 0.9999999877,
  "relative_error": 1.23e-08,
  "threshold": 0.0001
}
```

**Examples by domain:**

| Domain             | Quantity Conserved        |
|--------------------|---------------------------|
| `burgers`          | `L2_norm`                 |
| `maxwell`          | `electromagnetic_energy`  |
| `maxwell_3d`       | `electromagnetic_energy`  |
| `schrodinger`      | `norm`                    |
| `advection_diffusion` | `total_mass`           |
| `vlasov_poisson`   | `particle_count`          |
| `navier_stokes_2d` | `enstrophy`               |

---

### STABILITY

| Property          | Value                                                         |
|-------------------|---------------------------------------------------------------|
| **Tag**           | `STABILITY`                                                   |
| **Assertion**     | The simulation completed without numerical divergence          |
| **Witness fields**| `wall_time_s`, `time_steps`, `completed`                      |
| **Satisfied when**| `completed == true` (wall_time > 0 and no divergence)         |
| **Domains**       | All                                                           |

**Witness schema:**

```json
{
  "wall_time_s": 0.3421,
  "time_steps": 100,
  "completed": true
}
```

---

### BOUND

| Property          | Value                                                         |
|-------------------|---------------------------------------------------------------|
| **Tag**           | `BOUND`                                                       |
| **Assertion**     | All field values remain within physical bounds                 |
| **Witness fields**| `max_absolute_value`, `threshold`                             |
| **Satisfied when**| `max_absolute_value < threshold` (default: `1e15`)            |
| **Domains**       | All domains (when fields are returned)                        |

**Witness schema:**

```json
{
  "max_absolute_value": 1.234567,
  "threshold": 1e15
}
```

---

## Tag Rules

1. **Allowlist enforcement**: Only tags listed above may appear in certificates
2. **No new tags in PATCH**: New claim types require at minimum a MINOR version bump
3. **Tag format**: UPPER_SNAKE_CASE, maximum 32 characters
4. **Witness schema**: Every tag has a fixed witness schema.  No optional or extra fields
5. **Boolean `satisfied`**: Every claim has a `satisfied` boolean computed from its witness
6. **Deterministic evaluation**: Given identical witness data, `satisfied` must always
   produce the same result

---

## Future Tags (Reserved, Not Implemented)

These tags are reserved for future MINOR versions.  They MUST NOT
appear in v4.0.x certificates.

| Tag                  | Intended Use                                      |
|----------------------|---------------------------------------------------|
| `CONVERGENCE`        | Grid convergence (result stable under refinement) |
| `REPRODUCIBILITY`    | Deterministic replay (same input → same hash)     |
| `ENERGY_BOUND`       | Energy remains below a specific physical threshold |
| `CFL_SATISFIED`      | CFL condition met throughout simulation           |

---

## Verification Rules for Clients

When verifying a certificate, clients SHOULD:

1. Check that `certificate_version` is `"1.0.0"`
2. Verify the signature using the server's public key
3. Check that all `claims[].tag` values are in this registry
4. Check that all `claims[].satisfied` are `true`
5. Optionally: re-compute `satisfied` from the witness data to confirm server's evaluation
