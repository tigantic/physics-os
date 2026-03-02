# Claim Registry

**Baseline**: v4.0.0
**Source**: `physics_os/core/evidence.py` — `generate_claims()`
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

*(All previously reserved tags have been promoted to registered status
in v4.1.0 — see sections below.)*

---

### CONVERGENCE

| Property          | Value                                                         |
|-------------------|---------------------------------------------------------------|
| **Tag**           | `CONVERGENCE`                                                 |
| **Assertion**     | Grid convergence is verified (result stable under refinement) |
| **Witness fields**| `qoi`, `observed_order`, `required_order`, `refinement_levels`|
| **Satisfied when**| `observed_order >= required_order`                            |
| **Domains**       | All domains with convergence studies                          |

**Witness schema:**

```json
{
  "qoi": "L2_error_dudx",
  "observed_order": 2.13,
  "required_order": 2.0,
  "refinement_levels": 3
}
```

---

### REPRODUCIBILITY

| Property          | Value                                                         |
|-------------------|---------------------------------------------------------------|
| **Tag**           | `REPRODUCIBILITY`                                             |
| **Assertion**     | Deterministic replay — same input produces same config hash   |
| **Witness fields**| `determinism_tier`, `config_hash`, `seed`                     |
| **Satisfied when**| `config_hash` is non-empty                                    |
| **Domains**       | All                                                           |

**Witness schema:**

```json
{
  "determinism_tier": "reproducible",
  "config_hash": "a1b2c3d4e5f6...",
  "seed": 42
}
```

---

### ENERGY_BOUND

| Property          | Value                                                         |
|-------------------|---------------------------------------------------------------|
| **Tag**           | `ENERGY_BOUND`                                                |
| **Assertion**     | Energy remains below a specific physical threshold            |
| **Witness fields**| `quantity`, `value`, `threshold`                              |
| **Satisfied when**| `abs(value) < threshold`                                      |
| **Domains**       | Domains with energy conservation (NS 2D, Maxwell, etc.)       |

**Witness schema:**

```json
{
  "quantity": "kinetic_energy",
  "value": 1.234567e+02,
  "threshold": 1e+15
}
```

---

### CFL_SATISFIED

| Property          | Value                                                         |
|-------------------|---------------------------------------------------------------|
| **Tag**           | `CFL_SATISFIED`                                               |
| **Assertion**     | CFL condition met throughout the simulation                   |
| **Witness fields**| `max_cfl`, `cfl_limit`                                        |
| **Satisfied when**| `max_cfl <= cfl_limit`                                        |
| **Domains**       | All time-dependent domains                                    |

**Witness schema:**

```json
{
  "max_cfl": 0.4512,
  "cfl_limit": 1.0
}
```

---

### BOUNDEDNESS

| Property          | Value                                                         |
|-------------------|---------------------------------------------------------------|
| **Tag**           | `BOUNDEDNESS`                                                 |
| **Assertion**     | Physical boundedness predicates satisfied (e.g., ρ > 0, p > 0)|
| **Witness fields**| `predicates`, `all_satisfied`, `failed`                       |
| **Satisfied when**| All predicates in `predicates` map are `true`                 |
| **Domains**       | Compressible flows, phase-field, any domain with physical bounds|

**Witness schema:**

```json
{
  "predicates": {
    "rho_positive": true,
    "pressure_positive": true
  },
  "all_satisfied": true,
  "failed": []
}
```

---

## Verification Rules for Clients

When verifying a certificate, clients SHOULD:

1. Check that `certificate_version` is `"1.0.0"`
2. Verify the signature using the server's public key
3. Check that all `claims[].tag` values are in this registry
4. Check that all `claims[].satisfied` are `true`
5. Optionally: re-compute `satisfied` from the witness data to confirm server's evaluation
