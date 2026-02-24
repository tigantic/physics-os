# Pricing Model

**Baseline**: v4.0.0
**Scope**: Private alpha — shadow billing only (no real charges)
**Status**: DRAFT — prices are placeholders for billing infrastructure validation

---

## 1. Product Positioning

HyperTensor is **verifiable structured compute infrastructure**.
Physics simulation is the proof case.  Pricing reflects compute
consumption, not physics expertise.

Clients pay for:

- **Compute time** consumed by their simulations
- **Verification** (validation + certificate issuance)
- **Result delivery** (field data transfer)

Clients do NOT pay for:

- API access (included)
- Health/capabilities queries
- Failed jobs (no charge)
- Certificate verification (read-only)

---

## 2. Pricing Tiers (Alpha)

### 2.1 Compute Unit Definition

One **Compute Unit (CU)** represents:

```
1 CU = 1 second of single-core CPU execution
      = 0.1 seconds of GPU execution (10x multiplier)
```

CU consumption for a job:

```
CU = wall_time_s × device_multiplier
```

| Device Class | Multiplier | Rationale                       |
|--------------|------------|---------------------------------|
| `cpu`        | 1.0        | Baseline                        |
| `cuda`       | 10.0       | GPU time is more expensive      |

### 2.2 Alpha Packages

| Package         | CU Included | Price/Month | Overage Rate | Notes               |
|-----------------|-------------|-------------|--------------|----------------------|
| **Explorer**    | 100 CU      | $0          | N/A          | Invite-only alpha    |
| **Builder**     | 1,000 CU    | $49         | $0.05/CU     | Early adopter        |
| **Professional**| 10,000 CU   | $299        | $0.03/CU     | Production workloads |

**Alpha note**: All alpha users are on the Explorer package.
No real charges during alpha.  The billing system runs in shadow
mode to validate metering accuracy and invoice generation.

### 2.3 Per-Job Cost Estimate

Typical jobs and their approximate CU consumption:

| Domain            | n_bits | n_steps | Device | Wall Time (est.) | CU (est.) |
|-------------------|--------|---------|--------|-----------------|-----------|
| `burgers`         | 8      | 100     | CPU    | 0.3s            | 0.3       |
| `burgers`         | 12     | 1000    | CPU    | 15s             | 15        |
| `maxwell`         | 8      | 100     | CPU    | 0.5s            | 0.5       |
| `maxwell_3d`      | 6      | 100     | CPU    | 2.0s            | 2.0       |
| `navier_stokes_2d`| 8      | 500     | CPU    | 5.0s            | 5.0       |
| `vlasov_poisson`  | 8      | 100     | CPU    | 1.0s            | 1.0       |
| `schrodinger`     | 10     | 200     | CPU    | 3.0s            | 3.0       |
| Any domain        | 14     | 10000   | GPU    | 60s             | 600       |

---

## 3. What Is Metered

| Activity                              | Metered? | Unit                |
|---------------------------------------|----------|---------------------|
| `POST /v1/jobs` (success)             | Yes      | CU (wall_time × multiplier) |
| `POST /v1/jobs` (failure)             | No       | —                   |
| `GET /v1/jobs/{id}`                   | No       | —                   |
| `GET /v1/jobs/{id}/result`            | No       | —                   |
| `GET /v1/jobs/{id}/validation`        | No       | —                   |
| `GET /v1/jobs/{id}/certificate`       | No       | —                   |
| `POST /v1/validate`                   | No       | —                   |
| `GET /v1/capabilities`               | No       | —                   |
| `GET /v1/contracts`                  | No       | —                   |
| `GET /v1/health`                     | No       | —                   |
| Idempotent replay (same key)         | No       | —                   |

---

## 4. Billing Cycle

| Property            | Value                                       |
|---------------------|---------------------------------------------|
| Billing period      | Calendar month (UTC)                        |
| Invoice generated   | 1st of following month                      |
| Payment terms       | Net 30                                      |
| Currency            | USD                                         |
| Minimum charge      | $0 (Explorer)                               |

---

## 5. Shadow Billing (Alpha)

During alpha, the billing system:

1. **Meters** every successful job (records CU consumed)
2. **Calculates** monthly totals per API key
3. **Generates** shadow invoices (not sent to users)
4. **Does NOT** enforce quotas or charge real money

Shadow billing validates:

- Metering accuracy (CU calculation matches wall time)
- Invoice generation (correct line items and totals)
- API key → usage association
- Monthly aggregation

### 5.1 Shadow Invoice Format

```json
{
  "invoice_id": "INV-2024-01-alice",
  "period": "2024-01",
  "api_key_suffix": "...key1",
  "line_items": [
    {
      "date": "2024-01-15",
      "job_id": "uuid",
      "domain": "burgers",
      "device_class": "cpu",
      "wall_time_s": 0.34,
      "compute_units": 0.34,
      "unit_price_usd": 0.05
    }
  ],
  "total_cu": 45.6,
  "total_usd": 2.28,
  "package": "explorer",
  "included_cu": 100,
  "overage_cu": 0,
  "overage_usd": 0.00,
  "shadow": true
}
```

---

## 6. Future Considerations (Post-Alpha)

- **Quota enforcement**: Reject jobs when CU budget exhausted
- **Pre-paid credits**: Purchase CU blocks in advance
- **Reserved capacity**: Guaranteed GPU time slots
- **Enterprise pricing**: Custom CU rates + SLA
- **Data egress**: Charge for large field data downloads
- **Storage**: Charge for result retention beyond 7 days
