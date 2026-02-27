import { z } from "zod";

const SHA256 = z.string().regex(/^sha256:[a-f0-9]{64}$/i);
const ISO8601 = z.string();
const UUID = z.string();
const SemVer = z.string().regex(/^\d+\.\d+\.\d+$/);

const DataValueNumber = z.discriminatedUnion("status", [
  z.object({ status: z.literal("ok"), value: z.number() }),
  z.object({ status: z.literal("missing"), reason: z.string() }),
  z.object({ status: z.literal("invalid"), reason: z.string(), details: z.unknown().optional() }),
]);

const DataValueBool = z.discriminatedUnion("status", [
  z.object({ status: z.literal("ok"), value: z.boolean() }),
  z.object({ status: z.literal("missing"), reason: z.string() }),
  z.object({ status: z.literal("invalid"), reason: z.string(), details: z.unknown().optional() }),
]);

const MetricId = z.string();
const ArtifactId = z.string();
const ClaimId = z.string();
const GateId = z.string();

export const ProofPackageSchema = z.object({
  schema_version: z.literal("1.0.0"),

  meta: z.object({
    id: UUID,
    project_id: z.string(),
    domain_id: z.string(),
    solver: z.object({
      name: z.string(),
      version: SemVer,
      commit_hash: z.string(),
    }),
    environment: z.object({
      container_digest: SHA256,
      seed: z.number().int(),
      arch: z.enum(["x86_64", "arm64"]),
      os: z.string().optional(),
      deps_lock_hash: SHA256.optional(),
    }),
    timestamp: ISO8601,
  }),

  verdict: z.object({
    status: z.enum(["PASS", "FAIL", "WARN", "INCOMPLETE"]),
    reason: z.string().optional(),
    quality_score: z.number().min(0).max(1),
    quality_breakdown: z.record(z.string(), z.number().min(0).max(1)).optional(),
  }),

  claims: z.record(
    ClaimId,
    z.object({
      id: ClaimId,
      statement: z.string(),
      category: z.string(),
      tags: z.array(z.string()).default([]),
      gate_ids: z.array(GateId).default([]),
      evidence_refs: z.array(ArtifactId).default([]),
    }),
  ),

  gate_manifests: z.record(
    z.string(),
    z.object({
      id: z.string(),
      version: SemVer,
      gates: z.record(
        GateId,
        z.object({
          id: GateId,
          metric_id: MetricId,
          operator: z.enum(["lt", "gt", "eq", "range"]),
          threshold: z.union([z.number(), z.tuple([z.number(), z.number()])]),
          aggregation: z.enum(["step", "max", "min", "mean", "p95", "final"]).default("step"),
          validity: z
            .object({
              requires_artifacts: z.array(ArtifactId).default([]),
              requires_metrics: z.array(MetricId).default([]),
              domain_assumptions: z.array(z.string()).default([]),
            })
            .default({ requires_artifacts: [], requires_metrics: [], domain_assumptions: [] }),
        }),
      ),
    }),
  ),

  gate_results: z.record(
    GateId,
    z.object({
      gate_id: GateId,
      metric_id: MetricId,
      value: DataValueNumber,
      passed: DataValueBool,
      margin: DataValueNumber,
      evaluated_at: ISO8601.optional(),
      notes: z.string().optional(),
    }),
  ),

  timeline: z.object({
    step_count: z.number().int().nonnegative(),
    steps: z.array(
      z.object({
        step_index: z.number().int().nonnegative(),
        state_hash: SHA256,
        metrics: z.record(MetricId, DataValueNumber).default({}),
        artifact_refs: z.array(ArtifactId).default([]),
      }),
    ),
  }),

  artifacts: z.record(
    ArtifactId,
    z.object({
      id: ArtifactId,
      type: z.enum(["time_series", "field_2d", "field_3d", "tensor", "log", "table", "blob"]),
      mime_type: z.string(),
      hash: SHA256,
      size_bytes: z.number().int().nonnegative(),
      uri: z.string(),
      metadata: z.record(z.string(), z.unknown()).default({}),
    }),
  ),

  attestation: z.object({
    merkle_root: SHA256,
    signatures: z.array(
      z.object({
        key_id: z.string(),
        algorithm: z.enum(["ed25519", "p256", "rsa-pss"]).default("ed25519"),
        signature: z.string(),
        timestamp: ISO8601,
      }),
    ),
  }),

  verification: z
    .object({
      status: z.enum(["VERIFIED", "UNVERIFIED", "BROKEN_CHAIN", "UNSUPPORTED"]),
      verified_at: ISO8601.optional(),
      verifier_version: SemVer.optional(),
      failures: z
        .array(
          z.object({
            code: z.string(),
            message: z.string(),
            artifact_id: ArtifactId.optional(),
            gate_id: GateId.optional(),
          }),
        )
        .default([]),
    })
    .optional(),
});

export type ProofPackage = z.infer<typeof ProofPackageSchema>;
