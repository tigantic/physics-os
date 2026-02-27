import type { ProofPackage } from "../schema/proofPackage.zod.js";
import type { DomainPack } from "../schema/domainPack.zod.js";

/**
 * Lightweight summary returned by `listPackages()`.
 * Avoids loading full proof data for directory listings.
 */
export interface PackageSummary {
  /** Package directory name or unique identifier */
  readonly id: string;
  /** TPC domain identifier (e.g. "com.physics.vlasov") */
  readonly domain_id: string;
  /** Top-level verdict: PASS | FAIL | WARN | INCOMPLETE */
  readonly verdict_status: string;
  /** Quality score 0–1 */
  readonly quality_score: number;
  /** ISO-8601 creation timestamp */
  readonly timestamp: string;
  /** Solver name */
  readonly solver_name: string;
  /** Verification status if run (VERIFIED | UNVERIFIED | BROKEN_CHAIN | UNSUPPORTED) */
  readonly verification_status?: string;
}

/**
 * Successful artifact read result.
 */
export interface ArtifactReadOk {
  readonly ok: true;
  readonly bytes: Uint8Array;
  readonly hash: `sha256:${string}`;
  readonly mimeType: string;
}

/**
 * Failed artifact read result.
 */
export interface ArtifactReadFail {
  readonly ok: false;
  readonly reason: string;
}

export type ArtifactReadResult = ArtifactReadOk | ArtifactReadFail;

/**
 * Abstract data provider interface for loading proof packages, domain packs,
 * and streaming artifact bytes from any backing store.
 *
 * Consumers import the interface and program against it; the concrete
 * implementation is selected at runtime via `createProvider()`.
 */
export interface ProofDataProvider {
  /** Human-readable provider name for diagnostics. */
  readonly name: string;

  /**
   * List available proof packages without loading full data.
   * @returns Promise of lightweight package summaries.
   */
  listPackages(): Promise<readonly PackageSummary[]>;

  /**
   * Load a full proof package by its identifier.
   * Includes Zod validation, artifact hash verification (for fs provider),
   * and deep-freeze.
   *
   * @param id Package identifier (directory name for fs, opaque ID for HTTP).
   */
  loadPackage(id: string): Promise<ProofPackage>;

  /**
   * Load a domain pack by TPC domain_id or pack ID.
   *
   * @param domain TPC domain_id (e.g. "II.2") or pack ID (e.g. "com.physics.euler_3d").
   */
  loadDomainPack(domain: string): Promise<DomainPack>;

  /**
   * Read artifact bytes for a given package and artifact URI.
   *
   * @param packageId Package identifier.
   * @param artifactUri Artifact URI relative to the package root (e.g. "artifacts/timeseries.csv").
   */
  readArtifact(packageId: string, artifactUri: string): Promise<ArtifactReadResult>;
}
