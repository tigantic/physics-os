import fs from "node:fs/promises";
import path from "node:path";
import type { ProofPackage } from "../schema/proofPackage.zod.js";
import type { DomainPack } from "../schema/domainPack.zod.js";
import type { ProofDataProvider, PackageSummary, ArtifactReadResult } from "./types.js";
import { ProofPackageSchema } from "../schema/proofPackage.zod.js";
import { verifyProofPackageArtifacts } from "../proof/integrity.js";
import { readArtifactBytes } from "../proof/artifactStore.js";
import { loadDomainPackForDomain, loadDomainPackById } from "../domain/domainPackRegistry.js";
import { deepFreeze } from "../util/deepFreeze.js";

/**
 * Filesystem-backed ProofDataProvider.
 *
 * Reads proof packages and domain packs from the local fixtures directory.
 * This is the production default for standalone / self-hosted deployments.
 */
export class FilesystemProvider implements ProofDataProvider {
  readonly name = "filesystem";
  private readonly fixturesRoot: string;

  constructor(fixturesRoot: string) {
    this.fixturesRoot = path.resolve(fixturesRoot);
  }

  async listPackages(): Promise<readonly PackageSummary[]> {
    const packagesDir = path.join(this.fixturesRoot, "proof-packages");
    let entries: string[];
    try {
      const dirents = await fs.readdir(packagesDir, { withFileTypes: true });
      entries = dirents.filter((d) => d.isDirectory()).map((d) => d.name);
    } catch {
      return [];
    }

    const summaries: PackageSummary[] = [];
    for (const id of entries) {
      const jsonPath = path.join(packagesDir, id, "proofPackage.json");
      try {
        const rawText = await fs.readFile(jsonPath, "utf8");
        const raw = JSON.parse(rawText) as Record<string, unknown>;
        const meta = raw.meta as Record<string, unknown> | undefined;
        const verdict = raw.verdict as Record<string, unknown> | undefined;
        const solver = meta?.solver as Record<string, unknown> | undefined;

        summaries.push({
          id,
          domain_id: String(meta?.domain_id ?? "unknown"),
          verdict_status: String(verdict?.status ?? "INCOMPLETE"),
          quality_score: typeof verdict?.quality_score === "number" ? verdict.quality_score : 0,
          timestamp: String(meta?.timestamp ?? ""),
          solver_name: String(solver?.name ?? "unknown"),
        });
      } catch {
        // Skip packages with unreadable/invalid JSON — don't fail the listing
        continue;
      }
    }

    return summaries;
  }

  async loadPackage(id: string): Promise<ProofPackage> {
    assertSafeId(id, "package id");
    const bundleDir = path.join(this.fixturesRoot, "proof-packages", id);
    const jsonPath = path.join(bundleDir, "proofPackage.json");

    let rawText: string;
    try {
      rawText = await fs.readFile(jsonPath, "utf8");
    } catch {
      throw new Error(`Proof package not found: ${id}`);
    }

    let raw: unknown;
    try {
      raw = JSON.parse(rawText);
    } catch (err) {
      throw new Error(`Invalid JSON in proof package "${id}": ${err instanceof Error ? err.message : String(err)}`, {
        cause: err,
      });
    }

    const parsed = ProofPackageSchema.parse(raw);
    const verification = await verifyProofPackageArtifacts(parsed, bundleDir);

    return deepFreeze({
      ...parsed,
      verification: {
        status: verification.status,
        failures: verification.failures.map((f) => ({
          code: f.code,
          message: f.message,
          artifact_id: f.artifact_id,
        })),
        verifier_version: "0.1.0",
      },
    });
  }

  async loadDomainPack(domain: string): Promise<DomainPack> {
    // Try resolving as a TPC domain_id first, then as a direct pack ID
    try {
      return await loadDomainPackForDomain(this.fixturesRoot, domain);
    } catch {
      return await loadDomainPackById(this.fixturesRoot, domain);
    }
  }

  async readArtifact(packageId: string, artifactUri: string): Promise<ArtifactReadResult> {
    assertSafeId(packageId, "package id");
    const bundleDir = path.join(this.fixturesRoot, "proof-packages", packageId);
    const result = await readArtifactBytes(bundleDir, artifactUri);
    if (!result.ok) return result;

    // Determine MIME type from the artifact URI extension
    const ext = path.extname(artifactUri).toLowerCase();
    const mimeType = MIME_MAP[ext] ?? "application/octet-stream";

    return {
      ok: true,
      bytes: result.bytes,
      hash: result.hash,
      mimeType,
    };
  }
}

/** Validate that an ID contains only safe filesystem characters. */
function assertSafeId(id: string, label: string): void {
  if (/\.\.[\\/]/.test(id) || /[\\/]/.test(id)) {
    throw new Error(`${label} contains path traversal characters: ${id}`);
  }
  if (!/^[a-zA-Z0-9._-]+$/.test(id)) {
    throw new Error(`${label} contains invalid characters: ${id}`);
  }
}

/** Common MIME types for proof artifacts. */
const MIME_MAP: Record<string, string> = {
  ".csv": "text/csv",
  ".json": "application/json",
  ".log": "text/plain",
  ".txt": "text/plain",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".bin": "application/octet-stream",
};
