import { z } from "zod";
import type { ProofPackage } from "../schema/proofPackage.zod.js";
import type { DomainPack } from "../schema/domainPack.zod.js";
import type { ProofDataProvider, PackageSummary, ArtifactReadResult } from "./types.js";
import { ProofPackageSchema } from "../schema/proofPackage.zod.js";
import { DomainPackSchema } from "../schema/domainPack.zod.js";
import { deepFreeze } from "../util/deepFreeze.js";

/**
 * Zod schema for the `/api/packages` response.
 * Validates the summary list returned by the API.
 */
const PackageSummarySchema = z.object({
  id: z.string(),
  domain_id: z.string(),
  verdict_status: z.string(),
  quality_score: z.number(),
  timestamp: z.string(),
  solver_name: z.string(),
  verification_status: z.string().optional(),
});

const PackageListResponseSchema = z.object({
  packages: z.array(PackageSummarySchema),
});

/**
 * HTTP-backed ProofDataProvider.
 *
 * Fetches proof packages, domain packs, and artifact bytes from a remote
 * lUX API server. Used when `LUX_DATA_PROVIDER=http`.
 */
export class HttpProvider implements ProofDataProvider {
  readonly name = "http";
  private readonly baseUrl: string;
  private readonly headers: Record<string, string>;

  /**
   * @param baseUrl API base URL (e.g. "https://lux.example.com").
   *                Must NOT end with a trailing slash.
   * @param headers Optional extra headers (e.g. Authorization).
   */
  constructor(baseUrl: string, headers?: Record<string, string>) {
    // Strip trailing slash to avoid double-slash in URL construction
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.headers = {
      Accept: "application/json",
      ...headers,
    };
  }

  async listPackages(): Promise<readonly PackageSummary[]> {
    const res = await this.fetchJson(`${this.baseUrl}/api/packages`);
    const parsed = PackageListResponseSchema.parse(res);
    return parsed.packages;
  }

  async loadPackage(id: string): Promise<ProofPackage> {
    assertSafeId(id, "package id");
    const res = await this.fetchJson(`${this.baseUrl}/api/packages/${encodeURIComponent(id)}`);
    const proof = ProofPackageSchema.parse(res);
    return deepFreeze(proof);
  }

  async loadDomainPack(domain: string): Promise<DomainPack> {
    assertSafeId(domain, "domain id");
    const res = await this.fetchJson(`${this.baseUrl}/api/domains/${encodeURIComponent(domain)}`);
    const pack = DomainPackSchema.parse(res);
    return deepFreeze(pack);
  }

  async readArtifact(packageId: string, artifactUri: string): Promise<ArtifactReadResult> {
    assertSafeId(packageId, "package id");
    const url = `${this.baseUrl}/api/packages/${encodeURIComponent(packageId)}/artifacts/${encodeURIComponent(artifactUri)}`;

    let response: Response;
    try {
      response = await fetch(url, { headers: this.headers });
    } catch (err) {
      return {
        ok: false,
        reason: `Network error fetching artifact: ${err instanceof Error ? err.message : String(err)}`,
      };
    }

    if (!response.ok) {
      return {
        ok: false,
        reason: `HTTP ${response.status}: ${response.statusText}`,
      };
    }

    const arrayBuffer = await response.arrayBuffer();
    const bytes = new Uint8Array(arrayBuffer);

    // Hash and MIME from response headers if available
    const hash = response.headers.get("X-Artifact-Hash") as `sha256:${string}` | null;
    const mimeType = response.headers.get("Content-Type") ?? "application/octet-stream";

    if (!hash) {
      // Compute hash client-side if server didn't provide it
      const { sha256Prefixed } = await import("../util/hash.js");
      return {
        ok: true,
        bytes,
        hash: sha256Prefixed(bytes),
        mimeType,
      };
    }

    return { ok: true, bytes, hash, mimeType };
  }

  /**
   * Fetch JSON from the API, throwing descriptive errors on failure.
   */
  private async fetchJson(url: string): Promise<unknown> {
    let response: Response;
    try {
      response = await fetch(url, { headers: this.headers });
    } catch (err) {
      throw new Error(`Network error: ${err instanceof Error ? err.message : String(err)}`, { cause: err });
    }

    if (!response.ok) {
      let body = "";
      try {
        body = await response.text();
      } catch {
        /* ignore */
      }
      throw new Error(`HTTP ${response.status} from ${url}: ${body.slice(0, 500)}`);
    }

    return response.json();
  }
}

/** Validate that an ID contains only safe characters. */
function assertSafeId(id: string, label: string): void {
  if (/\.\.[\\/]/.test(id) || /[\\/]/.test(id)) {
    throw new Error(`${label} contains path traversal characters: ${id}`);
  }
  if (!/^[a-zA-Z0-9._-]+$/.test(id)) {
    throw new Error(`${label} contains invalid characters: ${id}`);
  }
}
