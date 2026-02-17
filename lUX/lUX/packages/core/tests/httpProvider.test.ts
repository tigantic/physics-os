import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { HttpProvider } from "../src/providers/HttpProvider.js";

/**
 * Tests for HttpProvider using mocked fetch.
 */

const BASE_URL = "https://lux.example.com";
let provider: HttpProvider;

// Minimal valid ProofPackage for response mocking
const mockProof = {
  schema_version: "1.0.0",
  meta: {
    id: "run-001",
    project_id: "hydro-sim",
    domain_id: "com.physics.vlasov",
    solver: { name: "TestSolver", version: "1.0.0", commit_hash: "abc123" },
    environment: { container_digest: "sha256:" + "a".repeat(64), seed: 42, arch: "x86_64" },
    timestamp: "2026-02-16T00:00:00Z",
  },
  verdict: { status: "PASS", reason: "All gates passed", quality_score: 0.95 },
  attestation: {
    merkle_root: "sha256:" + "b".repeat(64),
    signatures: [{ key_id: "k1", algorithm: "ed25519", signature: "sig", timestamp: "2026-02-16T00:00:00Z" }],
  },
  claims: {},
  gate_results: {},
  gate_manifests: {},
  artifacts: {},
  timeline: { step_count: 0, steps: [] },
};

const mockDomainPack = {
  id: "com.physics.vlasov",
  version: "1.0.0",
  metrics: {},
  gate_packs: {},
  viewers: [],
  templates: {
    executive_summary_metric_ids: [],
    publication_sections: [],
    citation_format: "bibtex",
  },
};

describe("HttpProvider", () => {
  beforeEach(() => {
    provider = new HttpProvider(BASE_URL);
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("listPackages", () => {
    it("fetches from /api/packages and validates response", async () => {
      const mockResponse = {
        ok: true,
        json: () =>
          Promise.resolve({
            packages: [
              {
                id: "pass",
                domain_id: "com.physics.vlasov",
                verdict_status: "PASS",
                quality_score: 0.95,
                timestamp: "2026-02-16T00:00:00Z",
                solver_name: "TestSolver",
              },
            ],
          }),
      };
      vi.mocked(fetch).mockResolvedValue(mockResponse as Response);

      const packages = await provider.listPackages();
      expect(packages).toHaveLength(1);
      expect(packages[0].id).toBe("pass");
      expect(fetch).toHaveBeenCalledWith(`${BASE_URL}/api/packages`, expect.any(Object));
    });

    it("throws on network error", async () => {
      vi.mocked(fetch).mockRejectedValue(new Error("ECONNREFUSED"));
      await expect(provider.listPackages()).rejects.toThrow("Network error");
    });

    it("throws on HTTP error with body", async () => {
      vi.mocked(fetch).mockResolvedValue({
        ok: false,
        status: 500,
        text: () => Promise.resolve("Internal Server Error"),
      } as Response);
      await expect(provider.listPackages()).rejects.toThrow("HTTP 500");
    });
  });

  describe("loadPackage", () => {
    it("fetches from /api/packages/:id and validates with ProofPackageSchema", async () => {
      vi.mocked(fetch).mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockProof),
      } as Response);

      const proof = await provider.loadPackage("pass");
      expect(proof.schema_version).toBe("1.0.0");
      expect(proof.meta.id).toBe("run-001");
      expect(fetch).toHaveBeenCalledWith(`${BASE_URL}/api/packages/pass`, expect.any(Object));
    });

    it("rejects path traversal in ID", async () => {
      await expect(provider.loadPackage("../etc/passwd")).rejects.toThrow("path traversal");
    });

    it("rejects invalid characters in ID", async () => {
      await expect(provider.loadPackage("pass!@#")).rejects.toThrow("invalid characters");
    });

    it("returns deep-frozen object", async () => {
      vi.mocked(fetch).mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockProof),
      } as Response);

      const proof = await provider.loadPackage("pass");
      expect(Object.isFrozen(proof)).toBe(true);
    });
  });

  describe("loadDomainPack", () => {
    it("fetches from /api/domains/:domain and validates with DomainPackSchema", async () => {
      vi.mocked(fetch).mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockDomainPack),
      } as Response);

      const pack = await provider.loadDomainPack("com.physics.vlasov");
      expect(pack.id).toBe("com.physics.vlasov");
      expect(fetch).toHaveBeenCalledWith(`${BASE_URL}/api/domains/com.physics.vlasov`, expect.any(Object));
    });
  });

  describe("readArtifact", () => {
    it("returns bytes and hash from response headers", async () => {
      const csvBytes = new TextEncoder().encode("step,value\n0,1.0\n1,2.0");
      vi.mocked(fetch).mockResolvedValue({
        ok: true,
        headers: new Headers({
          "Content-Type": "text/csv",
          "X-Artifact-Hash": "sha256:" + "c".repeat(64),
        }),
        arrayBuffer: () => Promise.resolve(csvBytes.buffer),
      } as Response);

      const result = await provider.readArtifact("pass", "artifacts/timeseries.csv");
      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.bytes.byteLength).toBeGreaterThan(0);
        expect(result.hash).toBe("sha256:" + "c".repeat(64));
        expect(result.mimeType).toBe("text/csv");
      }
    });

    it("computes hash client-side if server does not provide it", async () => {
      const csvBytes = new TextEncoder().encode("step,value\n0,1.0");
      vi.mocked(fetch).mockResolvedValue({
        ok: true,
        headers: new Headers({
          "Content-Type": "text/csv",
        }),
        arrayBuffer: () => Promise.resolve(csvBytes.buffer),
      } as Response);

      const result = await provider.readArtifact("pass", "artifacts/timeseries.csv");
      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.hash).toMatch(/^sha256:[a-f0-9]{64}$/i);
      }
    });

    it("returns failure on HTTP error", async () => {
      vi.mocked(fetch).mockResolvedValue({
        ok: false,
        status: 404,
        statusText: "Not Found",
      } as Response);

      const result = await provider.readArtifact("pass", "artifacts/missing.csv");
      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.reason).toContain("404");
      }
    });

    it("returns failure on network error", async () => {
      vi.mocked(fetch).mockRejectedValue(new Error("ECONNREFUSED"));

      const result = await provider.readArtifact("pass", "artifacts/timeseries.csv");
      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.reason).toContain("Network error");
      }
    });
  });

  describe("name", () => {
    it('returns "http"', () => {
      expect(provider.name).toBe("http");
    });
  });

  describe("constructor", () => {
    it("strips trailing slashes from base URL", async () => {
      const p = new HttpProvider("https://example.com///");
      vi.mocked(fetch).mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ packages: [] }),
      } as Response);

      await p.listPackages();
      expect(fetch).toHaveBeenCalledWith("https://example.com/api/packages", expect.any(Object));
    });

    it("merges custom headers", async () => {
      const p = new HttpProvider(BASE_URL, { Authorization: "Bearer token123" });
      vi.mocked(fetch).mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ packages: [] }),
      } as Response);

      await p.listPackages();
      const [, options] = vi.mocked(fetch).mock.calls[0];
      const headers = (options as RequestInit).headers as Record<string, string>;
      expect(headers.Authorization).toBe("Bearer token123");
      expect(headers.Accept).toBe("application/json");
    });
  });
});
