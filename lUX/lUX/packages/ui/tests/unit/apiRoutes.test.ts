import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock server-only (imported by provider.ts)
vi.mock("server-only", () => ({}));

// Mock the provider module before importing routes
const mockProvider = {
  name: "filesystem",
  listPackages: vi.fn(),
  loadPackage: vi.fn(),
  loadDomainPack: vi.fn(),
  readArtifact: vi.fn(),
};

vi.mock("@/config/provider", () => ({
  getProvider: () => Promise.resolve(mockProvider),
}));

describe("API Routes", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("GET /api/packages", () => {
    it("returns package summaries", async () => {
      const { GET } = await import("@/app/api/packages/route");
      mockProvider.listPackages.mockResolvedValue([
        {
          id: "pass",
          domain_id: "com.physics.vlasov",
          verdict_status: "PASS",
          quality_score: 0.95,
          timestamp: "2026-02-16T00:00:00Z",
          solver_name: "TestSolver",
        },
      ]);

      const response = await GET();
      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.packages).toHaveLength(1);
      expect(body.packages[0].id).toBe("pass");
    });

    it("returns Cache-Control header", async () => {
      const { GET } = await import("@/app/api/packages/route");
      mockProvider.listPackages.mockResolvedValue([]);

      const response = await GET();
      expect(response.headers.get("Cache-Control")).toContain("s-maxage=60");
    });

    it("returns 500 on provider error", async () => {
      const { GET } = await import("@/app/api/packages/route");
      mockProvider.listPackages.mockRejectedValue(new Error("Disk read failure"));

      const response = await GET();
      expect(response.status).toBe(500);
      const body = await response.json();
      expect(body.error).toContain("Disk read failure");
    });
  });

  describe("GET /api/packages/[id]", () => {
    it("returns a proof package", async () => {
      const { GET } = await import("@/app/api/packages/[id]/route");
      mockProvider.loadPackage.mockResolvedValue({ schema_version: "1.0.0", meta: { id: "run-001" } });

      const response = await GET(new Request("http://localhost/api/packages/pass"), {
        params: { id: "pass" },
      });
      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.schema_version).toBe("1.0.0");
    });

    it("returns 400 for invalid ID", async () => {
      const { GET } = await import("@/app/api/packages/[id]/route");
      const response = await GET(new Request("http://localhost/api/packages/bad!id"), {
        params: { id: "bad!id" },
      });
      expect(response.status).toBe(400);
    });

    it("returns 404 for missing package", async () => {
      const { GET } = await import("@/app/api/packages/[id]/route");
      mockProvider.loadPackage.mockRejectedValue(new Error("Proof package not found: missing"));

      const response = await GET(new Request("http://localhost/api/packages/missing"), {
        params: { id: "missing" },
      });
      expect(response.status).toBe(404);
    });

    it("returns Cache-Control header", async () => {
      const { GET } = await import("@/app/api/packages/[id]/route");
      mockProvider.loadPackage.mockResolvedValue({ schema_version: "1.0.0" });

      const response = await GET(new Request("http://localhost/api/packages/pass"), {
        params: { id: "pass" },
      });
      expect(response.headers.get("Cache-Control")).toContain("s-maxage=300");
    });
  });

  describe("GET /api/packages/[id]/artifacts/[...path]", () => {
    it("streams artifact bytes with correct headers", async () => {
      const { GET } = await import("@/app/api/packages/[id]/artifacts/[...path]/route");

      const bytes = new TextEncoder().encode("step,value\n0,1.0");
      mockProvider.readArtifact.mockResolvedValue({
        ok: true,
        bytes,
        hash: "sha256:" + "a".repeat(64),
        mimeType: "text/csv",
      });

      const response = await GET(new Request("http://localhost/api/packages/pass/artifacts/artifacts/timeseries.csv"), {
        params: { id: "pass", path: ["artifacts", "timeseries.csv"] },
      });
      expect(response.status).toBe(200);
      expect(response.headers.get("Content-Type")).toBe("text/csv");
      expect(response.headers.get("X-Artifact-Hash")).toBe("sha256:" + "a".repeat(64));
      expect(response.headers.get("Cache-Control")).toContain("immutable");
    });

    it("returns 404 for missing artifact", async () => {
      const { GET } = await import("@/app/api/packages/[id]/artifacts/[...path]/route");
      mockProvider.readArtifact.mockResolvedValue({
        ok: false,
        reason: "Artifact file missing",
      });

      const response = await GET(new Request("http://localhost/api/packages/pass/artifacts/artifacts/missing.csv"), {
        params: { id: "pass", path: ["artifacts", "missing.csv"] },
      });
      expect(response.status).toBe(404);
    });

    it("returns 400 for invalid package ID", async () => {
      const { GET } = await import("@/app/api/packages/[id]/artifacts/[...path]/route");
      const response = await GET(new Request("http://localhost/api/packages/bad!id/artifacts/file.csv"), {
        params: { id: "bad!id", path: ["file.csv"] },
      });
      expect(response.status).toBe(400);
    });
  });

  describe("GET /api/domains/[domain]", () => {
    it("returns a domain pack", async () => {
      const { GET } = await import("@/app/api/domains/[domain]/route");
      mockProvider.loadDomainPack.mockResolvedValue({
        id: "com.physics.vlasov",
        version: "1.0.0",
        metrics: {},
      });

      const response = await GET(new Request("http://localhost/api/domains/com.physics.vlasov"), {
        params: { domain: "com.physics.vlasov" },
      });
      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.id).toBe("com.physics.vlasov");
    });

    it("returns 400 for invalid domain ID", async () => {
      const { GET } = await import("@/app/api/domains/[domain]/route");
      const response = await GET(new Request("http://localhost/api/domains/bad!domain"), {
        params: { domain: "bad!domain" },
      });
      expect(response.status).toBe(400);
    });

    it("returns 404 for missing domain pack", async () => {
      const { GET } = await import("@/app/api/domains/[domain]/route");
      mockProvider.loadDomainPack.mockRejectedValue(new Error("DomainPack not found: nonexistent"));

      const response = await GET(new Request("http://localhost/api/domains/nonexistent"), {
        params: { domain: "nonexistent" },
      });
      expect(response.status).toBe(404);
    });

    it("returns Cache-Control with long max-age", async () => {
      const { GET } = await import("@/app/api/domains/[domain]/route");
      mockProvider.loadDomainPack.mockResolvedValue({ id: "test", version: "1.0.0" });

      const response = await GET(new Request("http://localhost/api/domains/test"), {
        params: { domain: "test" },
      });
      expect(response.headers.get("Cache-Control")).toContain("s-maxage=3600");
    });
  });
});
