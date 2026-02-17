import { describe, it, expect, vi, beforeEach } from "vitest";
import { ProviderNotFoundError } from "@luxury/core";

// Mock server-only (imported by provider.ts, logger.ts, metrics.ts)
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
  isProviderReady: () => true,
}));

// Mock observability modules
vi.mock("@/lib/logger", () => ({
  logger: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

vi.mock("@/lib/metrics", () => ({
  increment: vi.fn(),
  observe: vi.fn(),
}));

// Helper: create a Request with X-Request-Id header
function req(url: string, extraHeaders?: Record<string, string>) {
  return new Request(url, {
    headers: { "x-request-id": "test-req-id", ...extraHeaders },
  });
}

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

      const response = await GET(req("http://localhost/api/packages"));
      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.packages).toHaveLength(1);
      expect(body.packages[0].id).toBe("pass");
    });

    it("returns Cache-Control and Server-Timing headers", async () => {
      const { GET } = await import("@/app/api/packages/route");
      mockProvider.listPackages.mockResolvedValue([]);

      const response = await GET(req("http://localhost/api/packages"));
      expect(response.headers.get("Cache-Control")).toContain("s-maxage=60");
      expect(response.headers.get("Server-Timing")).toContain("list_packages;dur=");
      expect(response.headers.get("X-Request-Id")).toBe("test-req-id");
    });

    it("includes an ETag header on 200 response", async () => {
      const { GET } = await import("@/app/api/packages/route");
      mockProvider.listPackages.mockResolvedValue([{ id: "pkg-1" }]);

      const response = await GET(req("http://localhost/api/packages"));
      expect(response.status).toBe(200);
      const etag = response.headers.get("ETag");
      expect(etag).toMatch(/^W\/"[a-f0-9]{16}"$/);
    });

    it("returns 304 when If-None-Match matches ETag", async () => {
      const { GET } = await import("@/app/api/packages/route");
      mockProvider.listPackages.mockResolvedValue([{ id: "pkg-1" }]);

      // First request to capture ETag
      const first = await GET(req("http://localhost/api/packages"));
      const etag = first.headers.get("ETag")!;

      // Second request with matching If-None-Match
      const second = await GET(
        req("http://localhost/api/packages", { "If-None-Match": etag }),
      );
      expect(second.status).toBe(304);
      expect(second.headers.get("ETag")).toBe(etag);
    });

    it("returns 500 on provider error", async () => {
      const { GET } = await import("@/app/api/packages/route");
      mockProvider.listPackages.mockRejectedValue(new Error("Disk read failure"));

      const response = await GET(req("http://localhost/api/packages"));
      expect(response.status).toBe(500);
      const body = await response.json();
      expect(body.error).toContain("Disk read failure");
    });
  });

  describe("GET /api/packages/[id]", () => {
    it("returns a proof package", async () => {
      const { GET } = await import("@/app/api/packages/[id]/route");
      mockProvider.loadPackage.mockResolvedValue({ schema_version: "1.0.0", meta: { id: "run-001" } });

      const response = await GET(req("http://localhost/api/packages/pass"), {
        params: { id: "pass" },
      });
      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.schema_version).toBe("1.0.0");
    });

    it("returns 400 for invalid ID", async () => {
      const { GET } = await import("@/app/api/packages/[id]/route");
      const response = await GET(req("http://localhost/api/packages/bad!id"), {
        params: { id: "bad!id" },
      });
      expect(response.status).toBe(400);
    });

    it("returns 404 for missing package", async () => {
      const { GET } = await import("@/app/api/packages/[id]/route");
      mockProvider.loadPackage.mockRejectedValue(new ProviderNotFoundError("package", "missing"));

      const response = await GET(req("http://localhost/api/packages/missing"), {
        params: { id: "missing" },
      });
      expect(response.status).toBe(404);
    });

    it("returns Cache-Control header", async () => {
      const { GET } = await import("@/app/api/packages/[id]/route");
      mockProvider.loadPackage.mockResolvedValue({ schema_version: "1.0.0" });

      const response = await GET(req("http://localhost/api/packages/pass"), {
        params: { id: "pass" },
      });
      expect(response.headers.get("Cache-Control")).toContain("s-maxage=300");
      expect(response.headers.get("Server-Timing")).toContain("load_package;dur=");
    });

    it("includes an ETag header on 200 response", async () => {
      const { GET } = await import("@/app/api/packages/[id]/route");
      mockProvider.loadPackage.mockResolvedValue({ schema_version: "1.0.0", meta: { id: "run-001" } });

      const response = await GET(req("http://localhost/api/packages/pass"), {
        params: { id: "pass" },
      });
      expect(response.status).toBe(200);
      expect(response.headers.get("ETag")).toMatch(/^W\/"[a-f0-9]{16}"$/);
    });

    it("returns 304 when If-None-Match matches ETag", async () => {
      const { GET } = await import("@/app/api/packages/[id]/route");
      mockProvider.loadPackage.mockResolvedValue({ schema_version: "1.0.0", meta: { id: "run-001" } });

      const first = await GET(req("http://localhost/api/packages/pass"), {
        params: { id: "pass" },
      });
      const etag = first.headers.get("ETag")!;

      const second = await GET(
        req("http://localhost/api/packages/pass", { "If-None-Match": etag }),
        { params: { id: "pass" } },
      );
      expect(second.status).toBe(304);
      expect(second.headers.get("ETag")).toBe(etag);
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

      const response = await GET(req("http://localhost/api/packages/pass/artifacts/artifacts/timeseries.csv"), {
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

      const response = await GET(req("http://localhost/api/packages/pass/artifacts/artifacts/missing.csv"), {
        params: { id: "pass", path: ["artifacts", "missing.csv"] },
      });
      expect(response.status).toBe(404);
    });

    it("returns 400 for invalid package ID", async () => {
      const { GET } = await import("@/app/api/packages/[id]/artifacts/[...path]/route");
      const response = await GET(req("http://localhost/api/packages/bad!id/artifacts/file.csv"), {
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

      const response = await GET(req("http://localhost/api/domains/com.physics.vlasov"), {
        params: { domain: "com.physics.vlasov" },
      });
      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.id).toBe("com.physics.vlasov");
    });

    it("returns 400 for invalid domain ID", async () => {
      const { GET } = await import("@/app/api/domains/[domain]/route");
      const response = await GET(req("http://localhost/api/domains/bad!domain"), {
        params: { domain: "bad!domain" },
      });
      expect(response.status).toBe(400);
    });

    it("returns 404 for missing domain pack", async () => {
      const { GET } = await import("@/app/api/domains/[domain]/route");
      mockProvider.loadDomainPack.mockRejectedValue(new ProviderNotFoundError("domain", "nonexistent"));

      const response = await GET(req("http://localhost/api/domains/nonexistent"), {
        params: { domain: "nonexistent" },
      });
      expect(response.status).toBe(404);
    });

    it("returns Cache-Control with long max-age", async () => {
      const { GET } = await import("@/app/api/domains/[domain]/route");
      mockProvider.loadDomainPack.mockResolvedValue({ id: "test", version: "1.0.0" });

      const response = await GET(req("http://localhost/api/domains/test"), {
        params: { domain: "test" },
      });
      expect(response.headers.get("Cache-Control")).toContain("s-maxage=3600");
      expect(response.headers.get("Server-Timing")).toContain("load_domain;dur=");
    });

    it("includes an ETag header on 200 response", async () => {
      const { GET } = await import("@/app/api/domains/[domain]/route");
      mockProvider.loadDomainPack.mockResolvedValue({ id: "com.physics.vlasov", version: "1.0.0" });

      const response = await GET(req("http://localhost/api/domains/com.physics.vlasov"), {
        params: { domain: "com.physics.vlasov" },
      });
      expect(response.status).toBe(200);
      expect(response.headers.get("ETag")).toMatch(/^W\/"[a-f0-9]{16}"$/);
    });

    it("returns 304 when If-None-Match matches ETag", async () => {
      const { GET } = await import("@/app/api/domains/[domain]/route");
      mockProvider.loadDomainPack.mockResolvedValue({ id: "com.physics.vlasov", version: "1.0.0" });

      const first = await GET(req("http://localhost/api/domains/com.physics.vlasov"), {
        params: { domain: "com.physics.vlasov" },
      });
      const etag = first.headers.get("ETag")!;

      const second = await GET(
        req("http://localhost/api/domains/com.physics.vlasov", { "If-None-Match": etag }),
        { params: { domain: "com.physics.vlasov" } },
      );
      expect(second.status).toBe(304);
      expect(second.headers.get("ETag")).toBe(etag);
    });
  });
});
