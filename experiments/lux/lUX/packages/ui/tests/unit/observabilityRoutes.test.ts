import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock server-only
vi.mock("server-only", () => ({}));

// Mock provider
vi.mock("@/config/provider", () => ({
  getProvider: vi.fn(),
  isProviderReady: vi.fn(),
}));

// Mock logger
vi.mock("@/lib/logger", () => ({
  logger: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

describe("Observability API Routes", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("GET /api/ready", () => {
    it("returns 200 when provider is ready", async () => {
      const { isProviderReady, getProvider } = await import("@/config/provider");
      vi.mocked(isProviderReady).mockReturnValue(true);

      const { GET } = await import("@/app/api/ready/route");
      const response = await GET();
      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.status).toBe("ready");
      expect(body.version).toBeDefined();
      expect(body.commitSha).toBeDefined();
      // getProvider should NOT be called when already ready
      expect(getProvider).not.toHaveBeenCalled();
    });

    it("returns 200 after successful init when not yet ready", async () => {
      const { isProviderReady, getProvider } = await import("@/config/provider");
      vi.mocked(isProviderReady).mockReturnValue(false);
      vi.mocked(getProvider).mockResolvedValue({
        name: "filesystem",
        listPackages: vi.fn(),
        loadPackage: vi.fn(),
        loadDomainPack: vi.fn(),
        readArtifact: vi.fn(),
      });

      const { GET } = await import("@/app/api/ready/route");
      const response = await GET();
      expect(response.status).toBe(200);
      expect(getProvider).toHaveBeenCalledOnce();
    });

    it("returns 503 when provider fails to initialize", async () => {
      const { isProviderReady, getProvider } = await import("@/config/provider");
      vi.mocked(isProviderReady).mockReturnValue(false);
      vi.mocked(getProvider).mockRejectedValue(new Error("No fixtures root"));

      const { GET } = await import("@/app/api/ready/route");
      const response = await GET();
      expect(response.status).toBe(503);
      const body = await response.json();
      expect(body.status).toBe("not_ready");
      expect(body.reason).toContain("No fixtures root");
    });
  });

  describe("GET /api/metrics", () => {
    it("returns Prometheus text format", async () => {
      const { GET } = await import("@/app/api/metrics/route");
      const response = GET();
      expect(response.status).toBe(200);
      expect(response.headers.get("Content-Type")).toContain("text/plain");
    });

    it("has no-store cache control", async () => {
      const { GET } = await import("@/app/api/metrics/route");
      const response = GET();
      expect(response.headers.get("Cache-Control")).toBe("no-store");
    });

    it("includes Node.js runtime metrics", async () => {
      const { GET } = await import("@/app/api/metrics/route");
      const response = GET();
      const text = await response.text();
      expect(text).toContain("process_uptime_seconds");
      expect(text).toContain("nodejs_heap_used_bytes");
    });
  });

  describe("POST /api/csp-report", () => {
    it("logs CSP violation and returns 204", async () => {
      const { logger } = await import("@/lib/logger");
      const { POST } = await import("@/app/api/csp-report/route");

      const response = await POST(
        new Request("http://localhost/api/csp-report", {
          method: "POST",
          body: JSON.stringify({
            "csp-report": {
              "document-uri": "http://localhost/gallery",
              "violated-directive": "script-src",
              "blocked-uri": "http://evil.com/script.js",
              "original-policy": "script-src 'self'",
            },
          }),
          headers: { "Content-Type": "application/json" },
        }),
      );

      expect(response.status).toBe(204);
      expect(logger.warn).toHaveBeenCalledWith("csp.violation", expect.objectContaining({
        blockedUri: "http://evil.com/script.js",
        violatedDirective: "script-src",
      }));
    });

    it("handles Reporting API v1 format", async () => {
      const { logger } = await import("@/lib/logger");
      const { POST } = await import("@/app/api/csp-report/route");

      const response = await POST(
        new Request("http://localhost/api/csp-report", {
          method: "POST",
          body: JSON.stringify({
            blockedURL: "inline",
            effectiveDirective: "style-src-elem",
            url: "http://localhost/gallery",
          }),
          headers: { "Content-Type": "application/json" },
        }),
      );

      expect(response.status).toBe(204);
      expect(logger.warn).toHaveBeenCalledWith("csp.violation", expect.objectContaining({
        blockedUri: "inline",
        violatedDirective: "style-src-elem",
      }));
    });

    it("returns 400 on invalid JSON", async () => {
      const { POST } = await import("@/app/api/csp-report/route");

      const response = await POST(
        new Request("http://localhost/api/csp-report", {
          method: "POST",
          body: "not json",
          headers: { "Content-Type": "text/plain" },
        }),
      );

      expect(response.status).toBe(400);
    });
  });

  describe("POST /api/errors", () => {
    it("logs client error and returns 204", async () => {
      const { logger } = await import("@/lib/logger");
      const { POST } = await import("@/app/api/errors/route");

      const response = await POST(
        new Request("http://localhost/api/errors", {
          method: "POST",
          body: JSON.stringify({
            message: "Cannot read property 'x'",
            stack: "TypeError: ...\n  at fn (/src/...:1:1)",
            component: "GalleryError",
            url: "http://localhost/gallery?fixture=pass",
            timestamp: "2026-02-17T00:00:00Z",
            userAgent: "Mozilla/5.0",
          }),
          headers: { "Content-Type": "application/json" },
        }),
      );

      expect(response.status).toBe(204);
      expect(logger.error).toHaveBeenCalledWith("client.error", expect.objectContaining({
        message: "Cannot read property 'x'",
        component: "GalleryError",
        source: "error-boundary",
      }));
    });

    it("returns 400 for invalid payload", async () => {
      const { POST } = await import("@/app/api/errors/route");

      const response = await POST(
        new Request("http://localhost/api/errors", {
          method: "POST",
          body: JSON.stringify({ invalid: true }),
          headers: { "Content-Type": "application/json" },
        }),
      );

      expect(response.status).toBe(400);
    });

    it("returns 400 for non-JSON body", async () => {
      const { POST } = await import("@/app/api/errors/route");

      const response = await POST(
        new Request("http://localhost/api/errors", {
          method: "POST",
          body: "not json",
        }),
      );

      expect(response.status).toBe(400);
    });
  });
});
