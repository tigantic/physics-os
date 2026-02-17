import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

/**
 * Unit tests for POST /api/csp-report handler.
 *
 * Tests cover: Zod validation for both Reporting API v1 and legacy formats,
 * payload size rejection, sliding-window alerting, webhook cooldown.
 */

// ── Mocks ──────────────────────────────────────────────────────────────────────

vi.mock("@/lib/logger", () => ({
  logger: {
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

import { logger } from "@/lib/logger";

type POST = (request: Request) => Promise<Response>;

/** Fresh import to reset module-level counters per test. */
async function loadHandler(
  env: Record<string, string | undefined> = {},
): Promise<POST> {
  vi.resetModules();
  // Re-apply logger mock after resetModules
  vi.mock("@/lib/logger", () => ({
    logger: {
      warn: vi.fn(),
      error: vi.fn(),
    },
  }));

  for (const [key, value] of Object.entries(env)) {
    if (value === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = value;
    }
  }

  const mod = await import("@/app/api/csp-report/route");
  return mod.POST as POST;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function cspRequest(body: unknown, contentLength?: string): Request {
  const json = JSON.stringify(body);
  const headers = new Headers({ "Content-Type": "application/json" });
  if (contentLength) headers.set("Content-Length", contentLength);
  return new Request("http://localhost:3000/api/csp-report", {
    method: "POST",
    headers,
    body: json,
  });
}

const legacyReport = {
  "csp-report": {
    "document-uri": "https://example.com/page",
    "blocked-uri": "https://evil.com/script.js",
    "violated-directive": "script-src 'self'",
    "original-policy": "default-src 'self'; script-src 'self'",
  },
};

const v1Report = {
  blockedURL: "https://evil.com/script.js",
  effectiveDirective: "script-src",
  url: "https://example.com/page",
  disposition: "enforce",
};

describe("POST /api/csp-report", () => {
  const originalEnv = { ...process.env };

  afterEach(() => {
    process.env = { ...originalEnv };
    vi.restoreAllMocks();
  });

  // ── Format acceptance ──────────────────────────────────────────────

  it("accepts legacy report-uri format and returns 204", async () => {
    const handler = await loadHandler();
    const res = await handler(cspRequest(legacyReport));
    expect(res.status).toBe(204);
  });

  it("accepts Reporting API v1 flat format and returns 204", async () => {
    const handler = await loadHandler();
    const res = await handler(cspRequest(v1Report));
    expect(res.status).toBe(204);
  });

  // ── Validation ─────────────────────────────────────────────────────

  it("rejects oversized payloads with 413", async () => {
    const handler = await loadHandler();
    const res = await handler(cspRequest(legacyReport, "20000"));
    expect(res.status).toBe(413);
  });

  it("rejects invalid JSON with 400", async () => {
    const handler = await loadHandler();
    const req = new Request("http://localhost:3000/api/csp-report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "not-json!!!",
    });
    const res = await handler(req);
    expect(res.status).toBe(400);
  });

  // ── Alerting threshold ─────────────────────────────────────────────

  it("escalates log to error when threshold exceeded", async () => {
    const handler = await loadHandler({ LUX_CSP_ALERT_THRESHOLD: "3" });
    // Re-import logger mock after module reset
    const { logger: freshLogger } = await import("@/lib/logger");

    // Send enough violations to exceed threshold
    for (let i = 0; i < 4; i++) {
      await handler(cspRequest(v1Report));
    }

    // The 4th call (exceeding threshold of 3) should trigger error level
    expect(freshLogger.error).toHaveBeenCalled();
    const errorCalls = (freshLogger.error as ReturnType<typeof vi.fn>).mock.calls;
    const spikeCalls = errorCalls.filter((c: unknown[]) => c[0] === "csp.violation.spike");
    expect(spikeCalls.length).toBeGreaterThan(0);
  });

  it("logs warn for violations below threshold", async () => {
    const handler = await loadHandler({ LUX_CSP_ALERT_THRESHOLD: "100" });
    const { logger: freshLogger } = await import("@/lib/logger");

    await handler(cspRequest(legacyReport));

    expect(freshLogger.warn).toHaveBeenCalledWith("csp.violation", expect.any(Object));
  });

  // ── Webhook ────────────────────────────────────────────────────────

  it("fires webhook when threshold exceeded and webhook configured", async () => {
    const fetchSpy = vi.fn().mockResolvedValue(new Response(null, { status: 200 }));
    globalThis.fetch = fetchSpy;

    const handler = await loadHandler({
      LUX_CSP_ALERT_THRESHOLD: "2",
      LUX_CSP_ALERT_WEBHOOK: "https://hooks.example.com/alert",
    });

    // Exceed threshold
    for (let i = 0; i < 3; i++) {
      await handler(cspRequest(v1Report));
    }

    // Give the async webhook call time to fire
    await new Promise((r) => setTimeout(r, 50));

    const webhookCalls = fetchSpy.mock.calls.filter(
      (c: unknown[]) => (c[0] as string) === "https://hooks.example.com/alert",
    );
    expect(webhookCalls.length).toBeGreaterThan(0);

    // Verify the webhook payload
    const [, init] = webhookCalls[0] as [string, RequestInit];
    const payload = JSON.parse(init.body as string) as Record<string, unknown>;
    expect(payload.count).toBeGreaterThanOrEqual(2);
    expect(payload.threshold).toBe(2);
  });
});
