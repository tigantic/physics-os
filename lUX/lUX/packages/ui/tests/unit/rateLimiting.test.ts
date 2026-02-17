import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

/**
 * Tests for the rate-limiting subsystem of middleware.ts.
 *
 * The rate limiter uses a module-level Map that persists across dynamic
 * imports, so each test uses unique client IPs to avoid cross-contamination.
 */

// ── Shared state ───────────────────────────────────────────────────────────────

const setMock = vi.fn();
const getMock = vi.fn();

vi.mock("next/server", () => ({
  NextResponse: {
    next: () => ({
      headers: {
        set: setMock,
        get: getMock,
      },
    }),
    json: (body: unknown, init?: { status?: number; headers?: Record<string, string> }) => ({
      type: "json",
      body,
      status: init?.status ?? 200,
      headers: new Map(Object.entries(init?.headers ?? {})),
    }),
  },
}));

vi.mock("@/lib/session", () => ({
  isSessionEnabled: () => false,
  extractSessionToken: () => null,
  verifySessionToken: async () => ({ valid: false, reason: "not configured" }),
  createSessionToken: async () => "",
  sessionCookie: () => "",
}));

type MiddlewareFn = (req: unknown) => Promise<unknown> | unknown;

let testIpCounter = 0;
function uniqueIp(): string {
  testIpCounter++;
  return `10.${Math.floor(testIpCounter / 256) % 256}.${testIpCounter % 256}.${Math.floor(testIpCounter / 65536) % 256}`;
}

function fakeRequest(
  url = "http://localhost:3000/gallery",
  headers: Record<string, string> = {},
  ip = "127.0.0.1",
): unknown {
  const h = new Headers(headers);
  return {
    headers: h,
    nextUrl: new URL(url),
    ip,
  };
}

async function loadMiddleware(
  env: Record<string, string | undefined> = {},
): Promise<MiddlewareFn> {
  vi.resetModules();

  delete process.env.LUX_API_KEY;
  delete process.env.LUX_AUTH_ROLES;

  for (const [key, value] of Object.entries(env)) {
    if (value === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = value;
    }
  }

  const mod = await import("@/middleware");
  return mod.middleware as MiddlewareFn;
}

describe("rate limiting", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    setMock.mockClear();
    getMock.mockClear();
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    vi.restoreAllMocks();
  });

  it("sets X-RateLimit-Limit and X-RateLimit-Remaining headers", async () => {
    const mw = await loadMiddleware({ LUX_RATE_LIMIT_RPM: "100" });
    const ip = uniqueIp();
    await mw(fakeRequest("http://localhost:3000/gallery", {}, ip));
    const limitCall = setMock.mock.calls.find((c: string[]) => c[0] === "X-RateLimit-Limit");
    const remainingCall = setMock.mock.calls.find(
      (c: string[]) => c[0] === "X-RateLimit-Remaining",
    );
    expect(limitCall).toBeDefined();
    expect(limitCall![1]).toBe("100");
    expect(remainingCall).toBeDefined();
  });

  it("returns 429 when rate limit is exceeded", async () => {
    const mw = await loadMiddleware({ LUX_RATE_LIMIT_RPM: "300" });
    const ip = uniqueIp();

    // Fill up the rate limit
    for (let i = 0; i < 300; i++) {
      await mw(fakeRequest("http://localhost:3000/gallery", {}, ip));
    }

    // Next request should be rate-limited
    const result = (await mw(fakeRequest("http://localhost:3000/gallery", {}, ip))) as {
      type: string;
      status: number;
      body: { error: string };
      headers: Map<string, string>;
    };
    expect(result.type).toBe("json");
    expect(result.status).toBe(429);
    expect(result.body.error).toBe("Too many requests");
    expect(result.headers.get("Retry-After")).toBeDefined();
    expect(result.headers.get("X-RateLimit-Remaining")).toBe("0");
  });

  it("treats different IPs independently", async () => {
    const mw = await loadMiddleware({ LUX_RATE_LIMIT_RPM: "300" });
    const ip1 = uniqueIp();
    const ip2 = uniqueIp();

    // Two IPs each make 1 request — neither is rate limited
    const r1 = await mw(fakeRequest("http://localhost:3000/gallery", {}, ip1));
    const r2 = await mw(fakeRequest("http://localhost:3000/gallery", {}, ip2));
    expect((r1 as { type?: string }).type).not.toBe("json");
    expect((r2 as { type?: string }).type).not.toBe("json");
  });

  it("extracts client IP from x-forwarded-for header", async () => {
    const mw = await loadMiddleware({ LUX_RATE_LIMIT_RPM: "300" });

    // Use x-forwarded-for; the leftmost IP should be the client IP
    const ip1 = uniqueIp();
    const ip2 = uniqueIp();

    // Request with x-forwarded-for chain — leftmost IP is the client
    await mw(
      fakeRequest("http://localhost:3000/gallery", {
        "x-forwarded-for": `${ip1}, 192.168.1.1`,
      }),
    );

    // Another request from a different client IP
    setMock.mockClear();
    const result = await mw(
      fakeRequest("http://localhost:3000/gallery", {
        "x-forwarded-for": `${ip2}, 192.168.1.1`,
      }),
    );
    // Different real client IPs → both should succeed
    expect((result as { type?: string }).type).not.toBe("json");
  });

  it("uses leftmost IP from x-forwarded-for chain", async () => {
    const mw = await loadMiddleware({ LUX_RATE_LIMIT_RPM: "300" });
    const clientIp = uniqueIp();

    // Fill up the rate limit for this specific client IP
    for (let i = 0; i < 300; i++) {
      await mw(
        fakeRequest("http://localhost:3000/gallery", {
          "x-forwarded-for": `${clientIp}, 10.0.0.1, 172.16.0.1`,
        }),
      );
    }

    // 301st request from the same leftmost IP should be rate-limited
    const result = (await mw(
      fakeRequest("http://localhost:3000/gallery", {
        "x-forwarded-for": `${clientIp}, 10.0.0.2`,
      }),
    )) as { type: string; status: number };
    expect(result.type).toBe("json");
    expect(result.status).toBe(429);
  });

  it("decrements remaining count on successive requests", async () => {
    const mw = await loadMiddleware({ LUX_RATE_LIMIT_RPM: "300" });
    const ip = uniqueIp();

    // First request
    await mw(fakeRequest("http://localhost:3000/gallery", {}, ip));
    let remainingCall = setMock.mock.calls.find(
      (c: string[]) => c[0] === "X-RateLimit-Remaining",
    );
    expect(remainingCall![1]).toBe("299");

    // Second request
    setMock.mockClear();
    await mw(fakeRequest("http://localhost:3000/gallery", {}, ip));
    remainingCall = setMock.mock.calls.find(
      (c: string[]) => c[0] === "X-RateLimit-Remaining",
    );
    expect(remainingCall![1]).toBe("298");
  });
});
