import { describe, it, expect, vi, beforeEach } from "vitest";

// Minimal mock of NextResponse and NextRequest for middleware testing.
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
  },
}));

// Import after mock setup
import { middleware } from "@/middleware";

function fakeRequest(url = "http://localhost:3000/gallery"): Parameters<typeof middleware>[0] {
  return {
    headers: new Headers(),
    nextUrl: new URL(url),
  } as unknown as Parameters<typeof middleware>[0];
}

describe("middleware", () => {
  beforeEach(() => {
    setMock.mockClear();
    getMock.mockClear();
  });

  it("sets Content-Security-Policy header", () => {
    middleware(fakeRequest());
    const cspCall = setMock.mock.calls.find((c: string[]) => c[0] === "Content-Security-Policy");
    expect(cspCall).toBeDefined();
    expect(cspCall![1]).toContain("default-src 'self'");
    expect(cspCall![1]).toContain("script-src 'self'");
    expect(cspCall![1]).toContain("nonce-");
    expect(cspCall![1]).toContain("'strict-dynamic'");
    expect(cspCall![1]).toContain("frame-ancestors 'none'");
  });

  it("sets X-Content-Type-Options nosniff", () => {
    middleware(fakeRequest());
    const call = setMock.mock.calls.find((c: string[]) => c[0] === "X-Content-Type-Options");
    expect(call).toBeDefined();
    expect(call![1]).toBe("nosniff");
  });

  it("sets X-Frame-Options DENY", () => {
    middleware(fakeRequest());
    const call = setMock.mock.calls.find((c: string[]) => c[0] === "X-Frame-Options");
    expect(call).toBeDefined();
    expect(call![1]).toBe("DENY");
  });

  it("sets Referrer-Policy", () => {
    middleware(fakeRequest());
    const call = setMock.mock.calls.find((c: string[]) => c[0] === "Referrer-Policy");
    expect(call).toBeDefined();
    expect(call![1]).toBe("strict-origin-when-cross-origin");
  });

  it("sets Permissions-Policy", () => {
    middleware(fakeRequest());
    const call = setMock.mock.calls.find((c: string[]) => c[0] === "Permissions-Policy");
    expect(call).toBeDefined();
    expect(call![1]).toContain("camera=()");
    expect(call![1]).toContain("microphone=()");
  });

  it("sets Strict-Transport-Security (HSTS)", () => {
    middleware(fakeRequest());
    const call = setMock.mock.calls.find((c: string[]) => c[0] === "Strict-Transport-Security");
    expect(call).toBeDefined();
    expect(call![1]).toContain("max-age=63072000");
    expect(call![1]).toContain("includeSubDomains");
    expect(call![1]).toContain("preload");
  });

  it("sets Cross-Origin-Opener-Policy", () => {
    middleware(fakeRequest());
    const call = setMock.mock.calls.find((c: string[]) => c[0] === "Cross-Origin-Opener-Policy");
    expect(call).toBeDefined();
    expect(call![1]).toBe("same-origin");
  });

  it("sets X-DNS-Prefetch-Control", () => {
    middleware(fakeRequest());
    const call = setMock.mock.calls.find((c: string[]) => c[0] === "X-DNS-Prefetch-Control");
    expect(call).toBeDefined();
    expect(call![1]).toBe("off");
  });

  it("generates unique nonce per request", () => {
    middleware(fakeRequest());
    const csp1 = setMock.mock.calls.find((c: string[]) => c[0] === "Content-Security-Policy")![1] as string;
    setMock.mockClear();
    middleware(fakeRequest());
    const csp2 = setMock.mock.calls.find((c: string[]) => c[0] === "Content-Security-Policy")![1] as string;
    // Extract nonce values
    const nonce1 = csp1.match(/nonce-([A-Za-z0-9+/=]+)/)?.[1];
    const nonce2 = csp2.match(/nonce-([A-Za-z0-9+/=]+)/)?.[1];
    expect(nonce1).toBeDefined();
    expect(nonce2).toBeDefined();
    expect(nonce1).not.toBe(nonce2);
  });
});
