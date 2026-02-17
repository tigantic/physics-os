import { describe, it, expect, vi, beforeAll, afterAll, afterEach } from "vitest";

/**
 * Tests for the JWT session management module (session.ts).
 *
 * Node 20 provides the Web Crypto API globally, so no polyfill is needed.
 * We set `LUX_SESSION_SECRET` before importing the module so the
 * module-level constant picks it up.
 */

const TEST_SECRET = "test-secret-key-that-is-at-least-32-chars-long!!";

describe("session", () => {
  const originalEnv = { ...process.env };

  let isSessionEnabled: typeof import("@/lib/session").isSessionEnabled;
  let createSessionToken: typeof import("@/lib/session").createSessionToken;
  let verifySessionToken: typeof import("@/lib/session").verifySessionToken;
  let sessionCookie: typeof import("@/lib/session").sessionCookie;
  let clearSessionCookie: typeof import("@/lib/session").clearSessionCookie;
  let extractSessionToken: typeof import("@/lib/session").extractSessionToken;
  let invalidateSession: typeof import("@/lib/session").invalidateSession;
  let COOKIE_NAME: string;
  let SESSION_TTL_SEC: number;

  beforeAll(async () => {
    vi.resetModules();
    process.env.LUX_SESSION_SECRET = TEST_SECRET;
    process.env.LUX_SESSION_TTL_SEC = "3600";

    const mod = await import("@/lib/session");
    isSessionEnabled = mod.isSessionEnabled;
    createSessionToken = mod.createSessionToken;
    verifySessionToken = mod.verifySessionToken;
    sessionCookie = mod.sessionCookie;
    clearSessionCookie = mod.clearSessionCookie;
    extractSessionToken = mod.extractSessionToken;
    invalidateSession = mod.invalidateSession;
    COOKIE_NAME = mod.COOKIE_NAME;
    SESSION_TTL_SEC = mod.SESSION_TTL_SEC;
  });

  afterAll(() => {
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // ── isSessionEnabled ─────────────────────────────────────────────

  describe("isSessionEnabled", () => {
    it("returns true when secret is >= 32 chars", () => {
      expect(isSessionEnabled()).toBe(true);
    });
  });

  // ── Token lifecycle ──────────────────────────────────────────────

  describe("createSessionToken + verifySessionToken", () => {
    it("creates a valid three-part JWT", async () => {
      const token = await createSessionToken("test1234", "viewer");
      const parts = token.split(".");
      expect(parts.length).toBe(3);
      // Header should be base64url-encoded JSON
      const header = JSON.parse(atob(parts[0].replace(/-/g, "+").replace(/_/g, "/")));
      expect(header.alg).toBe("HS256");
      expect(header.typ).toBe("JWT");
    });

    it("round-trips: verify succeeds for a freshly-created token", async () => {
      const token = await createSessionToken("key12345", "admin");
      const result = await verifySessionToken(token);
      expect(result.valid).toBe(true);
      if (result.valid) {
        expect(result.claims.sub).toBe("key12345");
        expect(result.claims.role).toBe("admin");
        expect(result.claims.jti).toBeTruthy();
        expect(result.claims.iat).toBeGreaterThan(0);
        expect(result.claims.exp).toBeGreaterThan(result.claims.iat);
        expect(result.shouldRefresh).toBe(false);
      }
    });

    it("rejects malformed token (wrong number of parts)", async () => {
      const result = await verifySessionToken("only.two");
      expect(result.valid).toBe(false);
      if (!result.valid) {
        expect(result.reason).toBe("Malformed token");
      }
    });

    it("rejects token with tampered payload", async () => {
      const token = await createSessionToken("key12345", "viewer");
      const parts = token.split(".");
      // Tamper with the payload
      parts[1] = parts[1] + "X";
      const tampered = parts.join(".");
      const result = await verifySessionToken(tampered);
      expect(result.valid).toBe(false);
      if (!result.valid) {
        expect(result.reason).toMatch(/invalid|signature|payload/i);
      }
    });

    it("rejects token with tampered signature", async () => {
      const token = await createSessionToken("key12345", "viewer");
      const parts = token.split(".");
      // Tamper with the signature
      parts[2] = "aaaaaaaaaaaaaaaa";
      const tampered = parts.join(".");
      const result = await verifySessionToken(tampered);
      expect(result.valid).toBe(false);
      if (!result.valid) {
        expect(result.reason).toBe("Invalid signature");
      }
    });

    it("rejects expired token", async () => {
      // Create token, then fake time to after expiry
      const token = await createSessionToken("expiry", "viewer");
      const originalNow = Date.now;
      Date.now = () => originalNow() + (SESSION_TTL_SEC + 1) * 1000;
      try {
        const result = await verifySessionToken(token);
        expect(result.valid).toBe(false);
        if (!result.valid) {
          expect(result.reason).toBe("Token expired");
        }
      } finally {
        Date.now = originalNow;
      }
    });

    it("sets shouldRefresh when token is in final 25% of TTL", async () => {
      const token = await createSessionToken("refresh", "viewer");
      // Jump to 80% through TTL
      const originalNow = Date.now;
      Date.now = () => originalNow() + SESSION_TTL_SEC * 0.8 * 1000;
      try {
        const result = await verifySessionToken(token);
        expect(result.valid).toBe(true);
        if (result.valid) {
          expect(result.shouldRefresh).toBe(true);
        }
      } finally {
        Date.now = originalNow;
      }
    });

    it("each token has a unique jti", async () => {
      const [t1, t2] = await Promise.all([
        createSessionToken("a", "viewer"),
        createSessionToken("b", "viewer"),
      ]);
      const r1 = await verifySessionToken(t1);
      const r2 = await verifySessionToken(t2);
      expect(r1.valid && r2.valid).toBe(true);
      if (r1.valid && r2.valid) {
        expect(r1.claims.jti).not.toBe(r2.claims.jti);
      }
    });
  });

  // ── Revocation ───────────────────────────────────────────────────

  describe("invalidateSession", () => {
    it("revokes a token by jti", async () => {
      const token = await createSessionToken("revoke", "viewer");
      const result1 = await verifySessionToken(token);
      expect(result1.valid).toBe(true);

      if (result1.valid) {
        invalidateSession(result1.claims.jti);
        const result2 = await verifySessionToken(token);
        expect(result2.valid).toBe(false);
        if (!result2.valid) {
          expect(result2.reason).toBe("Token revoked");
        }
      }
    });
  });

  // ── Cookie helpers ───────────────────────────────────────────────

  describe("sessionCookie", () => {
    it("builds a Set-Cookie header with HttpOnly, Secure, SameSite", () => {
      const cookie = sessionCookie("my-jwt-token");
      expect(cookie).toContain(`${COOKIE_NAME}=my-jwt-token`);
      expect(cookie).toContain("HttpOnly");
      expect(cookie).toContain("Secure");
      expect(cookie).toContain("SameSite=Strict");
      expect(cookie).toContain("Path=/");
      expect(cookie).toContain(`Max-Age=${SESSION_TTL_SEC}`);
    });
  });

  describe("clearSessionCookie", () => {
    it("sets Max-Age=0 to clear the cookie", () => {
      const cookie = clearSessionCookie();
      expect(cookie).toContain(`${COOKIE_NAME}=`);
      expect(cookie).toContain("Max-Age=0");
    });
  });

  describe("extractSessionToken", () => {
    it("extracts token from a Cookie header string", () => {
      const header = `other=abc; ${COOKIE_NAME}=jwt-value-here; another=xyz`;
      expect(extractSessionToken(header)).toBe("jwt-value-here");
    });

    it("returns null for missing cookie", () => {
      expect(extractSessionToken("other=abc")).toBeNull();
    });

    it("returns null for null input", () => {
      expect(extractSessionToken(null)).toBeNull();
    });

    it("extracts when session cookie is the first cookie", () => {
      const header = `${COOKIE_NAME}=first-value; other=abc`;
      expect(extractSessionToken(header)).toBe("first-value");
    });
  });
});
