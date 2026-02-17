import { describe, it, expect } from "vitest";
import {
  authenticateRequest,
  hasRole,
  isPublicPath,
  parseRoleMap,
} from "@/lib/auth";
import type { AuthRole } from "@/lib/auth";

// ── parseRoleMap ───────────────────────────────────────────────────────────────

describe("parseRoleMap", () => {
  it("returns empty map for undefined input", () => {
    expect(parseRoleMap(undefined)).toEqual({});
  });

  it("returns empty map for empty string", () => {
    expect(parseRoleMap("")).toEqual({});
  });

  it("returns empty map for invalid JSON", () => {
    expect(parseRoleMap("{broken")).toEqual({});
  });

  it("returns empty map for non-object JSON", () => {
    expect(parseRoleMap("[1,2,3]")).toEqual({});
  });

  it("parses valid role map", () => {
    const map = parseRoleMap('{"key-a": "admin", "key-b": "viewer"}');
    expect(map).toEqual({ "key-a": "admin", "key-b": "viewer" });
  });

  it("ignores invalid role values", () => {
    const map = parseRoleMap('{"a": "admin", "b": "superuser", "c": 42}');
    expect(map).toEqual({ a: "admin" });
  });

  it("returns frozen object", () => {
    const map = parseRoleMap('{"key-a": "admin"}');
    expect(Object.isFrozen(map)).toBe(true);
  });
});

// ── hasRole ────────────────────────────────────────────────────────────────────

describe("hasRole", () => {
  it("admin satisfies all roles", () => {
    expect(hasRole("admin", "viewer")).toBe(true);
    expect(hasRole("admin", "auditor")).toBe(true);
    expect(hasRole("admin", "admin")).toBe(true);
  });

  it("auditor satisfies viewer and auditor", () => {
    expect(hasRole("auditor", "viewer")).toBe(true);
    expect(hasRole("auditor", "auditor")).toBe(true);
    expect(hasRole("auditor", "admin")).toBe(false);
  });

  it("viewer only satisfies viewer", () => {
    expect(hasRole("viewer", "viewer")).toBe(true);
    expect(hasRole("viewer", "auditor")).toBe(false);
    expect(hasRole("viewer", "admin")).toBe(false);
  });
});

// ── isPublicPath ───────────────────────────────────────────────────────────────

describe("isPublicPath", () => {
  const publicPaths = [
    "/api/health",
    "/api/ready",
    "/api/metrics",
    "/api/csp-report",
    "/api/errors",
  ];

  it.each(publicPaths)("%s is public", (path) => {
    expect(isPublicPath(path)).toBe(true);
  });

  const protectedPaths = [
    "/gallery",
    "/api/packages",
    "/api/domains/test",
    "/",
  ];

  it.each(protectedPaths)("%s is not public", (path) => {
    expect(isPublicPath(path)).toBe(false);
  });
});

// ── authenticateRequest ────────────────────────────────────────────────────────

describe("authenticateRequest", () => {
  const emptyRoles: Readonly<Record<string, AuthRole>> = Object.freeze({});

  it("returns admin when auth is disabled (no API key)", () => {
    const result = authenticateRequest(null, undefined, emptyRoles);
    expect(result).toEqual({ authenticated: true, role: "admin", keyId: "anonymous" });
  });

  it("returns unauthenticated when no Authorization header", () => {
    const result = authenticateRequest(null, "secret-key-123", emptyRoles);
    expect(result).toEqual({
      authenticated: false,
      reason: "Missing Authorization header",
    });
  });

  it("returns unauthenticated for non-Bearer format", () => {
    const result = authenticateRequest("Basic abc123", "secret-key-123", emptyRoles);
    expect(result).toEqual({
      authenticated: false,
      reason: "Invalid Authorization format (expected Bearer)",
    });
  });

  it("returns unauthenticated for wrong key", () => {
    const result = authenticateRequest("Bearer wrong-key", "secret-key-123", emptyRoles);
    expect(result).toEqual({ authenticated: false, reason: "Invalid API key" });
  });

  it("returns authenticated with viewer role by default", () => {
    const result = authenticateRequest("Bearer secret-key-123", "secret-key-123", emptyRoles);
    expect(result.authenticated).toBe(true);
    if (result.authenticated) {
      expect(result.role).toBe("viewer");
      expect(result.keyId).toBe("secret-k...");
    }
  });

  it("returns role from role map", () => {
    const roles = Object.freeze({ "secret-key-123": "admin" as AuthRole });
    const result = authenticateRequest("Bearer secret-key-123", "secret-key-123", roles);
    expect(result.authenticated).toBe(true);
    if (result.authenticated) {
      expect(result.role).toBe("admin");
    }
  });

  it("rejects keys with different lengths (timing-safe)", () => {
    const result = authenticateRequest("Bearer short", "much-longer-key", emptyRoles);
    expect(result).toEqual({ authenticated: false, reason: "Invalid API key" });
  });

  it("is case-insensitive for Bearer prefix", () => {
    const result = authenticateRequest("bearer secret-key-123", "secret-key-123", emptyRoles);
    expect(result.authenticated).toBe(true);
  });
});
