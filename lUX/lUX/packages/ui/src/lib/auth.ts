/**
 * Authentication & authorization utilities for lUX.
 *
 * Auth model:
 *   - **Disabled by default**: If `LUX_API_KEY` is not set, all requests are public.
 *   - **API key mode**: When `LUX_API_KEY` is set, `Authorization: Bearer <key>` is required.
 *   - **RBAC**: Optional `LUX_AUTH_ROLES` maps keys → role (`viewer` | `auditor` | `admin`).
 *     If a key exists in `LUX_API_KEY` but not in the role map, the default role is `viewer`.
 *
 * Roles:
 *   - `viewer`: Read-only access to gallery and proof packages.
 *   - `auditor`: Viewer + comparison mode + integrity deep-dive.
 *   - `admin`: Full access to all modes.
 *
 * This module does NOT use `server-only` because it is imported by middleware,
 * which runs in the Edge runtime (neither pure server nor client).
 */

/** Supported authentication roles, ordered by privilege. */
export type AuthRole = "viewer" | "auditor" | "admin";

/** Result of an authentication check. */
export type AuthResult =
  | { authenticated: true; role: AuthRole; keyId: string }
  | { authenticated: false; reason: string };

/** Privilege hierarchy: higher index = more privilege. */
const ROLE_HIERARCHY: readonly AuthRole[] = ["viewer", "auditor", "admin"] as const;

/**
 * Check whether a given role satisfies a minimum required role.
 *
 * @param actual  The user's role.
 * @param required  The minimum required role.
 * @returns `true` if `actual` has equal or higher privilege.
 */
export function hasRole(actual: AuthRole, required: AuthRole): boolean {
  return ROLE_HIERARCHY.indexOf(actual) >= ROLE_HIERARCHY.indexOf(required);
}

/**
 * Parse the `LUX_AUTH_ROLES` JSON string into a key→role map.
 *
 * Expected format: `{ "<api-key>": "admin", "<other-key>": "viewer" }`
 * If the value is not valid JSON or not an object, returns an empty map.
 *
 * @param raw  Raw `LUX_AUTH_ROLES` env var string.
 * @returns Frozen map of API key → role.
 */
export function parseRoleMap(raw: string | undefined): Readonly<Record<string, AuthRole>> {
  if (!raw?.trim()) return Object.freeze({});
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return Object.freeze({});
    const result: Record<string, AuthRole> = {};
    for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (typeof value === "string" && ROLE_HIERARCHY.includes(value as AuthRole)) {
        result[key] = value as AuthRole;
      }
    }
    return Object.freeze(result);
  } catch {
    return Object.freeze({});
  }
}

/**
 * Paths that are always publicly accessible, regardless of auth configuration.
 * These are infrastructure endpoints used by health checks, monitoring, and error reporting.
 */
const PUBLIC_PATHS = new Set([
  "/api/health",
  "/api/ready",
  "/api/metrics",
  "/api/csp-report",
  "/api/errors",
]);

/**
 * Check whether a given request path is exempt from authentication.
 *
 * @param pathname  The URL pathname.
 * @returns `true` if the path is always public.
 */
export function isPublicPath(pathname: string): boolean {
  return PUBLIC_PATHS.has(pathname);
}

/**
 * Authenticate an incoming request using the `Authorization: Bearer` header.
 *
 * @param authHeader  Value of the `Authorization` header (may be null).
 * @param apiKey  The expected API key (`LUX_API_KEY`). If falsy, auth is disabled.
 * @param roleMap  Key→role mapping from `parseRoleMap()`.
 * @returns Authentication result.
 */
export function authenticateRequest(
  authHeader: string | null,
  apiKey: string | undefined,
  roleMap: Readonly<Record<string, AuthRole>>,
): AuthResult {
  // Auth disabled — treat as admin
  if (!apiKey) {
    return { authenticated: true, role: "admin", keyId: "anonymous" };
  }

  if (!authHeader) {
    return { authenticated: false, reason: "Missing Authorization header" };
  }

  // Extract Bearer token
  const match = authHeader.match(/^Bearer\s+(.+)$/i);
  if (!match) {
    return { authenticated: false, reason: "Invalid Authorization format (expected Bearer)" };
  }

  const token = match[1];

  // Constant-time comparison to prevent timing attacks
  if (!timingSafeEqual(token, apiKey)) {
    return { authenticated: false, reason: "Invalid API key" };
  }

  // Determine role from role map, defaulting to viewer
  const role: AuthRole = roleMap[token] ?? "viewer";
  // Key ID for logging — first 8 chars for traceability without full exposure.
  // Uses ASCII "..." (not Unicode ellipsis) so the value is valid in HTTP headers.
  const keyId = token.slice(0, 8) + "...";

  return { authenticated: true, role, keyId };
}

/**
 * Constant-time string comparison.
 * Prevents timing side-channel attacks by always comparing full length.
 */
function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}
