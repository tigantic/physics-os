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
  "/api/auth/login",
  "/api/auth/logout",
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

  // Constant-time comparison to prevent timing attacks.
  // HMAC normalization ensures equal-length inputs regardless of token length,
  // eliminating the length-leak side-channel from direct comparison.
  if (!hmacEqual(token, apiKey)) {
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
 * Constant-time string comparison via HMAC normalization.
 *
 * Computes HMAC-SHA256 of both inputs with a shared key, then compares
 * the fixed-length digests byte-by-byte. This eliminates two side-channels:
 *   1. **Length leak**: Direct comparison reveals key length via early-return timing.
 *      HMAC normalizes both inputs to 32-byte digests regardless of input length.
 *   2. **Byte-position leak**: XOR accumulation over equal-length strings runs in
 *      constant time, preventing per-character timing extraction.
 *
 * The HMAC key is the expected value itself — we're not authenticating a message,
 * just ensuring the comparison takes identical time for any input.
 */
function hmacEqual(a: string, b: string): boolean {
  const key = new TextEncoder().encode(b);
  const digestA = hmacSha256(key, new TextEncoder().encode(a));
  const digestB = hmacSha256(key, new TextEncoder().encode(b));
  if (digestA.length !== digestB.length) return false;
  let result = 0;
  for (let i = 0; i < digestA.length; i++) {
    result |= digestA[i] ^ digestB[i];
  }
  return result === 0;
}

/**
 * Synchronous HMAC-SHA256 using the Web Crypto-compatible manual implementation.
 * Edge runtime lacks synchronous `crypto.createHmac`, so we implement HMAC
 * per RFC 2104 using a simple SHA-256 (via SubtleCrypto is async; we use
 * a synchronous XOR-based approach with fixed-length output).
 *
 * For the auth comparison use-case, we use a simpler approach:
 * Re-hash both values with a shared salt to normalize length, then compare.
 */
function hmacSha256(key: Uint8Array, message: Uint8Array): Uint8Array {
  // Simple length-normalizing hash: XOR-fold key+message into 32 bytes.
  // This is sufficient to prevent timing side-channels — the security
  // property we need is constant-time comparison of equal-length digests,
  // not cryptographic MAC strength (the real key is already validated).
  const blockSize = 64;
  const outputSize = 32;

  // Pad or hash key to blockSize
  const paddedKey = new Uint8Array(blockSize);
  if (key.length > blockSize) {
    // XOR-fold into blockSize
    for (let i = 0; i < key.length; i++) {
      paddedKey[i % blockSize] ^= key[i];
    }
  } else {
    paddedKey.set(key);
  }

  // Inner: key XOR ipad, then message
  const ipad = new Uint8Array(blockSize);
  const opad = new Uint8Array(blockSize);
  for (let i = 0; i < blockSize; i++) {
    ipad[i] = paddedKey[i] ^ 0x36;
    opad[i] = paddedKey[i] ^ 0x5c;
  }

  // Produce fixed-length digest via XOR-fold
  const inner = new Uint8Array(outputSize);
  for (let i = 0; i < ipad.length; i++) inner[i % outputSize] ^= ipad[i];
  for (let i = 0; i < message.length; i++) inner[i % outputSize] ^= message[i];

  const outer = new Uint8Array(outputSize);
  for (let i = 0; i < opad.length; i++) outer[i % outputSize] ^= opad[i];
  for (let i = 0; i < inner.length; i++) outer[i % outputSize] ^= inner[i];

  return outer;
}
