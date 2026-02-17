/**
 * Session management for lUX using signed JWT tokens in httpOnly cookies.
 *
 * Uses HMAC-SHA256 (HS256) via the Web Crypto API for token signing,
 * which is async but fully supported in both Node.js and Edge Runtime.
 *
 * Tokens carry `{ sub, role, iat, exp }` claims with configurable TTL.
 * Refresh-before-expiry allows seamless extension without full re-auth.
 *
 * The signing key is derived from `LUX_SESSION_SECRET` (required for
 * session mode). If unset, session features are disabled.
 */

import type { AuthRole } from "./auth";

/* ── Configuration ─────────────────────────────────────────────────── */

const SESSION_SECRET = process.env.LUX_SESSION_SECRET?.trim() ?? "";
const SESSION_TTL_SEC = Number(process.env.LUX_SESSION_TTL_SEC) || 3600; // 1 hour
const SESSION_REFRESH_WINDOW_SEC = Math.floor(SESSION_TTL_SEC / 4); // Refresh in final 25%
const COOKIE_NAME = "__lux_session";

/** Whether session management is available (secret configured). */
export function isSessionEnabled(): boolean {
  return SESSION_SECRET.length >= 32;
}

/* ── In-memory revocation set ──────────────────────────────────────── */

/**
 * Revoked session IDs. In a multi-instance deployment this should be
 * replaced with a shared store (Redis, etc.). For single-instance
 * standalone deployments, an in-memory Set suffices.
 */
const revokedSessions = new Set<string>();
const MAX_REVOKED = 10_000;

function revokeSession(jti: string): void {
  if (revokedSessions.size >= MAX_REVOKED) {
    // Evict oldest (insertion order)
    const first = revokedSessions.values().next().value;
    if (first !== undefined) revokedSessions.delete(first);
  }
  revokedSessions.add(jti);
}

function isRevoked(jti: string): boolean {
  return revokedSessions.has(jti);
}

/* ── Base64url helpers ─────────────────────────────────────────────── */

function base64urlEncode(data: Uint8Array): string {
  const binStr = Array.from(data, (b) => String.fromCharCode(b)).join("");
  return btoa(binStr).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64urlDecode(str: string): Uint8Array {
  const padded = str.replace(/-/g, "+").replace(/_/g, "/");
  const binStr = atob(padded);
  return Uint8Array.from(binStr, (c) => c.charCodeAt(0));
}

function textToBase64url(text: string): string {
  return base64urlEncode(new TextEncoder().encode(text));
}

/* ── HMAC-SHA256 via Web Crypto ────────────────────────────────────── */

async function getSigningKey(): Promise<CryptoKey> {
  const raw = new TextEncoder().encode(SESSION_SECRET);
  return crypto.subtle.importKey("raw", raw, { name: "HMAC", hash: "SHA-256" }, false, [
    "sign",
    "verify",
  ]);
}

async function sign(payload: string): Promise<string> {
  const key = await getSigningKey();
  const data = new TextEncoder().encode(payload);
  const sig = await crypto.subtle.sign("HMAC", key, data);
  return base64urlEncode(new Uint8Array(sig));
}

async function verify(payload: string, signature: string): Promise<boolean> {
  const key = await getSigningKey();
  const data = new TextEncoder().encode(payload);
  const sig = base64urlDecode(signature);
  return crypto.subtle.verify("HMAC", key, sig.buffer as ArrayBuffer, data);
}

/* ── JWT claims ────────────────────────────────────────────────────── */

export interface SessionClaims {
  /** Subject: key identifier (first 8 chars) */
  sub: string;
  /** Role assigned at login */
  role: AuthRole;
  /** Issued-at (epoch seconds) */
  iat: number;
  /** Expiry (epoch seconds) */
  exp: number;
  /** JWT ID for revocation tracking */
  jti: string;
}

/* ── Token creation ────────────────────────────────────────────────── */

/**
 * Create a signed JWT session token.
 *
 * @param keyId  Short identifier for the API key (e.g. first 8 chars)
 * @param role   Authenticated role
 * @returns Signed JWT string
 */
export async function createSessionToken(keyId: string, role: AuthRole): Promise<string> {
  const now = Math.floor(Date.now() / 1000);
  const claims: SessionClaims = {
    sub: keyId,
    role,
    iat: now,
    exp: now + SESSION_TTL_SEC,
    jti: crypto.randomUUID(),
  };

  const header = textToBase64url(JSON.stringify({ alg: "HS256", typ: "JWT" }));
  const payload = textToBase64url(JSON.stringify(claims));
  const sigInput = `${header}.${payload}`;
  const signature = await sign(sigInput);

  return `${sigInput}.${signature}`;
}

/* ── Token verification ────────────────────────────────────────────── */

export interface VerifyResult {
  valid: true;
  claims: SessionClaims;
  /** True if the token is within the refresh window (final 25% of TTL). */
  shouldRefresh: boolean;
}

export interface VerifyError {
  valid: false;
  reason: string;
}

/**
 * Verify and decode a JWT session token.
 *
 * Checks:
 *   1. Structure (three-part JWT)
 *   2. HMAC-SHA256 signature
 *   3. Expiry (`exp` claim)
 *   4. Revocation set
 */
export async function verifySessionToken(token: string): Promise<VerifyResult | VerifyError> {
  const parts = token.split(".");
  if (parts.length !== 3) {
    return { valid: false, reason: "Malformed token" };
  }

  const [header, payload, signature] = parts;
  const sigInput = `${header}.${payload}`;

  const isValid = await verify(sigInput, signature);
  if (!isValid) {
    return { valid: false, reason: "Invalid signature" };
  }

  let claims: SessionClaims;
  try {
    const decoded = new TextDecoder().decode(base64urlDecode(payload));
    claims = JSON.parse(decoded) as SessionClaims;
  } catch {
    return { valid: false, reason: "Invalid payload" };
  }

  const now = Math.floor(Date.now() / 1000);
  if (claims.exp <= now) {
    return { valid: false, reason: "Token expired" };
  }

  if (isRevoked(claims.jti)) {
    return { valid: false, reason: "Token revoked" };
  }

  const shouldRefresh = claims.exp - now <= SESSION_REFRESH_WINDOW_SEC;
  return { valid: true, claims, shouldRefresh };
}

/* ── Cookie helpers ────────────────────────────────────────────────── */

/** Build a `Set-Cookie` header value for a new session. */
export function sessionCookie(token: string): string {
  const maxAge = SESSION_TTL_SEC;
  return [
    `${COOKIE_NAME}=${token}`,
    "Path=/",
    "HttpOnly",
    "Secure",
    "SameSite=Strict",
    `Max-Age=${maxAge}`,
  ].join("; ");
}

/** Build a `Set-Cookie` header value that clears the session. */
export function clearSessionCookie(): string {
  return [
    `${COOKIE_NAME}=`,
    "Path=/",
    "HttpOnly",
    "Secure",
    "SameSite=Strict",
    "Max-Age=0",
  ].join("; ");
}

/** Extract the session token from a `Cookie` header string. */
export function extractSessionToken(cookieHeader: string | null): string | null {
  if (!cookieHeader) return null;
  const match = cookieHeader.match(new RegExp(`(?:^|;\\s*)${COOKIE_NAME}=([^;]+)`));
  return match?.[1] ?? null;
}

/** Revoke a session by its JWT ID. */
export function invalidateSession(jti: string): void {
  revokeSession(jti);
}

export { COOKIE_NAME, SESSION_TTL_SEC };
