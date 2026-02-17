import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { authenticateRequest, hasRole, isPublicPath, parseRoleMap } from "@/lib/auth";
import type { AuthRole } from "@/lib/auth";
import {
  isSessionEnabled,
  extractSessionToken,
  verifySessionToken,
  createSessionToken,
  sessionCookie,
} from "@/lib/session";

// ── Auth configuration (read once at module level) ─────────────────────────────
const LUX_API_KEY = process.env.LUX_API_KEY?.trim() || undefined;
const ROLE_MAP = parseRoleMap(process.env.LUX_AUTH_ROLES);

// ── Rate limiting ──────────────────────────────────────────────────────────────

/**
 * In-memory sliding-window rate limiter.
 * Tracks request timestamps per IP within a 60-second window.
 * Configurable via `LUX_RATE_LIMIT_RPM` (default: 300 req/min).
 *
 * In a multi-instance deployment each worker tracks its own window.
 * For truly shared state a distributed store (Redis, etc.) is required,
 * but per-worker limiting still provides meaningful protection against
 * runaway clients and simple flooding.
 */
const RATE_LIMIT_RPM = Number(process.env.LUX_RATE_LIMIT_RPM) || 300;
const RATE_WINDOW_MS = 60_000;
const RATE_CLEANUP_INTERVAL_MS = 120_000;
const RATE_MAX_TRACKED_IPS = 10_000;

const ipWindows = new Map<string, number[]>();
let lastCleanup = Date.now();

function getClientIp(request: NextRequest): string {
  // x-forwarded-for may contain a chain; take the leftmost (client) IP
  const forwarded = request.headers.get("x-forwarded-for");
  if (forwarded) {
    const first = forwarded.split(",")[0].trim();
    if (first) return first;
  }
  return request.ip ?? "unknown";
}

/**
 * Returns the number of remaining requests in the current window,
 * or -1 if the limit has been exceeded.
 */
function checkRateLimit(ip: string): { remaining: number; retryAfterSec: number } {
  const now = Date.now();

  // Periodic housekeeping to prevent unbounded memory growth
  if (now - lastCleanup > RATE_CLEANUP_INTERVAL_MS) {
    lastCleanup = now;
    const cutoff = now - RATE_WINDOW_MS;
    for (const [key, timestamps] of ipWindows) {
      // Remove stale entries, drop empty buckets
      const pruned = timestamps.filter((t) => t > cutoff);
      if (pruned.length === 0) {
        ipWindows.delete(key);
      } else {
        ipWindows.set(key, pruned);
      }
    }
    // Hard cap: evict oldest entries if too many IPs
    if (ipWindows.size > RATE_MAX_TRACKED_IPS) {
      const excess = ipWindows.size - RATE_MAX_TRACKED_IPS;
      const keys = ipWindows.keys();
      for (let i = 0; i < excess; i++) {
        const next = keys.next();
        if (next.done) break;
        ipWindows.delete(next.value);
      }
    }
  }

  let timestamps = ipWindows.get(ip);
  if (!timestamps) {
    timestamps = [];
    ipWindows.set(ip, timestamps);
  }

  // Evict entries outside the window
  const cutoff = now - RATE_WINDOW_MS;
  while (timestamps.length > 0 && timestamps[0] < cutoff) {
    timestamps.shift();
  }

  if (timestamps.length >= RATE_LIMIT_RPM) {
    // Estimate when the oldest entry falls out of the window
    const oldestInWindow = timestamps[0];
    const retryAfterSec = Math.ceil((oldestInWindow + RATE_WINDOW_MS - now) / 1_000);
    return { remaining: 0, retryAfterSec: Math.max(1, retryAfterSec) };
  }

  timestamps.push(now);
  return { remaining: RATE_LIMIT_RPM - timestamps.length, retryAfterSec: 0 };
}

/**
 * Minimum role required per path prefix.
 * More specific prefixes are checked first.
 * Paths not listed here require `viewer` by default.
 */
const PATH_ROLE_REQUIREMENTS: ReadonlyArray<{ prefix: string; role: AuthRole }> = [
  // Admin-only endpoints could be added here, e.g.:
  // { prefix: "/api/admin", role: "admin" },
  { prefix: "/api/packages", role: "viewer" },
  { prefix: "/api/domains", role: "viewer" },
  { prefix: "/gallery", role: "viewer" },
];

function requiredRoleForPath(pathname: string): AuthRole {
  for (const entry of PATH_ROLE_REQUIREMENTS) {
    if (pathname.startsWith(entry.prefix)) return entry.role;
  }
  return "viewer";
}

/**
 * Middleware that enforces authentication and injects a per-request CSP nonce.
 *
 * Authentication:
 *   - If `LUX_API_KEY` is not set, all requests are public (no auth required).
 *   - If set, `Authorization: Bearer <key>` is required on non-public paths.
 *   - Infrastructure endpoints (/api/health, /api/ready, /api/metrics, /api/csp-report,
 *     /api/errors) are always publicly accessible.
 *
 * Also sets:
 *   - `X-Request-Id` for end-to-end request tracing
 *   - `Reporting-Endpoints` + `report-to` for CSP violation monitoring
 *   - Standard security headers (HSTS, COOP, X-Frame-Options, etc.)
 */
export async function middleware(request: NextRequest) {
  const requestId = request.headers.get("x-request-id") ?? crypto.randomUUID();
  const pathname = request.nextUrl.pathname;

  // ── Rate limiting ────────────────────────────────────────────────────────────
  const clientIp = getClientIp(request);
  const { remaining, retryAfterSec } = checkRateLimit(clientIp);

  if (remaining === 0) {
    return NextResponse.json(
      { error: "Too many requests" },
      {
        status: 429,
        headers: {
          "Retry-After": String(retryAfterSec),
          "X-RateLimit-Limit": String(RATE_LIMIT_RPM),
          "X-RateLimit-Remaining": "0",
          "X-Request-Id": requestId,
        },
      },
    );
  }

  // ── Auth gate ────────────────────────────────────────────────────────────────
  let authRole: string | undefined;
  let authKeyId: string | undefined;
  let refreshedSessionCookie: string | undefined;

  if (LUX_API_KEY && !isPublicPath(pathname)) {
    let authenticated = false;

    // Strategy 1: Bearer token (API key)
    const authHeader = request.headers.get("authorization");
    if (authHeader) {
      const authResult = authenticateRequest(authHeader, LUX_API_KEY, ROLE_MAP);
      if (authResult.authenticated) {
        authenticated = true;
        authRole = authResult.role;
        authKeyId = authResult.keyId;
      }
    }

    // Strategy 2: Session cookie (JWT)
    if (!authenticated && isSessionEnabled()) {
      const cookieHeader = request.headers.get("cookie");
      const token = extractSessionToken(cookieHeader);
      if (token) {
        const sessionResult = await verifySessionToken(token);
        if (sessionResult.valid) {
          authenticated = true;
          authRole = sessionResult.claims.role;
          authKeyId = sessionResult.claims.sub;

          // Transparently refresh tokens nearing expiry
          if (sessionResult.shouldRefresh) {
            const newToken = await createSessionToken(
              sessionResult.claims.sub,
              sessionResult.claims.role,
            );
            refreshedSessionCookie = sessionCookie(newToken);
          }
        }
      }
    }

    if (!authenticated) {
      return NextResponse.json(
        { error: "Authentication required" },
        {
          status: 401,
          headers: {
            "WWW-Authenticate": 'Bearer realm="lUX"',
            "X-Request-Id": requestId,
          },
        },
      );
    }

    const requiredRole = requiredRoleForPath(pathname);
    if (!hasRole(authRole as AuthRole, requiredRole)) {
      return NextResponse.json(
        { error: `Insufficient permissions: requires ${requiredRole}, have ${authRole}` },
        {
          status: 403,
          headers: { "X-Request-Id": requestId },
        },
      );
    }
  }
  const nonce = Buffer.from(crypto.randomUUID()).toString("base64");

  const csp = [
    `default-src 'self'`,
    `script-src 'self' 'nonce-${nonce}' 'strict-dynamic'`,
    `style-src 'self' 'unsafe-inline'`,
    `img-src 'self' data: blob:`,
    `font-src 'self' data:`,
    `connect-src 'self'`,
    `frame-ancestors 'none'`,
    `base-uri 'self'`,
    `form-action 'self'`,
    `report-uri /api/csp-report`,
    `report-to csp-endpoint`,
  ].join("; ");

  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-nonce", nonce);
  requestHeaders.set("x-request-id", requestId);
  if (authRole) requestHeaders.set("x-auth-role", authRole);
  if (authKeyId) requestHeaders.set("x-auth-key-id", authKeyId);

  const response = NextResponse.next({ request: { headers: requestHeaders } });
  response.headers.set("Content-Security-Policy", csp);
  response.headers.set("X-Content-Type-Options", "nosniff");
  response.headers.set("X-Frame-Options", "DENY");
  response.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
  response.headers.set("Permissions-Policy", "camera=(), microphone=(), geolocation=()");
  response.headers.set("Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload");
  response.headers.set("X-DNS-Prefetch-Control", "off");
  response.headers.set("Cross-Origin-Opener-Policy", "same-origin");
  response.headers.set("X-Request-Id", requestId);
  response.headers.set("X-RateLimit-Limit", String(RATE_LIMIT_RPM));
  response.headers.set("X-RateLimit-Remaining", String(remaining));
  response.headers.set(
    "Reporting-Endpoints",
    `csp-endpoint="/api/csp-report"`,
  );

  // Attach refreshed session cookie if the token was near expiry
  if (refreshedSessionCookie) {
    response.headers.set("Set-Cookie", refreshedSessionCookie);
  }

  return response;
}

export const config = {
  matcher: [
    /*
     * Match all request paths except:
     * - _next/static (static files)
     * - _next/image (image optimization)
     * - favicon.ico, robots.txt, sitemap.xml
     */
    {
      source: "/((?!_next/static|_next/image|favicon\\.ico|robots\\.txt|sitemap\\.xml).*)",
      missing: [{ type: "header", key: "next-router-prefetch" }],
    },
  ],
};
