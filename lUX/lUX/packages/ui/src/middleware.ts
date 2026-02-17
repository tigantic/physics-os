import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { authenticateRequest, hasRole, isPublicPath, parseRoleMap } from "@/lib/auth";
import type { AuthRole } from "@/lib/auth";

// ── Auth configuration (read once at module level) ─────────────────────────────
const LUX_API_KEY = process.env.LUX_API_KEY?.trim() || undefined;
const ROLE_MAP = parseRoleMap(process.env.LUX_AUTH_ROLES);

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
export function middleware(request: NextRequest) {
  const requestId = request.headers.get("x-request-id") ?? crypto.randomUUID();
  const pathname = request.nextUrl.pathname;

  // ── Auth gate ────────────────────────────────────────────────────────────────
  let authRole: string | undefined;
  let authKeyId: string | undefined;

  if (LUX_API_KEY && !isPublicPath(pathname)) {
    const authResult = authenticateRequest(
      request.headers.get("authorization"),
      LUX_API_KEY,
      ROLE_MAP,
    );

    if (!authResult.authenticated) {
      return NextResponse.json(
        { error: authResult.reason },
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
    if (!hasRole(authResult.role, requiredRole)) {
      return NextResponse.json(
        { error: `Insufficient permissions: requires ${requiredRole}, have ${authResult.role}` },
        {
          status: 403,
          headers: { "X-Request-Id": requestId },
        },
      );
    }

    // Propagate auth context downstream via headers (set on requestHeaders below)
    authRole = authResult.role;
    authKeyId = authResult.keyId;
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
  response.headers.set(
    "Reporting-Endpoints",
    `csp-endpoint="/api/csp-report"`,
  );

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
