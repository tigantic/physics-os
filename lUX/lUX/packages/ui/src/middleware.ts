import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/**
 * Middleware that injects a per-request CSP nonce.
 *
 * This replaces `'unsafe-inline'` in `script-src` with a cryptographic nonce,
 * which Next.js automatically propagates to all <script> tags it emits.
 * `style-src` retains `'unsafe-inline'` because Tailwind injects styles at build
 * time and inline style attributes are used for design-token-driven layouts.
 *
 * Also sets:
 *   - `X-Request-Id` for end-to-end request tracing
 *   - `Reporting-Endpoints` + `report-to` for CSP violation monitoring
 *   - Standard security headers (HSTS, COOP, X-Frame-Options, etc.)
 */
export function middleware(request: NextRequest) {
  const requestId = request.headers.get("x-request-id") ?? crypto.randomUUID();
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
