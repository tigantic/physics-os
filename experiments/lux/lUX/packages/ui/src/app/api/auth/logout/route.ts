import { NextResponse } from "next/server";
import {
  isSessionEnabled,
  extractSessionToken,
  verifySessionToken,
  invalidateSession,
  clearSessionCookie,
} from "@/lib/session";

/**
 * POST /api/auth/logout — Invalidate the current session and clear the cookie.
 *
 * If a valid session cookie is present, the session's JWT ID is added to the
 * in-memory revocation set so the token cannot be reused even before expiry.
 * The httpOnly cookie is always cleared regardless of token validity.
 */
export async function POST(request: Request) {
  if (!isSessionEnabled()) {
    return NextResponse.json(
      { error: "Session management is not configured" },
      { status: 501 },
    );
  }

  const cookieHeader = request.headers.get("cookie");
  const token = extractSessionToken(cookieHeader);

  if (token) {
    const result = await verifySessionToken(token);
    if (result.valid) {
      invalidateSession(result.claims.jti);
    }
  }

  return NextResponse.json(
    { ok: true },
    {
      status: 200,
      headers: { "Set-Cookie": clearSessionCookie() },
    },
  );
}

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
