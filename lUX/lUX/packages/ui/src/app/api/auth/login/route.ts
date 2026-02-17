import { NextResponse } from "next/server";
import { authenticateRequest, parseRoleMap } from "@/lib/auth";
import {
  isSessionEnabled,
  createSessionToken,
  sessionCookie,
} from "@/lib/session";

const LUX_API_KEY = process.env.LUX_API_KEY?.trim() || undefined;
const ROLE_MAP = parseRoleMap(process.env.LUX_AUTH_ROLES);

/**
 * POST /api/auth/login — Exchange an API key for a session cookie.
 *
 * Request body: `{ "apiKey": "<key>" }`
 * On success: Sets an httpOnly session cookie and returns 200 with
 * `{ role, expiresAt }`. On failure: returns 401.
 *
 * Requires `LUX_SESSION_SECRET` (≥32 chars) to be configured.
 */
export async function POST(request: Request) {
  if (!isSessionEnabled()) {
    return NextResponse.json(
      { error: "Session management is not configured (set LUX_SESSION_SECRET)" },
      { status: 501 },
    );
  }

  if (!LUX_API_KEY) {
    return NextResponse.json(
      { error: "Auth is disabled (LUX_API_KEY not set)" },
      { status: 501 },
    );
  }

  let body: { apiKey?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  if (!body.apiKey || typeof body.apiKey !== "string") {
    return NextResponse.json({ error: "Missing apiKey field" }, { status: 400 });
  }

  const authResult = authenticateRequest(
    `Bearer ${body.apiKey}`,
    LUX_API_KEY,
    ROLE_MAP,
  );

  if (!authResult.authenticated) {
    return NextResponse.json(
      { error: authResult.reason },
      { status: 401, headers: { "WWW-Authenticate": 'Bearer realm="lUX"' } },
    );
  }

  const token = await createSessionToken(authResult.keyId, authResult.role);
  const cookie = sessionCookie(token);

  return NextResponse.json(
    {
      role: authResult.role,
      expiresAt: new Date(
        Date.now() + (Number(process.env.LUX_SESSION_TTL_SEC) || 3600) * 1000,
      ).toISOString(),
    },
    {
      status: 200,
      headers: { "Set-Cookie": cookie },
    },
  );
}

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
