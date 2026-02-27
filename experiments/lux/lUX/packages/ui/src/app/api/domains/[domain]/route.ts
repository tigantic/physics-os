import { NextResponse } from "next/server";
import { z } from "zod";
import { getProvider } from "@/config/provider";
import { ProviderNotFoundError } from "@luxury/core";
import { logger } from "@/lib/logger";
import { startTimer, serverTimingHeader } from "@/lib/timing";
import { increment, observe } from "@/lib/metrics";
import { computeETag, isNotModified } from "@/lib/etag";

const ParamsSchema = z.object({
  domain: z
    .string()
    .min(1)
    .regex(/^[a-zA-Z0-9._-]+$/, "Invalid domain ID"),
});

/**
 * GET /api/domains/[domain] — Load a domain pack.
 *
 * Accepts either a TPC domain_id (e.g. "II.2") or a direct pack ID
 * (e.g. "com.physics.euler_3d"). The provider resolves via manifest
 * lookup with fallback to direct ID match.
 *
 * Response:
 *   200: DomainPack
 *   400: { error: string } — invalid domain ID
 *   404: { error: string } — domain pack not found
 *   500: { error: string } — server error
 */
export async function GET(request: Request, { params }: { params: { domain: string } }) {
  const requestId = request.headers.get("x-request-id") ?? "unknown";
  const parsed = ParamsSchema.safeParse(params);
  if (!parsed.success) {
    increment("lux_http_errors_total", "/api/domains/[domain]");
    return NextResponse.json(
      { error: `Invalid domain ID: ${parsed.error.issues.map((i) => i.message).join(", ")}` },
      { status: 400, headers: { "X-Request-Id": requestId } },
    );
  }

  const timer = startTimer("load_domain");

  try {
    const provider = await getProvider();
    const domainPack = await provider.loadDomainPack(parsed.data.domain);
    const timing = timer.stop();

    increment("lux_http_requests_total", "/api/domains/[domain]");
    observe("lux_http_duration_ms", timing.durationMs);
    logger.info("api.domains.load", { requestId, domain: parsed.data.domain, durationMs: timing.durationMs });

    const etag = computeETag(domainPack as unknown as Record<string, unknown>);

    if (isNotModified(request, etag)) {
      return new NextResponse(null, {
        status: 304,
        headers: {
          ETag: etag,
          "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400",
          "X-Request-Id": requestId,
        },
      });
    }

    return NextResponse.json(domainPack, {
      status: 200,
      headers: {
        ETag: etag,
        "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400",
        "Server-Timing": serverTimingHeader(timing),
        "X-Request-Id": requestId,
      },
    });
  } catch (err) {
    const timing = timer.stop();
    const message = err instanceof Error ? err.message : "Failed to load domain pack";
    const status = err instanceof ProviderNotFoundError ? 404 : 500;

    increment("lux_http_errors_total", "/api/domains/[domain]");
    logger.error("api.domains.load.error", {
      requestId,
      domain: parsed.data.domain,
      error: message,
      status,
      durationMs: timing.durationMs,
    });

    return NextResponse.json(
      { error: message },
      { status, headers: { "X-Request-Id": requestId } },
    );
  }
}

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
