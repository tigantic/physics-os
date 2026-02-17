import { NextResponse } from "next/server";
import { getProvider } from "@/config/provider";
import { logger } from "@/lib/logger";
import { startTimer, serverTimingHeader } from "@/lib/timing";
import { increment, observe } from "@/lib/metrics";

/**
 * GET /api/packages — List available proof packages.
 *
 * Returns a JSON array of PackageSummary objects with lightweight metadata
 * for each available proof package. Does not load full proof data.
 *
 * Response:
 *   200: { packages: PackageSummary[] }
 *   500: { error: string }
 */
export async function GET(request: Request) {
  const requestId = request.headers.get("x-request-id") ?? "unknown";
  const timer = startTimer("list_packages");

  try {
    const provider = await getProvider();
    const packages = await provider.listPackages();
    const timing = timer.stop();

    increment("lux_http_requests_total", "/api/packages");
    observe("lux_http_duration_ms", timing.durationMs);
    logger.info("api.packages.list", { requestId, count: packages.length, durationMs: timing.durationMs });

    return NextResponse.json(
      { packages },
      {
        status: 200,
        headers: {
          "Cache-Control": "public, s-maxage=60, stale-while-revalidate=120",
          "Server-Timing": serverTimingHeader(timing),
          "X-Request-Id": requestId,
        },
      },
    );
  } catch (err) {
    const timing = timer.stop();
    increment("lux_http_errors_total", "/api/packages");
    logger.error("api.packages.list.error", {
      requestId,
      error: err instanceof Error ? err.message : String(err),
      durationMs: timing.durationMs,
    });

    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Failed to list packages" },
      { status: 500, headers: { "X-Request-Id": requestId } },
    );
  }
}

export const runtime = "nodejs";
