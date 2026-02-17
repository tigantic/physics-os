import { NextResponse } from "next/server";
import { z } from "zod";
import { getProvider } from "@/config/provider";
import { logger } from "@/lib/logger";
import { startTimer, serverTimingHeader } from "@/lib/timing";
import { increment, observe } from "@/lib/metrics";

const ParamsSchema = z.object({
  id: z
    .string()
    .min(1)
    .regex(/^[a-zA-Z0-9._-]+$/, "Invalid package ID"),
});

/**
 * GET /api/packages/[id] — Load a full proof package by ID.
 *
 * Returns the complete ProofPackage JSON, including verification results.
 *
 * Response:
 *   200: ProofPackage
 *   400: { error: string } — invalid ID
 *   404: { error: string } — package not found
 *   500: { error: string } — server error
 */
export async function GET(request: Request, { params }: { params: { id: string } }) {
  const requestId = request.headers.get("x-request-id") ?? "unknown";
  const parsed = ParamsSchema.safeParse(params);
  if (!parsed.success) {
    increment("lux_http_errors_total", "/api/packages/[id]");
    return NextResponse.json(
      { error: `Invalid package ID: ${parsed.error.issues.map((i) => i.message).join(", ")}` },
      { status: 400, headers: { "X-Request-Id": requestId } },
    );
  }

  const timer = startTimer("load_package");

  try {
    const provider = await getProvider();
    const proof = await provider.loadPackage(parsed.data.id);
    const timing = timer.stop();

    increment("lux_http_requests_total", "/api/packages/[id]");
    observe("lux_http_duration_ms", timing.durationMs);
    logger.info("api.packages.load", { requestId, packageId: parsed.data.id, durationMs: timing.durationMs });

    return NextResponse.json(proof, {
      status: 200,
      headers: {
        "Cache-Control": "public, s-maxage=300, stale-while-revalidate=600",
        "Server-Timing": serverTimingHeader(timing),
        "X-Request-Id": requestId,
      },
    });
  } catch (err) {
    const timing = timer.stop();
    const message = err instanceof Error ? err.message : "Failed to load package";
    const status = message.includes("not found") ? 404 : 500;

    increment("lux_http_errors_total", "/api/packages/[id]");
    logger.error("api.packages.load.error", {
      requestId,
      packageId: parsed.data.id,
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
