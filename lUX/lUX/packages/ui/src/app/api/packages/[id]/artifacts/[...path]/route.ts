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
  path: z.array(z.string().min(1)).min(1, "Artifact path required"),
});

/**
 * GET /api/packages/[id]/artifacts/[...path] — Stream artifact bytes.
 *
 * Serves raw artifact files (CSV, log, etc.) for a given proof package.
 * Returns the raw bytes with appropriate Content-Type and hash headers.
 *
 * Response:
 *   200: Raw bytes with Content-Type and X-Artifact-Hash headers
 *   400: { error: string } — invalid params
 *   404: { error: string } — artifact not found
 *   500: { error: string } — server error
 */
export async function GET(request: Request, { params }: { params: { id: string; path: string[] } }) {
  const requestId = request.headers.get("x-request-id") ?? "unknown";
  const parsed = ParamsSchema.safeParse(params);
  if (!parsed.success) {
    increment("lux_http_errors_total", "/api/packages/[id]/artifacts");
    return NextResponse.json(
      { error: `Invalid parameters: ${parsed.error.issues.map((i) => i.message).join(", ")}` },
      { status: 400, headers: { "X-Request-Id": requestId } },
    );
  }

  const artifactUri = parsed.data.path.join("/");
  const timer = startTimer("read_artifact");

  try {
    const provider = await getProvider();
    const result = await provider.readArtifact(parsed.data.id, artifactUri);

    if (!result.ok) {
      const timing = timer.stop();
      increment("lux_http_errors_total", "/api/packages/[id]/artifacts");
      logger.warn("api.artifacts.not_found", {
        requestId,
        packageId: parsed.data.id,
        artifactUri,
        reason: result.reason,
        durationMs: timing.durationMs,
      });
      return NextResponse.json(
        { error: result.reason },
        { status: 404, headers: { "X-Request-Id": requestId } },
      );
    }

    const timing = timer.stop();
    increment("lux_http_requests_total", "/api/packages/[id]/artifacts");
    observe("lux_http_duration_ms", timing.durationMs);
    logger.info("api.artifacts.read", {
      requestId,
      packageId: parsed.data.id,
      artifactUri,
      bytes: result.bytes.byteLength,
      durationMs: timing.durationMs,
    });

    return new NextResponse(Buffer.from(result.bytes), {
      status: 200,
      headers: {
        "Content-Type": result.mimeType,
        "Content-Length": String(result.bytes.byteLength),
        "X-Artifact-Hash": result.hash,
        "Cache-Control": "public, s-maxage=86400, stale-while-revalidate=604800, immutable",
        "Server-Timing": serverTimingHeader(timing),
        "X-Request-Id": requestId,
      },
    });
  } catch (err) {
    const timing = timer.stop();
    increment("lux_http_errors_total", "/api/packages/[id]/artifacts");
    logger.error("api.artifacts.error", {
      requestId,
      packageId: parsed.data.id,
      artifactUri,
      error: err instanceof Error ? err.message : String(err),
      durationMs: timing.durationMs,
    });

    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Failed to read artifact" },
      { status: 500, headers: { "X-Request-Id": requestId } },
    );
  }
}

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
