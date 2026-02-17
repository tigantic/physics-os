import { NextResponse } from "next/server";
import { z } from "zod";
import { logger } from "@/lib/logger";

const ErrorReportSchema = z.object({
  message: z.string().max(4096),
  stack: z.string().max(16384).optional(),
  digest: z.string().max(256).optional(),
  component: z.string().max(128),
  url: z.string().max(2048),
  timestamp: z.string(),
  userAgent: z.string().max(512),
});

/**
 * POST /api/errors — Receive client-side error reports.
 *
 * Called by the `reportError()` utility in error boundaries.
 * Validates the payload with Zod and logs as a structured error
 * for aggregation by ELK / Datadog / CloudWatch.
 *
 * Returns 204 No Content on success.
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const parsed = ErrorReportSchema.safeParse(body);

    if (!parsed.success) {
      return NextResponse.json(
        { error: "Invalid error report" },
        { status: 400 },
      );
    }

    logger.error("client.error", {
      ...parsed.data,
      source: "error-boundary",
      requestId: request.headers.get("x-request-id"),
    });

    return new NextResponse(null, { status: 204 });
  } catch {
    return new NextResponse(null, { status: 400 });
  }
}

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
