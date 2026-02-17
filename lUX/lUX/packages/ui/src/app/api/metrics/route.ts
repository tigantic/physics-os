import { NextResponse } from "next/server";
import { toPrometheus } from "@/lib/metrics";

/**
 * GET /api/metrics — Prometheus-compatible metrics endpoint.
 *
 * Exposes Node.js runtime metrics (heap, RSS, uptime) and application
 * counters (request counts, error counts, latency summaries).
 *
 * Content-Type follows the Prometheus text exposition format spec.
 * @see https://prometheus.io/docs/instrumenting/exposition_formats/
 */
export function GET() {
  return new NextResponse(toPrometheus(), {
    status: 200,
    headers: {
      "Content-Type": "text/plain; version=0.0.4; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
