import { NextResponse } from "next/server";

/**
 * Health check endpoint for orchestrator probes (Kubernetes, ECS, Docker HEALTHCHECK).
 * Returns 200 with basic service info.
 */
export function GET() {
  return NextResponse.json(
    {
      status: "ok",
      service: "lux-proof-viewer",
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
    },
    { status: 200 },
  );
}

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
