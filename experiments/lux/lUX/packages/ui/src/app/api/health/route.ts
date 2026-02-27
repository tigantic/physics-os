import { NextResponse } from "next/server";
import { isProviderReady } from "@/config/provider";

const VERSION = process.env.LUX_VERSION ?? "dev";
const COMMIT_SHA = process.env.LUX_COMMIT_SHA ?? "unknown";

/**
 * GET /api/health — Liveness probe for orchestrator checks.
 *
 * Returns 200 with extended service diagnostics:
 *   - Version and commit SHA for deployment tracking
 *   - Provider readiness status
 *   - Memory usage for resource monitoring
 *   - Process uptime for restart detection
 *
 * Use for Docker HEALTHCHECK, Kubernetes livenessProbe, and
 * uptime monitoring services.
 */
export function GET() {
  const mem = process.memoryUsage();

  return NextResponse.json(
    {
      status: "ok",
      service: "lux-proof-viewer",
      version: VERSION,
      commitSha: COMMIT_SHA,
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      provider: {
        ready: isProviderReady(),
      },
      memory: {
        heapUsed: mem.heapUsed,
        heapTotal: mem.heapTotal,
        rss: mem.rss,
        external: mem.external,
      },
    },
    { status: 200 },
  );
}

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
