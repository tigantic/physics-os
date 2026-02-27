import { NextResponse } from "next/server";
import { getProvider, isProviderReady } from "@/config/provider";

const VERSION = process.env.LUX_VERSION ?? "dev";
const COMMIT_SHA = process.env.LUX_COMMIT_SHA ?? "unknown";

/**
 * GET /api/ready — Readiness probe for orchestrator checks.
 *
 * Returns 200 when the data provider is initialized and can serve requests.
 * Returns 503 when the provider is not yet ready or fails to initialize.
 *
 * Use this for Kubernetes `readinessProbe` and load balancer health checks.
 * Traffic should not be routed to the instance until this returns 200.
 */
export async function GET() {
  // Fast path: provider already initialized
  if (isProviderReady()) {
    return NextResponse.json(
      {
        status: "ready",
        version: VERSION,
        commitSha: COMMIT_SHA,
        timestamp: new Date().toISOString(),
      },
      { status: 200 },
    );
  }

  // Slow path: attempt initialization
  try {
    await getProvider();
    return NextResponse.json(
      {
        status: "ready",
        version: VERSION,
        commitSha: COMMIT_SHA,
        timestamp: new Date().toISOString(),
      },
      { status: 200 },
    );
  } catch (err) {
    return NextResponse.json(
      {
        status: "not_ready",
        reason: err instanceof Error ? err.message : "Provider initialization failed",
        version: VERSION,
        commitSha: COMMIT_SHA,
        timestamp: new Date().toISOString(),
      },
      { status: 503 },
    );
  }
}

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
