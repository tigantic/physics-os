import { NextResponse } from "next/server";
import { logger } from "@/lib/logger";

/**
 * POST /api/csp-report — Receive Content-Security-Policy violation reports.
 *
 * Accepts both formats:
 *   - Reporting API v1 (`report-to` directive): `{ ... }` body
 *   - Legacy `report-uri` directive: `{ "csp-report": { ... } }` body
 *
 * Logs all violations as structured warnings for monitoring and alerting.
 * Returns 204 No Content on success.
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();

    // `report-uri` wraps in "csp-report"; Reporting API sends flat
    const report = body["csp-report"] ?? body;

    logger.warn("csp.violation", {
      blockedUri: report["blocked-uri"] ?? report.blockedURL ?? "unknown",
      violatedDirective: report["violated-directive"] ?? report.effectiveDirective ?? "unknown",
      documentUri: report["document-uri"] ?? report.url ?? "unknown",
      originalPolicy: report["original-policy"],
      sourceFile: report["source-file"] ?? report.sourceFile,
      lineNumber: report["line-number"] ?? report.lineNumber,
      columnNumber: report["column-number"] ?? report.columnNumber,
      disposition: report.disposition,
      requestId: request.headers.get("x-request-id"),
    });

    return new NextResponse(null, { status: 204 });
  } catch {
    return new NextResponse(null, { status: 400 });
  }
}

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
