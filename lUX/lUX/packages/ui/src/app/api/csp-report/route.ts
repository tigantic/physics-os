import { NextResponse } from "next/server";
import { z } from "zod";
import { logger } from "@/lib/logger";

/**
 * Zod schema for CSP violation reports.
 * Accepts both Reporting API v1 (`report-to` directive) flat format
 * and legacy `report-uri` directive `{ "csp-report": { ... } }` wrapper.
 * Field lengths are capped to prevent log-flooding attacks.
 */
const CspFieldSchema = z.string().max(2048).optional();

const CspViolationSchema = z.object({
  "blocked-uri": CspFieldSchema,
  blockedURL: CspFieldSchema,
  "violated-directive": CspFieldSchema,
  effectiveDirective: CspFieldSchema,
  "document-uri": CspFieldSchema,
  url: CspFieldSchema,
  "original-policy": z.string().max(4096).optional(),
  "source-file": CspFieldSchema,
  sourceFile: CspFieldSchema,
  "line-number": z.number().int().nonnegative().optional(),
  lineNumber: z.number().int().nonnegative().optional(),
  "column-number": z.number().int().nonnegative().optional(),
  columnNumber: z.number().int().nonnegative().optional(),
  disposition: z.string().max(64).optional(),
}).passthrough();

const CspReportWrapperSchema = z.union([
  z.object({ "csp-report": CspViolationSchema }).passthrough(),
  CspViolationSchema,
]);

/**
 * POST /api/csp-report — Receive Content-Security-Policy violation reports.
 *
 * Accepts both formats:
 *   - Reporting API v1 (`report-to` directive): `{ ... }` body
 *   - Legacy `report-uri` directive: `{ "csp-report": { ... } }` body
 *
 * Validates payload structure and field lengths to prevent log-flooding.
 * Logs all violations as structured warnings for monitoring and alerting.
 * Returns 204 No Content on success.
 */
export async function POST(request: Request) {
  try {
    // Reject oversized payloads (16 KiB limit for a single violation report)
    const contentLength = request.headers.get("content-length");
    if (contentLength && parseInt(contentLength, 10) > 16_384) {
      return new NextResponse(null, { status: 413 });
    }

    const body = await request.json();
    const parsed = CspReportWrapperSchema.safeParse(body);
    if (!parsed.success) {
      return new NextResponse(null, { status: 400 });
    }

    // `report-uri` wraps in "csp-report"; Reporting API sends flat
    const data = parsed.data;
    const report = (typeof data === "object" && data !== null && "csp-report" in data)
      ? (data as Record<string, unknown>)["csp-report"] as Record<string, unknown>
      : data as Record<string, unknown>;

    logger.warn("csp.violation", {
      blockedUri: report["blocked-uri"] ?? report["blockedURL"] ?? "unknown",
      violatedDirective: report["violated-directive"] ?? report["effectiveDirective"] ?? "unknown",
      documentUri: report["document-uri"] ?? report["url"] ?? "unknown",
      originalPolicy: report["original-policy"],
      sourceFile: report["source-file"] ?? report["sourceFile"],
      lineNumber: report["line-number"] ?? report["lineNumber"],
      columnNumber: report["column-number"] ?? report["columnNumber"],
      disposition: report["disposition"],
      requestId: request.headers.get("x-request-id"),
    });

    return new NextResponse(null, { status: 204 });
  } catch {
    return new NextResponse(null, { status: 400 });
  }
}

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
