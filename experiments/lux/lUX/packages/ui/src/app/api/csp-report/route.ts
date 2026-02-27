import { NextResponse } from "next/server";
import { z } from "zod";
import { logger } from "@/lib/logger";

/* ── Alert thresholds ──────────────────────────────────────────────── */

/**
 * Sliding-window counter that tracks violations per time window.
 * When violations exceed the threshold, the route escalates log level
 * from `warn` to `error` and optionally fires a webhook.
 */
const WINDOW_MS = 60_000; // 1-minute sliding window
const ALERT_THRESHOLD = Number(process.env.LUX_CSP_ALERT_THRESHOLD) || 25;
const ALERT_WEBHOOK = process.env.LUX_CSP_ALERT_WEBHOOK ?? "";
const ALERT_COOLDOWN_MS = 5 * 60_000; // Max one webhook per 5 minutes

const violationTimestamps: number[] = [];
let lastAlertFiredAt = 0;

function recordViolation(): number {
  const now = Date.now();
  violationTimestamps.push(now);
  // Evict entries outside the sliding window
  const cutoff = now - WINDOW_MS;
  while (violationTimestamps.length > 0 && violationTimestamps[0] < cutoff) {
    violationTimestamps.shift();
  }
  return violationTimestamps.length;
}

async function fireWebhookAlert(count: number, sample: Record<string, unknown>): Promise<void> {
  if (!ALERT_WEBHOOK) return;
  const now = Date.now();
  if (now - lastAlertFiredAt < ALERT_COOLDOWN_MS) return;
  lastAlertFiredAt = now;

  try {
    await fetch(ALERT_WEBHOOK, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: `⚠️ CSP violation spike: ${count} violations in the last 60 s`,
        timestamp: new Date().toISOString(),
        threshold: ALERT_THRESHOLD,
        count,
        sample,
      }),
      signal: AbortSignal.timeout(5_000),
    });
  } catch (err) {
    logger.error("csp.alert.webhook_failed", { error: String(err) });
  }
}

/* ── Zod validation ────────────────────────────────────────────────── */

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

/* ── Route handler ─────────────────────────────────────────────────── */

/**
 * POST /api/csp-report — Receive Content-Security-Policy violation reports.
 *
 * Accepts both formats:
 *   - Reporting API v1 (`report-to` directive): `{ ... }` body
 *   - Legacy `report-uri` directive: `{ "csp-report": { ... } }` body
 *
 * Validates payload structure and field lengths to prevent log-flooding.
 * Tracks violation rate and escalates to `error` level (+ webhook) when
 * violations exceed `LUX_CSP_ALERT_THRESHOLD` per minute (default 25).
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

    const structured = {
      blockedUri: report["blocked-uri"] ?? report["blockedURL"] ?? "unknown",
      violatedDirective: report["violated-directive"] ?? report["effectiveDirective"] ?? "unknown",
      documentUri: report["document-uri"] ?? report["url"] ?? "unknown",
      originalPolicy: report["original-policy"],
      sourceFile: report["source-file"] ?? report["sourceFile"],
      lineNumber: report["line-number"] ?? report["lineNumber"],
      columnNumber: report["column-number"] ?? report["columnNumber"],
      disposition: report["disposition"],
      requestId: request.headers.get("x-request-id"),
    };

    const windowCount = recordViolation();

    if (windowCount >= ALERT_THRESHOLD) {
      logger.error("csp.violation.spike", { ...structured, windowCount, threshold: ALERT_THRESHOLD });
      // Fire webhook asynchronously — do not block the response
      void fireWebhookAlert(windowCount, structured);
    } else {
      logger.warn("csp.violation", structured);
    }

    return new NextResponse(null, { status: 204 });
  } catch {
    return new NextResponse(null, { status: 400 });
  }
}

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
