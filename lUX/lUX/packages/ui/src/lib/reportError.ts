"use client";

/**
 * Client-side error reporting utility.
 *
 * Beacons error details to /api/errors for structured server-side logging.
 * Uses `navigator.sendBeacon` for reliability (survives page unload).
 * Falls back to `fetch` with `keepalive: true`.
 *
 * Error reporting must never throw — all failures are silently swallowed.
 */

interface ErrorReport {
  readonly message: string;
  readonly stack?: string;
  readonly digest?: string;
  readonly component: string;
  readonly url: string;
  readonly timestamp: string;
  readonly userAgent: string;
}

/**
 * Report a caught error from an error boundary or catch block.
 *
 * @param error - The error object (must have `message`; `digest` and `stack` optional)
 * @param component - Identifier for the error source (e.g. "RootError", "GalleryError")
 */
export function reportError(error: Error & { digest?: string }, component: string): void {
  // Always log locally for developer visibility
  console.error(`[lUX] ${component}:`, error);

  const report: ErrorReport = {
    message: error.message,
    stack: error.stack,
    digest: error.digest,
    component,
    url: typeof window !== "undefined" ? window.location.href : "",
    timestamp: new Date().toISOString(),
    userAgent: typeof navigator !== "undefined" ? navigator.userAgent : "",
  };

  const body = JSON.stringify(report);

  try {
    if (typeof navigator !== "undefined" && navigator.sendBeacon) {
      navigator.sendBeacon("/api/errors", new Blob([body], { type: "application/json" }));
    } else if (typeof fetch !== "undefined") {
      fetch("/api/errors", {
        method: "POST",
        body,
        headers: { "Content-Type": "application/json" },
        keepalive: true,
      }).catch(() => {
        /* swallow rejected promise */
      });
    }
  } catch {
    // Swallow — error reporting must never throw
  }
}
