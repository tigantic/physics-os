"use client";

/**
 * Client-side error reporting utility.
 *
 * Beacons error details to /api/errors for structured server-side logging.
 * Uses `navigator.sendBeacon` for reliability (survives page unload).
 * Falls back to `fetch` with `keepalive: true`.
 *
 * Includes a breadcrumb trail (circular buffer, max 25 entries) so that error
 * reports arrive with the user's recent action history for easier debugging.
 *
 * Error reporting must never throw — all failures are silently swallowed.
 */

/* ── Breadcrumbs ───────────────────────────────────────────────────── */

export type BreadcrumbType = "navigation" | "action" | "error" | "state" | "lifecycle";

export interface Breadcrumb {
  readonly type: BreadcrumbType;
  readonly label: string;
  readonly timestamp: string;
  readonly data?: Record<string, unknown>;
}

const MAX_BREADCRUMBS = 25;
const breadcrumbs: Breadcrumb[] = [];

/**
 * Push a breadcrumb onto the circular buffer. When the buffer is full the
 * oldest entry is discarded. Safe to call from any client code path.
 */
export function addBreadcrumb(
  type: BreadcrumbType,
  label: string,
  data?: Record<string, unknown>,
): void {
  const entry: Breadcrumb = {
    type,
    label,
    timestamp: new Date().toISOString(),
    data,
  };
  if (breadcrumbs.length >= MAX_BREADCRUMBS) {
    breadcrumbs.shift();
  }
  breadcrumbs.push(entry);
}

/** Return a frozen snapshot of the current breadcrumb trail. */
export function getBreadcrumbs(): readonly Breadcrumb[] {
  return Object.freeze([...breadcrumbs]);
}

/* ── Automatic breadcrumb collectors ───────────────────────────────── */

if (typeof window !== "undefined") {
  // Track client-side navigation (pushState / popstate)
  window.addEventListener("popstate", () => {
    addBreadcrumb("navigation", "popstate", { url: window.location.href });
  });

  // Track visibility changes (tab switches, minimise, etc.)
  document.addEventListener("visibilitychange", () => {
    addBreadcrumb("lifecycle", `visibility:${document.visibilityState}`);
  });

  // Intercept pushState/replaceState so SPA navigation generates crumbs
  const originalPush = history.pushState.bind(history);
  const originalReplace = history.replaceState.bind(history);

  history.pushState = function pushStateTraced(...args: Parameters<typeof history.pushState>) {
    addBreadcrumb("navigation", "pushState", { url: String(args[2] ?? "") });
    return originalPush(...args);
  };

  history.replaceState = function replaceStateTraced(
    ...args: Parameters<typeof history.replaceState>
  ) {
    addBreadcrumb("navigation", "replaceState", { url: String(args[2] ?? "") });
    return originalReplace(...args);
  };
}

/* ── Error report ──────────────────────────────────────────────────── */

interface ErrorReport {
  readonly message: string;
  readonly stack?: string;
  readonly digest?: string;
  readonly component: string;
  readonly url: string;
  readonly timestamp: string;
  readonly userAgent: string;
  readonly breadcrumbs: readonly Breadcrumb[];
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

  // Record the error itself as a breadcrumb before sending
  addBreadcrumb("error", error.message, { component, digest: error.digest });

  const report: ErrorReport = {
    message: error.message,
    stack: error.stack,
    digest: error.digest,
    component,
    url: typeof window !== "undefined" ? window.location.href : "",
    timestamp: new Date().toISOString(),
    userAgent: typeof navigator !== "undefined" ? navigator.userAgent : "",
    breadcrumbs: getBreadcrumbs(),
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
