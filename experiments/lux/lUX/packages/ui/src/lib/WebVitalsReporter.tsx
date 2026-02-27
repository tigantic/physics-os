"use client";

import { useEffect } from "react";

/**
 * Client component that collects Core Web Vitals using the standard
 * PerformanceObserver API and beacons results to a configurable endpoint.
 *
 * Measured metrics:
 *   - TTFB  (Time to First Byte)
 *   - FCP   (First Contentful Paint)
 *   - LCP   (Largest Contentful Paint)
 *   - CLS   (Cumulative Layout Shift)
 *   - INP   (Interaction to Next Paint)
 *
 * Reporting is disabled by default. Set the `NEXT_PUBLIC_LUX_VITALS_ENDPOINT`
 * environment variable at build time to enable (e.g. "/api/vitals").
 *
 * @see https://web.dev/articles/vitals
 */

interface VitalMetric {
  readonly name: string;
  readonly value: number;
  readonly rating: "good" | "needs-improvement" | "poor";
  readonly url: string;
  readonly timestamp: string;
}

/**
 * Google-recommended thresholds: [good, needs-improvement].
 * Values above the second threshold are "poor".
 */
const THRESHOLDS: Readonly<Record<string, readonly [number, number]>> = {
  TTFB: [800, 1800],
  FCP: [1800, 3000],
  LCP: [2500, 4000],
  CLS: [0.1, 0.25],
  INP: [200, 500],
};

function rate(name: string, value: number): VitalMetric["rating"] {
  const t = THRESHOLDS[name];
  if (!t) return "good";
  if (value <= t[0]) return "good";
  if (value <= t[1]) return "needs-improvement";
  return "poor";
}

function beacon(metric: VitalMetric): void {
  const endpoint = process.env.NEXT_PUBLIC_LUX_VITALS_ENDPOINT;
  if (!endpoint) return;

  const body = JSON.stringify(metric);
  try {
    if (navigator.sendBeacon) {
      navigator.sendBeacon(endpoint, new Blob([body], { type: "application/json" }));
    } else {
      void fetch(endpoint, {
        method: "POST",
        body,
        headers: { "Content-Type": "application/json" },
        keepalive: true,
      });
    }
  } catch {
    // Swallow — vitals reporting must not throw
  }
}

function report(name: string, value: number): void {
  beacon({
    name,
    value: Math.round(value * 1000) / 1000,
    rating: rate(name, value),
    url: window.location.href,
    timestamp: new Date().toISOString(),
  });
}

/**
 * Renders nothing. Registers PerformanceObservers on mount and reports
 * final metrics when the page is hidden (ensures LCP/CLS/INP capture).
 */
export function WebVitalsReporter(): null {
  useEffect(() => {
    if (!process.env.NEXT_PUBLIC_LUX_VITALS_ENDPOINT) return;
    if (typeof PerformanceObserver === "undefined") return;

    // ── TTFB ──────────────────────────────────────────
    try {
      const nav = performance.getEntriesByType("navigation")[0] as
        | PerformanceNavigationTiming
        | undefined;
      if (nav) {
        report("TTFB", nav.responseStart - nav.requestStart);
      }
    } catch {
      /* unsupported */
    }

    // ── FCP ───────────────────────────────────────────
    const fcpObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.name === "first-contentful-paint") {
          report("FCP", entry.startTime);
          fcpObserver.disconnect();
        }
      }
    });
    try {
      fcpObserver.observe({ type: "paint", buffered: true });
    } catch {
      /* unsupported */
    }

    // ── LCP ───────────────────────────────────────────
    let lcpValue = 0;
    const lcpObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        lcpValue = entry.startTime;
      }
    });
    try {
      lcpObserver.observe({ type: "largest-contentful-paint", buffered: true });
    } catch {
      /* unsupported */
    }

    // ── CLS ───────────────────────────────────────────
    let clsValue = 0;
    const clsObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const shift = entry as PerformanceEntry & {
          hadRecentInput?: boolean;
          value?: number;
        };
        if (!shift.hadRecentInput && shift.value) {
          clsValue += shift.value;
        }
      }
    });
    try {
      clsObserver.observe({ type: "layout-shift", buffered: true });
    } catch {
      /* unsupported */
    }

    // ── INP ───────────────────────────────────────────
    let inpValue = 0;
    const inpObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const dur = (entry as PerformanceEntry & { duration: number }).duration;
        if (dur > inpValue) inpValue = dur;
      }
    });
    try {
      inpObserver.observe({ type: "event", buffered: true });
    } catch {
      /* unsupported */
    }

    // ── Report on page hide (final LCP, CLS, INP) ────
    function onVisibilityChange() {
      if (document.visibilityState === "hidden") {
        if (lcpValue > 0) report("LCP", lcpValue);
        report("CLS", clsValue);
        if (inpValue > 0) report("INP", inpValue);

        lcpObserver.disconnect();
        clsObserver.disconnect();
        inpObserver.disconnect();
        document.removeEventListener("visibilitychange", onVisibilityChange);
      }
    }

    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      fcpObserver.disconnect();
      lcpObserver.disconnect();
      clsObserver.disconnect();
      inpObserver.disconnect();
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, []);

  return null;
}
