import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, cleanup } from "@testing-library/react";
import React from "react";

// Store callbacks registered by PerformanceObserver
const observers: Array<{ type: string; callback: (list: { getEntries: () => PerformanceEntry[] }) => void }> = [];

class MockPerformanceObserver {
  private callback: (list: { getEntries: () => PerformanceEntry[] }) => void;
  constructor(callback: (list: { getEntries: () => PerformanceEntry[] }) => void) {
    this.callback = callback;
  }
  observe(opts: { type: string; buffered?: boolean }) {
    observers.push({ type: opts.type, callback: this.callback });
  }
  disconnect() {
    const idx = observers.findIndex((o) => o.callback === this.callback);
    if (idx >= 0) observers.splice(idx, 1);
  }
}

describe("WebVitalsReporter", () => {
  const originalEnv = process.env.NEXT_PUBLIC_LUX_VITALS_ENDPOINT;
  let sendBeaconSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    observers.length = 0;
    sendBeaconSpy = vi.fn().mockReturnValue(true);
    Object.defineProperty(globalThis, "navigator", {
      value: { sendBeacon: sendBeaconSpy, userAgent: "test" },
      writable: true,
      configurable: true,
    });
    // @ts-expect-error — test override
    globalThis.PerformanceObserver = MockPerformanceObserver;
    // Mock navigation timing for TTFB
    vi.spyOn(performance, "getEntriesByType").mockReturnValue([
      { requestStart: 10, responseStart: 80 } as unknown as PerformanceEntry,
    ]);
  });

  afterEach(() => {
    process.env.NEXT_PUBLIC_LUX_VITALS_ENDPOINT = originalEnv;
    cleanup();
    vi.restoreAllMocks();
  });

  it("renders null (no visible output)", async () => {
    process.env.NEXT_PUBLIC_LUX_VITALS_ENDPOINT = "/api/vitals";
    const { WebVitalsReporter } = await import("@/lib/WebVitalsReporter");

    const { container } = render(React.createElement(WebVitalsReporter));
    expect(container.innerHTML).toBe("");
  });

  it("does not register observers when env var absent", async () => {
    delete process.env.NEXT_PUBLIC_LUX_VITALS_ENDPOINT;
    // Re-import to pick up env change
    vi.resetModules();
    const { WebVitalsReporter } = await import("@/lib/WebVitalsReporter");

    render(React.createElement(WebVitalsReporter));
    expect(observers).toHaveLength(0);
  });

  it("registers observers when enabled", async () => {
    process.env.NEXT_PUBLIC_LUX_VITALS_ENDPOINT = "/api/vitals";
    vi.resetModules();
    const { WebVitalsReporter } = await import("@/lib/WebVitalsReporter");

    render(React.createElement(WebVitalsReporter));

    // Should register paint (FCP), largest-contentful-paint (LCP),
    // layout-shift (CLS), event (INP)
    const types = observers.map((o) => o.type);
    expect(types).toContain("paint");
    expect(types).toContain("largest-contentful-paint");
    expect(types).toContain("layout-shift");
    expect(types).toContain("event");
  });

  it("reports TTFB immediately on mount", async () => {
    process.env.NEXT_PUBLIC_LUX_VITALS_ENDPOINT = "/api/vitals";
    vi.resetModules();
    const { WebVitalsReporter } = await import("@/lib/WebVitalsReporter");

    render(React.createElement(WebVitalsReporter));

    // TTFB should be reported immediately (80 - 10 = 70ms)
    expect(sendBeaconSpy).toHaveBeenCalled();
    const blob = sendBeaconSpy.mock.calls[0]![1] as Blob;
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toBe("application/json");
    const body = JSON.parse(await blob.text()) as {
      name: string;
      value: number;
      rating: string;
    };
    expect(body.name).toBe("TTFB");
    expect(body.value).toBe(70);
    expect(body.rating).toBe("good");
  });

  it("disconnects observers on unmount", async () => {
    process.env.NEXT_PUBLIC_LUX_VITALS_ENDPOINT = "/api/vitals";
    vi.resetModules();
    const { WebVitalsReporter } = await import("@/lib/WebVitalsReporter");

    const { unmount } = render(React.createElement(WebVitalsReporter));
    expect(observers.length).toBeGreaterThan(0);

    unmount();
    expect(observers).toHaveLength(0);
  });
});
