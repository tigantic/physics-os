import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { reportError } from "@/lib/reportError";

describe("reportError", () => {
  const originalError = console.error;
  const errorSpy = vi.fn();

  let sendBeaconSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    console.error = errorSpy;
    sendBeaconSpy = vi.fn().mockReturnValue(true);
    Object.defineProperty(globalThis, "navigator", {
      value: { sendBeacon: sendBeaconSpy, userAgent: "test-ua" },
      writable: true,
      configurable: true,
    });
  });

  afterEach(() => {
    console.error = originalError;
    vi.restoreAllMocks();
  });

  it("calls console.error with component prefix", () => {
    const err = new Error("test");
    reportError(err, "RootError");
    expect(errorSpy).toHaveBeenCalledWith("[lUX] RootError:", err);
  });

  it("beacons JSON payload to /api/errors", () => {
    const err = new Error("boom");
    reportError(err, "GalleryError");

    expect(sendBeaconSpy).toHaveBeenCalledOnce();
    const [url, body] = sendBeaconSpy.mock.calls[0] as [string, string];
    expect(url).toBe("/api/errors");

    const parsed = JSON.parse(body) as Record<string, unknown>;
    expect(parsed.message).toBe("boom");
    expect(parsed.component).toBe("GalleryError");
    expect(parsed.stack).toBeDefined();
    expect(parsed.timestamp).toBeDefined();
    expect(parsed.userAgent).toBe("test-ua");
  });

  it("includes digest when present", () => {
    const err = Object.assign(new Error("digest-err"), { digest: "abc123" });
    reportError(err, "Test");

    const [, body] = sendBeaconSpy.mock.calls[0] as [string, string];
    const parsed = JSON.parse(body) as Record<string, unknown>;
    expect(parsed.digest).toBe("abc123");
  });

  it("falls back to fetch when sendBeacon unavailable", () => {
    const fetchSpy = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    globalThis.fetch = fetchSpy;
    Object.defineProperty(globalThis, "navigator", {
      value: { userAgent: "test" },
      writable: true,
      configurable: true,
    });

    reportError(new Error("fallback"), "Test");

    expect(fetchSpy).toHaveBeenCalledOnce();
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/errors");
    expect(init.method).toBe("POST");
    expect(init.keepalive).toBe(true);
  });

  it("never throws even if beacon and fetch both fail", () => {
    Object.defineProperty(globalThis, "navigator", {
      value: {
        sendBeacon: () => {
          throw new Error("sendBeacon exploded");
        },
        userAgent: "test",
      },
      writable: true,
      configurable: true,
    });

    expect(() => reportError(new Error("safe"), "Test")).not.toThrow();
  });

  it("includes url from window.location", () => {
    Object.defineProperty(globalThis, "window", {
      value: { location: { href: "http://localhost/gallery?x=1" } },
      writable: true,
      configurable: true,
    });

    reportError(new Error("url test"), "Comp");

    const [, body] = sendBeaconSpy.mock.calls[0] as [string, string];
    const parsed = JSON.parse(body) as Record<string, unknown>;
    expect(parsed.url).toBe("http://localhost/gallery?x=1");
  });
});
