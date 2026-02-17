import { describe, it, expect, beforeEach } from "vitest";
import { addBreadcrumb, getBreadcrumbs } from "@/lib/reportError";
import type { Breadcrumb } from "@/lib/reportError";

/**
 * Tests for the breadcrumb subsystem of reportError.
 *
 * The circular buffer, addBreadcrumb(), and getBreadcrumbs() are the
 * public API. The automatic collectors (popstate, pushState, replaceState)
 * are tested indirectly via the browser event system in jsdom.
 */
describe("addBreadcrumb + getBreadcrumbs", () => {
  /** The breadcrumb buffer is module-level global state, so we prefill before
   *  each test by adding enough entries to flush prior state. Since we can't
   *  clear the internal array directly, we rely on the circular eviction. */
  beforeEach(() => {
    // Flood the buffer past its 25-entry cap so subsequent tests start
    // with a known-empty-ish tail (only our flood entries).
    for (let i = 0; i < 30; i++) {
      addBreadcrumb("state", `flush-${i}`);
    }
  });

  it("returns breadcrumbs that were just added", () => {
    addBreadcrumb("action", "clicked-button", { target: "#save" });
    const trail = getBreadcrumbs();
    const last = trail[trail.length - 1];
    expect(last.type).toBe("action");
    expect(last.label).toBe("clicked-button");
    expect(last.data).toEqual({ target: "#save" });
    expect(last.timestamp).toBeTruthy();
  });

  it("never exceeds 25 entries", () => {
    for (let i = 0; i < 40; i++) {
      addBreadcrumb("navigation", `nav-${i}`);
    }
    expect(getBreadcrumbs().length).toBeLessThanOrEqual(25);
  });

  it("evicts oldest entry when buffer is full", () => {
    // Add exactly 25, then one more — oldest should be gone
    for (let i = 0; i < 25; i++) {
      addBreadcrumb("action", `a-${i}`);
    }
    addBreadcrumb("action", "overflow");

    const trail = getBreadcrumbs();
    const labels = trail.map((b: Breadcrumb) => b.label);
    expect(labels).not.toContain("a-0");
    expect(labels).toContain("overflow");
    expect(trail.length).toBe(25);
  });

  it("returns a frozen snapshot (immutable)", () => {
    addBreadcrumb("lifecycle", "mount");
    const trail = getBreadcrumbs();
    expect(Object.isFrozen(trail)).toBe(true);
  });

  it("snapshot is independent of further mutations", () => {
    addBreadcrumb("state", "before-snapshot");
    const snapshot = getBreadcrumbs();
    const lenBefore = snapshot.length;

    addBreadcrumb("state", "after-snapshot");
    expect(snapshot.length).toBe(lenBefore);
  });

  it("includes ISO-8601 timestamp on every breadcrumb", () => {
    addBreadcrumb("error", "oops");
    const trail = getBreadcrumbs();
    const last = trail[trail.length - 1];
    // ISO-8601 pattern: YYYY-MM-DDTHH:mm:ss.sssZ
    expect(last.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z$/);
  });

  it("supports all breadcrumb types", () => {
    const types = ["navigation", "action", "error", "state", "lifecycle"] as const;
    for (const t of types) {
      addBreadcrumb(t, `type-${t}`);
    }
    const trail = getBreadcrumbs();
    const last5 = trail.slice(-5);
    const foundTypes = last5.map((b: Breadcrumb) => b.type);
    for (const t of types) {
      expect(foundTypes).toContain(t);
    }
  });

  it("allows undefined data", () => {
    addBreadcrumb("action", "no-data");
    const trail = getBreadcrumbs();
    const last = trail[trail.length - 1];
    expect(last.data).toBeUndefined();
  });
});
