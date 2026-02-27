import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock server-only
vi.mock("server-only", () => ({}));

import {
  defineCounter,
  increment,
  setGauge,
  defineHistogram,
  observe,
  toPrometheus,
  resetMetrics,
} from "@/lib/metrics";

describe("metrics", () => {
  beforeEach(() => {
    resetMetrics();
  });

  describe("counters", () => {
    it("increments by 1 by default", () => {
      increment("test_counter", "route_a");
      increment("test_counter", "route_a");
      const output = toPrometheus();
      expect(output).toContain('test_counter{route="route_a"} 2');
    });

    it("increments by custom value", () => {
      increment("test_counter", "route_b", 5);
      const output = toPrometheus();
      expect(output).toContain('test_counter{route="route_b"} 5');
    });

    it("creates counter on first increment", () => {
      increment("auto_counter", "x");
      const output = toPrometheus();
      expect(output).toContain("auto_counter");
    });

    it("defineCounter sets help text", () => {
      defineCounter("defined_counter", "A test counter");
      increment("defined_counter", "y");
      const output = toPrometheus();
      expect(output).toContain("# HELP defined_counter A test counter");
      expect(output).toContain("# TYPE defined_counter counter");
    });
  });

  describe("gauges", () => {
    it("sets absolute value", () => {
      setGauge("test_gauge", 42, "A test gauge");
      const output = toPrometheus();
      expect(output).toContain("test_gauge 42");
      expect(output).toContain("# HELP test_gauge A test gauge");
      expect(output).toContain("# TYPE test_gauge gauge");
    });

    it("overwrites previous value", () => {
      setGauge("test_gauge", 10);
      setGauge("test_gauge", 20);
      const output = toPrometheus();
      expect(output).toContain("test_gauge 20");
      expect(output).not.toContain("test_gauge 10");
    });
  });

  describe("histograms", () => {
    it("tracks count and sum", () => {
      observe("test_histogram", 10);
      observe("test_histogram", 20);
      observe("test_histogram", 30);
      const output = toPrometheus();
      expect(output).toContain("test_histogram_count 3");
      expect(output).toContain("test_histogram_sum 60");
    });

    it("defineHistogram sets help text", () => {
      defineHistogram("defined_histogram", "A test histogram");
      observe("defined_histogram", 5);
      const output = toPrometheus();
      expect(output).toContain("# HELP defined_histogram A test histogram");
      expect(output).toContain("# TYPE defined_histogram histogram");
    });
  });

  describe("toPrometheus", () => {
    it("includes Node.js runtime metrics", () => {
      const output = toPrometheus();
      expect(output).toContain("process_uptime_seconds");
      expect(output).toContain("nodejs_heap_used_bytes");
      expect(output).toContain("nodejs_heap_total_bytes");
      expect(output).toContain("nodejs_rss_bytes");
      expect(output).toContain("nodejs_external_bytes");
    });

    it("ends with newline", () => {
      const output = toPrometheus();
      expect(output.endsWith("\n")).toBe(true);
    });

    it("has valid Prometheus format (# HELP / # TYPE lines)", () => {
      const output = toPrometheus();
      const lines = output.split("\n").filter((l) => l.startsWith("#"));
      for (const line of lines) {
        expect(line).toMatch(/^# (HELP|TYPE) /);
      }
    });
  });

  describe("resetMetrics", () => {
    it("clears all counters, gauges, and histograms", () => {
      increment("a", "x");
      setGauge("b", 1);
      observe("c", 1);
      resetMetrics();
      const output = toPrometheus();
      expect(output).not.toContain("a{");
      expect(output).not.toContain("\nb ");
      expect(output).not.toContain("c_count");
    });
  });
});
