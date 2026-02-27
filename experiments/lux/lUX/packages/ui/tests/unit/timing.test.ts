import { describe, it, expect } from "vitest";
import { startTimer, serverTimingHeader } from "@/lib/timing";

describe("timing", () => {
  describe("startTimer", () => {
    it("returns a timer that stops with name and duration", () => {
      const timer = startTimer("test_op");
      const entry = timer.stop();

      expect(entry.name).toBe("test_op");
      expect(typeof entry.durationMs).toBe("number");
      expect(entry.durationMs).toBeGreaterThanOrEqual(0);
    });

    it("measures elapsed time", async () => {
      const timer = startTimer("slow_op");
      await new Promise((r) => setTimeout(r, 20));
      const entry = timer.stop();

      expect(entry.durationMs).toBeGreaterThanOrEqual(10);
    });

    it("returns a frozen entry", () => {
      const timer = startTimer("x");
      const entry = timer.stop();
      expect(Object.isFrozen(entry)).toBe(true);
    });
  });

  describe("serverTimingHeader", () => {
    it("formats a single entry", () => {
      const header = serverTimingHeader({ name: "db", durationMs: 12.5 });
      expect(header).toBe("db;dur=12.5");
    });

    it("formats multiple entries comma-separated", () => {
      const header = serverTimingHeader(
        { name: "db", durationMs: 12.5 },
        { name: "render", durationMs: 3.2 },
      );
      expect(header).toBe("db;dur=12.5, render;dur=3.2");
    });

    it("formats zero entries as empty string", () => {
      const header = serverTimingHeader();
      expect(header).toBe("");
    });
  });
});
