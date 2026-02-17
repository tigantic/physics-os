/**
 * Server-Timing measurement utility.
 *
 * Provides a simple start/stop timer interface that produces values
 * compatible with the Server-Timing HTTP header.
 *
 * Usage:
 *   const timer = startTimer("load_package");
 *   const pkg = await provider.loadPackage(id);
 *   const entry = timer.stop();
 *   headers.set("Server-Timing", serverTimingHeader(entry));
 *
 * @see https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Server-Timing
 */

export interface TimingEntry {
  readonly name: string;
  readonly durationMs: number;
}

export interface Timer {
  /** Stop the timer and return the completed entry. */
  stop(): TimingEntry;
}

/**
 * Start a named timer. Call `.stop()` to capture the elapsed duration.
 * Uses `performance.now()` for sub-millisecond precision.
 */
export function startTimer(name: string): Timer {
  const start = performance.now();
  return {
    stop(): TimingEntry {
      const durationMs = Math.round((performance.now() - start) * 100) / 100;
      return Object.freeze({ name, durationMs });
    },
  };
}

/**
 * Format one or more timing entries into a Server-Timing header value.
 *
 * @example
 * serverTimingHeader({ name: "db", durationMs: 12.5 }, { name: "render", durationMs: 3.2 })
 * // → "db;dur=12.5, render;dur=3.2"
 */
export function serverTimingHeader(...entries: TimingEntry[]): string {
  return entries.map((e) => `${e.name};dur=${e.durationMs}`).join(", ");
}
