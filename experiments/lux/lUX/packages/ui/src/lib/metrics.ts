import "server-only";

/**
 * Lightweight in-memory metrics collection with Prometheus exposition.
 *
 * Zero-dependency metrics store designed for low-to-medium traffic proof
 * viewer deployments. For high-traffic scenarios, replace with prom-client
 * or OpenTelemetry metrics SDK.
 *
 * Thread safety: Node.js is single-threaded — no locks required.
 */

/* ── Counter ───────────────────────────────────────────── */

interface CounterData {
  help: string;
  labels: Map<string, number>;
}

const counters = new Map<string, CounterData>();

/**
 * Pre-declare a counter with its help text.
 * Safe to call multiple times — subsequent calls are no-ops.
 */
export function defineCounter(name: string, help: string): void {
  if (!counters.has(name)) {
    counters.set(name, { help, labels: new Map() });
  }
}

/**
 * Increment a counter by `by` (default 1).
 * Auto-creates the counter if not already defined.
 *
 * @param name - Metric name (e.g. "lux_http_requests_total")
 * @param label - Label value for the `route` label (e.g. "/api/packages")
 * @param by - Increment value (default 1)
 */
export function increment(name: string, label = "total", by = 1): void {
  let counter = counters.get(name);
  if (!counter) {
    counter = { help: name, labels: new Map() };
    counters.set(name, counter);
  }
  counter.labels.set(label, (counter.labels.get(label) ?? 0) + by);
}

/* ── Gauge ─────────────────────────────────────────────── */

interface GaugeData {
  help: string;
  value: number;
}

const gauges = new Map<string, GaugeData>();

/**
 * Set a gauge to an absolute value.
 */
export function setGauge(name: string, value: number, help?: string): void {
  gauges.set(name, { help: help ?? name, value });
}

/* ── Histogram (simplified: sum + count) ───────────────── */

interface HistogramData {
  help: string;
  count: number;
  sum: number;
}

const histograms = new Map<string, HistogramData>();

/**
 * Pre-declare a histogram with its help text.
 */
export function defineHistogram(name: string, help: string): void {
  if (!histograms.has(name)) {
    histograms.set(name, { help, count: 0, sum: 0 });
  }
}

/**
 * Record a histogram observation (e.g. request latency in ms).
 */
export function observe(name: string, value: number): void {
  let h = histograms.get(name);
  if (!h) {
    h = { help: name, count: 0, sum: 0 };
    histograms.set(name, h);
  }
  h.count++;
  h.sum += value;
}

/* ── Prometheus Exposition Format ──────────────────────── */

/**
 * Render all metrics in Prometheus text exposition format.
 *
 * @see https://prometheus.io/docs/instrumenting/exposition_formats/
 */
export function toPrometheus(): string {
  const lines: string[] = [];

  // Node.js runtime metrics (always present)
  const mem = process.memoryUsage();

  lines.push("# HELP process_uptime_seconds Node.js process uptime in seconds");
  lines.push("# TYPE process_uptime_seconds gauge");
  lines.push(`process_uptime_seconds ${process.uptime()}`);

  lines.push("# HELP nodejs_heap_used_bytes Node.js heap used bytes");
  lines.push("# TYPE nodejs_heap_used_bytes gauge");
  lines.push(`nodejs_heap_used_bytes ${mem.heapUsed}`);

  lines.push("# HELP nodejs_heap_total_bytes Node.js heap total bytes");
  lines.push("# TYPE nodejs_heap_total_bytes gauge");
  lines.push(`nodejs_heap_total_bytes ${mem.heapTotal}`);

  lines.push("# HELP nodejs_rss_bytes Node.js resident set size bytes");
  lines.push("# TYPE nodejs_rss_bytes gauge");
  lines.push(`nodejs_rss_bytes ${mem.rss}`);

  lines.push("# HELP nodejs_external_bytes Node.js external memory bytes");
  lines.push("# TYPE nodejs_external_bytes gauge");
  lines.push(`nodejs_external_bytes ${mem.external}`);

  // Application counters
  for (const [name, data] of counters) {
    lines.push(`# HELP ${name} ${data.help}`);
    lines.push(`# TYPE ${name} counter`);
    for (const [label, value] of data.labels) {
      lines.push(`${name}{route="${label}"} ${value}`);
    }
  }

  // Application gauges
  for (const [name, data] of gauges) {
    lines.push(`# HELP ${name} ${data.help}`);
    lines.push(`# TYPE ${name} gauge`);
    lines.push(`${name} ${data.value}`);
  }

  // Application histograms (summary-style: count + sum)
  for (const [name, data] of histograms) {
    lines.push(`# HELP ${name} ${data.help}`);
    lines.push(`# TYPE ${name} histogram`);
    lines.push(`${name}_count ${data.count}`);
    lines.push(`${name}_sum ${data.sum}`);
  }

  return lines.join("\n") + "\n";
}

/* ── Reset (testing only) ──────────────────────────────── */

/**
 * Clear all metrics. Intended for test isolation only.
 */
export function resetMetrics(): void {
  counters.clear();
  gauges.clear();
  histograms.clear();
}
