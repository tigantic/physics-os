import "server-only";

/**
 * Structured JSON logger for server-side use.
 *
 * Output format: one JSON object per line (NDJSON), compatible with
 * ELK, Datadog, CloudWatch, Grafana Loki, and other log aggregators.
 *
 * Request ID correlation: pass `requestId` in the data object to
 * correlate logs with specific HTTP requests.
 *
 * Log level filtering: set LUX_LOG_LEVEL env var (debug|info|warn|error).
 * Default: "info". Reads env on every call to allow runtime reconfiguration
 * without process restart.
 */

export type LogLevel = "debug" | "info" | "warn" | "error";

const LEVEL_SEVERITY: Readonly<Record<LogLevel, number>> = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40,
};

const SERVICE = "lux-proof-viewer";
const VALID_LEVELS = new Set(Object.keys(LEVEL_SEVERITY));

function minSeverity(): number {
  const raw = process.env.LUX_LOG_LEVEL?.trim().toLowerCase();
  if (raw && VALID_LEVELS.has(raw)) return LEVEL_SEVERITY[raw as LogLevel];
  return LEVEL_SEVERITY.info;
}

function serializeValue(value: unknown): unknown {
  if (value instanceof Error) {
    return { name: value.name, message: value.message, stack: value.stack };
  }
  return value;
}

function emit(level: LogLevel, msg: string, data?: Record<string, unknown>): void {
  const severity = LEVEL_SEVERITY[level];
  if (severity < minSeverity()) return;

  const entry: Record<string, unknown> = {
    level,
    severity,
    msg,
    service: SERVICE,
    timestamp: new Date().toISOString(),
    pid: process.pid,
  };

  if (data) {
    for (const [key, value] of Object.entries(data)) {
      entry[key] = serializeValue(value);
    }
  }

  const line = JSON.stringify(entry) + "\n";

  if (severity >= LEVEL_SEVERITY.warn) {
    process.stderr.write(line);
  } else {
    process.stdout.write(line);
  }
}

export const logger = Object.freeze({
  debug: (msg: string, data?: Record<string, unknown>) => emit("debug", msg, data),
  info: (msg: string, data?: Record<string, unknown>) => emit("info", msg, data),
  warn: (msg: string, data?: Record<string, unknown>) => emit("warn", msg, data),
  error: (msg: string, data?: Record<string, unknown>) => emit("error", msg, data),
});

export type Logger = typeof logger;
