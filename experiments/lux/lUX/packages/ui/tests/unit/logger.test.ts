import { describe, it, expect, vi, beforeEach, afterEach, type MockInstance } from "vitest";

// Mock server-only
vi.mock("server-only", () => ({}));

describe("logger", () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let stdoutSpy: MockInstance<any>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let stderrSpy: MockInstance<any>;
  const originalEnv = process.env.LUX_LOG_LEVEL;

  beforeEach(() => {
    stdoutSpy = vi.spyOn(process.stdout, "write").mockReturnValue(true);
    stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    delete process.env.LUX_LOG_LEVEL;
  });

  afterEach(() => {
    stdoutSpy.mockRestore();
    stderrSpy.mockRestore();
    if (originalEnv !== undefined) {
      process.env.LUX_LOG_LEVEL = originalEnv;
    } else {
      delete process.env.LUX_LOG_LEVEL;
    }
  });

  it("emits info to stdout as JSON", async () => {
    const { logger } = await import("@/lib/logger");
    logger.info("test.message", { key: "value" });

    expect(stdoutSpy).toHaveBeenCalledOnce();
    const line = stdoutSpy.mock.calls[0][0] as string;
    const parsed = JSON.parse(line);
    expect(parsed.level).toBe("info");
    expect(parsed.msg).toBe("test.message");
    expect(parsed.key).toBe("value");
    expect(parsed.service).toBe("lux-proof-viewer");
    expect(parsed.timestamp).toBeDefined();
    expect(parsed.pid).toBe(process.pid);
  });

  it("emits warn to stderr", async () => {
    const { logger } = await import("@/lib/logger");
    logger.warn("test.warning");

    expect(stderrSpy).toHaveBeenCalledOnce();
    const parsed = JSON.parse(stderrSpy.mock.calls[0][0] as string);
    expect(parsed.level).toBe("warn");
    expect(parsed.severity).toBe(30);
  });

  it("emits error to stderr", async () => {
    const { logger } = await import("@/lib/logger");
    logger.error("test.error");

    expect(stderrSpy).toHaveBeenCalledOnce();
    const parsed = JSON.parse(stderrSpy.mock.calls[0][0] as string);
    expect(parsed.level).toBe("error");
    expect(parsed.severity).toBe(40);
  });

  it("emits debug to stdout", async () => {
    process.env.LUX_LOG_LEVEL = "debug";
    const { logger } = await import("@/lib/logger");
    logger.debug("test.debug");

    expect(stdoutSpy).toHaveBeenCalled();
    const parsed = JSON.parse(stdoutSpy.mock.calls[0][0] as string);
    expect(parsed.level).toBe("debug");
    expect(parsed.severity).toBe(10);
  });

  it("filters below min level (default info)", async () => {
    const { logger } = await import("@/lib/logger");
    logger.debug("should.not.appear");

    expect(stdoutSpy).not.toHaveBeenCalled();
    expect(stderrSpy).not.toHaveBeenCalled();
  });

  it("respects LUX_LOG_LEVEL=error filter", async () => {
    process.env.LUX_LOG_LEVEL = "error";
    const { logger } = await import("@/lib/logger");

    logger.info("should.not.appear");
    logger.warn("should.not.appear");
    expect(stdoutSpy).not.toHaveBeenCalled();
    expect(stderrSpy).not.toHaveBeenCalled();

    logger.error("should.appear");
    expect(stderrSpy).toHaveBeenCalledOnce();
  });

  it("serializes Error objects in data", async () => {
    const { logger } = await import("@/lib/logger");
    const error = new Error("test error");
    logger.error("with.error", { err: error });

    const parsed = JSON.parse(stderrSpy.mock.calls[0][0] as string);
    expect(parsed.err.name).toBe("Error");
    expect(parsed.err.message).toBe("test error");
    expect(parsed.err.stack).toBeDefined();
  });

  it("outputs valid NDJSON (newline terminated)", async () => {
    const { logger } = await import("@/lib/logger");
    logger.info("test");

    const line = stdoutSpy.mock.calls[0][0] as string;
    expect(line.endsWith("\n")).toBe(true);
    expect(() => JSON.parse(line.trim())).not.toThrow();
  });

  it("includes severity number", async () => {
    const { logger } = await import("@/lib/logger");
    logger.info("test");

    const parsed = JSON.parse(stdoutSpy.mock.calls[0][0] as string);
    expect(parsed.severity).toBe(20);
  });
});
