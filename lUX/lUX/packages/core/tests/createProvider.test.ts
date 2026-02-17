import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { createProvider } from "../src/providers/createProvider.js";

describe("createProvider", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    // Reset env between tests
    delete process.env.LUX_DATA_PROVIDER;
    delete process.env.LUX_FIXTURES_ROOT;
    delete process.env.LUX_API_BASE_URL;
  });

  afterEach(() => {
    process.env = { ...originalEnv };
  });

  it("defaults to filesystem provider", async () => {
    const provider = await createProvider({ fixturesRoot: "/tmp/fixtures" });
    expect(provider.name).toBe("filesystem");
  });

  it("creates filesystem provider from explicit kind", async () => {
    const provider = await createProvider({ kind: "filesystem", fixturesRoot: "/tmp/fixtures" });
    expect(provider.name).toBe("filesystem");
  });

  it("creates http provider from explicit kind", async () => {
    const provider = await createProvider({ kind: "http", apiBaseUrl: "https://example.com" });
    expect(provider.name).toBe("http");
  });

  it("reads LUX_DATA_PROVIDER=http from env", async () => {
    process.env.LUX_DATA_PROVIDER = "http";
    process.env.LUX_API_BASE_URL = "https://example.com";
    const provider = await createProvider();
    expect(provider.name).toBe("http");
  });

  it("reads LUX_FIXTURES_ROOT from env for filesystem", async () => {
    process.env.LUX_FIXTURES_ROOT = "/tmp/fixtures";
    const provider = await createProvider();
    expect(provider.name).toBe("filesystem");
  });

  it("throws when filesystem provider has no fixturesRoot", async () => {
    await expect(createProvider({ kind: "filesystem" })).rejects.toThrow("fixturesRoot");
  });

  it("throws when http provider has no apiBaseUrl", async () => {
    await expect(createProvider({ kind: "http" })).rejects.toThrow("apiBaseUrl");
  });

  it("config.kind overrides LUX_DATA_PROVIDER env", async () => {
    process.env.LUX_DATA_PROVIDER = "http";
    const provider = await createProvider({ kind: "filesystem", fixturesRoot: "/tmp/fixtures" });
    expect(provider.name).toBe("filesystem");
  });
});
