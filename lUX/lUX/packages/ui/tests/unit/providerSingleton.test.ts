import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock server-only which provider.ts imports
vi.mock("server-only", () => ({}));

// Mock createProvider from @luxury/core
const mockProvider = {
  name: "filesystem",
  listPackages: vi.fn(),
  loadPackage: vi.fn(),
  loadDomainPack: vi.fn(),
  readArtifact: vi.fn(),
};

vi.mock("@luxury/core", async (importOriginal) => {
  const mod = (await importOriginal()) as Record<string, unknown>;
  return {
    ...mod,
    createProvider: vi.fn(() => Promise.resolve(mockProvider)),
  };
});

describe("getProvider", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.clearAllMocks();
  });

  it("returns a ProofDataProvider", async () => {
    const { getProvider } = await import("@/config/provider");
    const provider = await getProvider();
    expect(provider).toBeDefined();
    expect(provider.name).toBe("filesystem");
  });

  it("returns the same instance on subsequent calls", async () => {
    const { getProvider } = await import("@/config/provider");
    const p1 = await getProvider();
    const p2 = await getProvider();
    expect(p1).toBe(p2);
  });

  it("resetProvider clears the singleton", async () => {
    const { getProvider, resetProvider } = await import("@/config/provider");
    await getProvider();
    resetProvider();
    // After reset, createProvider is called again on next getProvider
    const { createProvider } = await import("@luxury/core");
    const callCount = vi.mocked(createProvider).mock.calls.length;
    await getProvider();
    expect(vi.mocked(createProvider).mock.calls.length).toBe(callCount + 1);
  });
});
