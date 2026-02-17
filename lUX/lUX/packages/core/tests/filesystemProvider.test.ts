import { describe, it, expect, beforeAll } from "vitest";
import path from "node:path";
import { FilesystemProvider } from "../src/providers/FilesystemProvider.js";

const FIXTURES_ROOT = path.resolve(__dirname, "fixtures");

describe("FilesystemProvider", () => {
  let provider: FilesystemProvider;

  beforeAll(() => {
    provider = new FilesystemProvider(FIXTURES_ROOT);
  });

  describe("listPackages", () => {
    it("returns summaries for all fixture packages", async () => {
      const packages = await provider.listPackages();
      expect(packages.length).toBeGreaterThanOrEqual(5);

      const ids = packages.map((p) => p.id);
      expect(ids).toContain("pass");
      expect(ids).toContain("fail");
      expect(ids).toContain("warn");
      expect(ids).toContain("incomplete");
      expect(ids).toContain("tampered");
    });

    it("each summary has required fields", async () => {
      const packages = await provider.listPackages();
      for (const pkg of packages) {
        expect(typeof pkg.id).toBe("string");
        expect(typeof pkg.domain_id).toBe("string");
        expect(typeof pkg.verdict_status).toBe("string");
        expect(typeof pkg.quality_score).toBe("number");
        expect(typeof pkg.timestamp).toBe("string");
        expect(typeof pkg.solver_name).toBe("string");
      }
    });

    it("returns empty array for nonexistent root", async () => {
      const p = new FilesystemProvider("/nonexistent/path");
      const packages = await p.listPackages();
      expect(packages).toEqual([]);
    });
  });

  describe("loadPackage", () => {
    it("loads and validates a valid proof package", async () => {
      const proof = await provider.loadPackage("pass");
      expect(proof.schema_version).toBe("1.0.0");
      expect(proof.meta.id).toBeTruthy();
      expect(proof.verdict.status).toBe("PASS");
      expect(proof.verification).toBeDefined();
    });

    it("includes verification results", async () => {
      const proof = await provider.loadPackage("pass");
      expect(proof.verification?.status).toBe("VERIFIED");
      expect(proof.verification?.failures).toEqual([]);
    });

    it("rejects path traversal in package ID", async () => {
      await expect(provider.loadPackage("../etc/passwd")).rejects.toThrow("path traversal");
    });

    it("rejects invalid characters in package ID", async () => {
      await expect(provider.loadPackage("pass!@#")).rejects.toThrow("invalid characters");
    });

    it("throws for nonexistent package", async () => {
      await expect(provider.loadPackage("nonexistent-package")).rejects.toThrow("not found");
    });

    it("returns deep-frozen object", async () => {
      const proof = await provider.loadPackage("pass");
      expect(Object.isFrozen(proof)).toBe(true);
      expect(Object.isFrozen(proof.meta)).toBe(true);
    });
  });

  describe("loadDomainPack", () => {
    it("loads a domain pack by pack ID", async () => {
      const pack = await provider.loadDomainPack("com.physics.vlasov");
      expect(pack.id).toBe("com.physics.vlasov");
      expect(pack.version).toBeTruthy();
      expect(pack.metrics).toBeDefined();
    });

    it("throws for nonexistent domain", async () => {
      await expect(provider.loadDomainPack("nonexistent.domain")).rejects.toThrow();
    });
  });

  describe("readArtifact", () => {
    it("reads a valid artifact", async () => {
      const proof = await provider.loadPackage("pass");
      const firstArt = Object.values(proof.artifacts)[0];
      if (!firstArt) return; // Skip if no artifacts

      const result = await provider.readArtifact("pass", firstArt.uri);
      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.bytes.byteLength).toBeGreaterThan(0);
        expect(result.hash).toMatch(/^sha256:[a-f0-9]{64}$/i);
        expect(typeof result.mimeType).toBe("string");
      }
    });

    it("returns mimeType based on file extension", async () => {
      const proof = await provider.loadPackage("pass");
      const csvArt = Object.values(proof.artifacts).find((a) => a.uri.endsWith(".csv"));
      if (!csvArt) return;

      const result = await provider.readArtifact("pass", csvArt.uri);
      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.mimeType).toBe("text/csv");
      }
    });

    it("rejects path traversal in package ID", async () => {
      await expect(provider.readArtifact("../etc", "passwd")).rejects.toThrow("path traversal");
    });

    it("returns failure for missing artifact", async () => {
      const result = await provider.readArtifact("pass", "artifacts/nonexistent.csv");
      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.reason).toBeTruthy();
      }
    });
  });

  describe("name", () => {
    it('returns "filesystem"', () => {
      expect(provider.name).toBe("filesystem");
    });
  });
});
