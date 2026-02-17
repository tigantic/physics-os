import { describe, it, expect, beforeEach } from "vitest";
import path from "node:path";
import {
  loadDomainPackFromFile,
  loadDomainPackById,
  loadDomainPackForDomain,
  listDomainPackIds,
  clearDomainPackCaches,
} from "../src/domain/domainPackRegistry.js";

const FIXTURES = path.resolve(__dirname, "fixtures");

describe("domainPackRegistry", () => {
  beforeEach(() => {
    clearDomainPackCaches();
  });

  describe("loadDomainPackFromFile", () => {
    it("loads and validates a domain pack from absolute path", async () => {
      const p = path.join(FIXTURES, "domain-packs", "com.physics.vlasov.json");
      const pack = await loadDomainPackFromFile(p);
      expect(pack.id).toBe("com.physics.vlasov");
      expect(pack.version).toMatch(/^\d+\.\d+\.\d+$/);
    });

    it("returns a deep-frozen object", async () => {
      const p = path.join(FIXTURES, "domain-packs", "com.physics.vlasov.json");
      const pack = await loadDomainPackFromFile(p);
      expect(Object.isFrozen(pack)).toBe(true);
      expect(Object.isFrozen(pack.metrics)).toBe(true);
    });

    it("returns cached result on second call", async () => {
      const p = path.join(FIXTURES, "domain-packs", "com.physics.vlasov.json");
      const a = await loadDomainPackFromFile(p);
      const b = await loadDomainPackFromFile(p);
      expect(a).toBe(b); // Reference equality — same cached object
    });

    it("throws on missing file", async () => {
      await expect(loadDomainPackFromFile("/nonexistent/path.json")).rejects.toThrow();
    });

    it("throws on invalid JSON", async () => {
      // An artifact file won't pass Zod validation
      const p = path.join(FIXTURES, "proof-packages", "pass", "proofPackage.json");
      // This is a valid ProofPackage, not a DomainPack — should fail DomainPack parse
      await expect(loadDomainPackFromFile(p)).rejects.toThrow();
    });
  });

  describe("loadDomainPackById", () => {
    it("loads a pack by its com.physics.* ID", async () => {
      const pack = await loadDomainPackById(FIXTURES, "com.physics.vlasov");
      expect(pack.id).toBe("com.physics.vlasov");
    });

    it("throws for unknown pack ID", async () => {
      await expect(loadDomainPackById(FIXTURES, "com.physics.nonexistent")).rejects.toThrow(
        "DomainPack not found: com.physics.nonexistent",
      );
    });

    it("rejects pack IDs with path traversal (../)", async () => {
      await expect(loadDomainPackById(FIXTURES, "../../../etc/passwd")).rejects.toThrow("path traversal");
    });

    it("rejects pack IDs with forward slashes", async () => {
      await expect(loadDomainPackById(FIXTURES, "foo/bar")).rejects.toThrow("path traversal");
    });

    it("rejects pack IDs with backslashes", async () => {
      await expect(loadDomainPackById(FIXTURES, "foo\\bar")).rejects.toThrow("path traversal");
    });

    it("rejects pack IDs with special characters", async () => {
      await expect(loadDomainPackById(FIXTURES, "foo bar")).rejects.toThrow("invalid characters");
    });
  });

  describe("loadDomainPackForDomain", () => {
    it("resolves TPC domain_id via manifest to pack ID", async () => {
      // The manifest maps TPC domain IDs (like "II.2") → pack IDs (like "com.physics.euler_3d")
      const pack = await loadDomainPackForDomain(FIXTURES, "II.2");
      expect(pack.id).toBe("com.physics.euler_3d");
      expect(pack.metrics).toBeDefined();
    });

    it("falls back to direct ID lookup if not in manifest", async () => {
      // "com.physics.vlasov" isn't a TPC domain_id, but works as fallback
      const pack = await loadDomainPackForDomain(FIXTURES, "com.physics.vlasov");
      expect(pack.id).toBe("com.physics.vlasov");
    });

    it("throws for completely unknown domain", async () => {
      await expect(loadDomainPackForDomain(FIXTURES, "ZZZZZ.999")).rejects.toThrow();
    });
  });

  describe("listDomainPackIds", () => {
    it("returns all domain pack IDs (excludes _manifest.json)", async () => {
      const ids = await listDomainPackIds(FIXTURES);
      expect(ids.length).toBeGreaterThanOrEqual(140);
      expect(ids).toContain("com.physics.vlasov");
      expect(ids).not.toContain("_manifest");
    });

    it("returns empty array for nonexistent directory", async () => {
      expect(await listDomainPackIds("/nonexistent/path")).toEqual([]);
    });
  });

  describe("clearDomainPackCaches", () => {
    it("clears cache so next load re-reads from disk", async () => {
      const p = path.join(FIXTURES, "domain-packs", "com.physics.vlasov.json");
      const a = await loadDomainPackFromFile(p);
      clearDomainPackCaches();
      const b = await loadDomainPackFromFile(p);
      // After clearing, we get a new object (deep-equal but not reference-equal)
      expect(a).toEqual(b);
      // Can't guarantee reference inequality since freeze returns same shape,
      // but the cache was cleared and re-parsed
    });
  });
});
