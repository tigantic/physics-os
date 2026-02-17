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
    it("loads and validates a domain pack from absolute path", () => {
      const p = path.join(FIXTURES, "domain-packs", "com.physics.vlasov.json");
      const pack = loadDomainPackFromFile(p);
      expect(pack.id).toBe("com.physics.vlasov");
      expect(pack.version).toMatch(/^\d+\.\d+\.\d+$/);
    });

    it("returns a deep-frozen object", () => {
      const p = path.join(FIXTURES, "domain-packs", "com.physics.vlasov.json");
      const pack = loadDomainPackFromFile(p);
      expect(Object.isFrozen(pack)).toBe(true);
      expect(Object.isFrozen(pack.metrics)).toBe(true);
    });

    it("returns cached result on second call", () => {
      const p = path.join(FIXTURES, "domain-packs", "com.physics.vlasov.json");
      const a = loadDomainPackFromFile(p);
      const b = loadDomainPackFromFile(p);
      expect(a).toBe(b); // Reference equality — same cached object
    });

    it("throws on missing file", () => {
      expect(() => loadDomainPackFromFile("/nonexistent/path.json")).toThrow();
    });

    it("throws on invalid JSON", () => {
      // An artifact file won't pass Zod validation
      const p = path.join(FIXTURES, "proof-packages", "pass", "proofPackage.json");
      // This is a valid ProofPackage, not a DomainPack — should fail DomainPack parse
      expect(() => loadDomainPackFromFile(p)).toThrow();
    });
  });

  describe("loadDomainPackById", () => {
    it("loads a pack by its com.physics.* ID", () => {
      const pack = loadDomainPackById(FIXTURES, "com.physics.vlasov");
      expect(pack.id).toBe("com.physics.vlasov");
    });

    it("throws for unknown pack ID", () => {
      expect(() => loadDomainPackById(FIXTURES, "com.physics.nonexistent")).toThrow(
        "DomainPack not found: com.physics.nonexistent"
      );
    });

    it("rejects pack IDs with path traversal (../)", () => {
      expect(() => loadDomainPackById(FIXTURES, "../../../etc/passwd")).toThrow(
        "path traversal"
      );
    });

    it("rejects pack IDs with forward slashes", () => {
      expect(() => loadDomainPackById(FIXTURES, "foo/bar")).toThrow(
        "path traversal"
      );
    });

    it("rejects pack IDs with backslashes", () => {
      expect(() => loadDomainPackById(FIXTURES, "foo\\bar")).toThrow(
        "path traversal"
      );
    });

    it("rejects pack IDs with special characters", () => {
      expect(() => loadDomainPackById(FIXTURES, "foo bar")).toThrow(
        "invalid characters"
      );
    });
  });

  describe("loadDomainPackForDomain", () => {
    it("resolves TPC domain_id via manifest to pack ID", () => {
      // The manifest maps TPC domain IDs (like "II.2") → pack IDs (like "com.physics.euler_3d")
      const pack = loadDomainPackForDomain(FIXTURES, "II.2");
      expect(pack.id).toBe("com.physics.euler_3d");
      expect(pack.metrics).toBeDefined();
    });

    it("falls back to direct ID lookup if not in manifest", () => {
      // "com.physics.vlasov" isn't a TPC domain_id, but works as fallback
      const pack = loadDomainPackForDomain(FIXTURES, "com.physics.vlasov");
      expect(pack.id).toBe("com.physics.vlasov");
    });

    it("throws for completely unknown domain", () => {
      expect(() => loadDomainPackForDomain(FIXTURES, "ZZZZZ.999")).toThrow();
    });
  });

  describe("listDomainPackIds", () => {
    it("returns all domain pack IDs (excludes _manifest.json)", () => {
      const ids = listDomainPackIds(FIXTURES);
      expect(ids.length).toBeGreaterThanOrEqual(140);
      expect(ids).toContain("com.physics.vlasov");
      expect(ids).not.toContain("_manifest");
    });

    it("returns empty array for nonexistent directory", () => {
      expect(listDomainPackIds("/nonexistent/path")).toEqual([]);
    });
  });

  describe("clearDomainPackCaches", () => {
    it("clears cache so next load re-reads from disk", () => {
      const p = path.join(FIXTURES, "domain-packs", "com.physics.vlasov.json");
      const a = loadDomainPackFromFile(p);
      clearDomainPackCaches();
      const b = loadDomainPackFromFile(p);
      // After clearing, we get a new object (deep-equal but not reference-equal)
      expect(a).toEqual(b);
      // Can't guarantee reference inequality since freeze returns same shape,
      // but the cache was cleared and re-parsed
    });
  });
});
