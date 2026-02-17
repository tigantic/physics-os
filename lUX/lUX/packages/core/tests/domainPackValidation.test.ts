import { describe, it, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { DomainPackSchema } from "../src/schema/domainPack.zod.js";

/**
 * Validates every generated domain pack against the DomainPack Zod schema.
 * This ensures the generator produces conformant output.
 */

const DOMAIN_PACKS_DIR = path.resolve(__dirname, "fixtures", "domain-packs");

function getPackFiles(): string[] {
  return fs
    .readdirSync(DOMAIN_PACKS_DIR)
    .filter((f) => f.endsWith(".json") && !f.startsWith("_"))
    .sort();
}

describe("domain pack validation — all 140 packs", () => {
  const files = getPackFiles();

  it("has at least 140 domain packs", () => {
    expect(files.length).toBeGreaterThanOrEqual(140);
  });

  it.each(files)("%s validates against DomainPackSchema", (file) => {
    const filePath = path.join(DOMAIN_PACKS_DIR, file);
    const raw: unknown = JSON.parse(fs.readFileSync(filePath, "utf8"));
    const result = DomainPackSchema.safeParse(raw);
    if (!result.success) {
      throw new Error(
        `${file} failed validation:\n${result.error.issues.map((i) => `  ${i.path.join(".")}: ${i.message}`).join("\n")}`
      );
    }
    // Verify pack ID matches filename
    const expectedId = file.replace(/\.json$/, "");
    expect(result.data.id).toBe(expectedId);
  });

  it("all packs have at least 2 metrics (conservation_residual + l2_drift)", () => {
    for (const file of files) {
      const raw = JSON.parse(fs.readFileSync(path.join(DOMAIN_PACKS_DIR, file), "utf8"));
      const pack = DomainPackSchema.parse(raw);
      expect(Object.keys(pack.metrics).length).toBeGreaterThanOrEqual(2);
      expect(pack.metrics["conservation_residual"]).toBeDefined();
      expect(pack.metrics["l2_drift"]).toBeDefined();
    }
  });

  it("all packs have at least one gate pack", () => {
    for (const file of files) {
      const raw = JSON.parse(fs.readFileSync(path.join(DOMAIN_PACKS_DIR, file), "utf8"));
      const pack = DomainPackSchema.parse(raw);
      expect(Object.keys(pack.gate_packs).length).toBeGreaterThanOrEqual(1);
    }
  });

  it("all packs have standard viewers", () => {
    for (const file of files) {
      const raw = JSON.parse(fs.readFileSync(path.join(DOMAIN_PACKS_DIR, file), "utf8"));
      const pack = DomainPackSchema.parse(raw);
      const viewerComponents = pack.viewers.map((v) => v.component);
      expect(viewerComponents).toContain("TimeSeriesViewer");
      expect(viewerComponents).toContain("LogViewer");
      expect(viewerComponents).toContain("TableViewer");
    }
  });

  it("executive_summary_metric_ids reference existing metrics", () => {
    for (const file of files) {
      const raw = JSON.parse(fs.readFileSync(path.join(DOMAIN_PACKS_DIR, file), "utf8"));
      const pack = DomainPackSchema.parse(raw);
      for (const mid of pack.templates.executive_summary_metric_ids) {
        expect(pack.metrics[mid]).toBeDefined();
      }
    }
  });
});

describe("domain pack manifest", () => {
  it("_manifest.json exists and maps all 140 TPC domain IDs", () => {
    const manifestPath = path.join(DOMAIN_PACKS_DIR, "_manifest.json");
    expect(fs.existsSync(manifestPath)).toBe(true);
    const manifest: Record<string, string> = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
    expect(Object.keys(manifest).length).toBe(140);
  });

  it("every manifest entry points to an existing pack file", () => {
    const manifestPath = path.join(DOMAIN_PACKS_DIR, "_manifest.json");
    const manifest: Record<string, string> = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
    for (const [domainId, packId] of Object.entries(manifest)) {
      const packPath = path.join(DOMAIN_PACKS_DIR, `${packId}.json`);
      expect(fs.existsSync(packPath), `${domainId} → ${packId} missing`).toBe(true);
    }
  });
});
