import { describe, it, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { ProofPackageSchema } from "../src/schema/proofPackage.zod.js";
import { DomainPackSchema } from "../src/schema/domainPack.zod.js";

function readJson(p: string) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

describe("schema", () => {
  it("parses ProofPackage fixture", () => {
    const p = path.resolve(__dirname, "fixtures", "proof-packages", "pass", "proofPackage.json");
    expect(() => ProofPackageSchema.parse(readJson(p))).not.toThrow();
  });

  it("parses DomainPack fixture", () => {
    const p = path.resolve(__dirname, "fixtures", "domain-packs", "com.physics.vlasov.json");
    expect(() => DomainPackSchema.parse(readJson(p))).not.toThrow();
  });
});
