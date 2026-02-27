import { describe, it, expect, beforeAll, afterAll } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { loadProofPackageFromDir } from "../src/proof/proofPackageLoader.js";

describe("proofPackageLoader", () => {
  let tmpDir: string;

  beforeAll(async () => {
    tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "proof-loader-test-"));
  });

  afterAll(async () => {
    await fs.rm(tmpDir, { recursive: true, force: true });
  });

  it("throws descriptive error when proofPackage.json is missing", async () => {
    const emptyDir = path.join(tmpDir, "empty");
    await fs.mkdir(emptyDir, { recursive: true });

    await expect(loadProofPackageFromDir(emptyDir)).rejects.toThrow("proofPackage.json missing");
  });

  it("throws descriptive error for invalid JSON", async () => {
    const badJsonDir = path.join(tmpDir, "bad-json");
    await fs.mkdir(badJsonDir, { recursive: true });
    await fs.writeFile(path.join(badJsonDir, "proofPackage.json"), "{not valid json!!!");

    await expect(loadProofPackageFromDir(badJsonDir)).rejects.toThrow("Invalid JSON in proofPackage.json");
  });

  it("throws a Zod validation error for valid JSON that doesn't match the schema", async () => {
    const badSchemaDir = path.join(tmpDir, "bad-schema");
    await fs.mkdir(badSchemaDir, { recursive: true });
    await fs.writeFile(path.join(badSchemaDir, "proofPackage.json"), JSON.stringify({ hello: "world" }));

    await expect(loadProofPackageFromDir(badSchemaDir)).rejects.toThrow();
  });
});
