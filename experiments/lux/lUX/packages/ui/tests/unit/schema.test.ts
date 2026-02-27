import { describe, it, expect } from "vitest";
import path from "node:path";
import { loadProofPackageFromDir } from "@luxury/core";

describe("schema", () => {
  it("loads pass fixture without errors", async () => {
    const fixturesRoot = path.resolve(process.cwd(), "..", "core", "tests", "fixtures");
    const dir = path.resolve(fixturesRoot, "proof-packages", "pass");
    const loaded = await loadProofPackageFromDir(dir);
    expect(loaded.proof.meta.domain_id).toBeTruthy();
  });
});
