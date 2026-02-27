import { describe, it, expect } from "vitest";
import path from "node:path";
import { loadProofPackageFromDir } from "../src/proof/proofPackageLoader.js";

describe("integrity", () => {
  it("detects BROKEN_CHAIN for tampered fixture", async () => {
    const dir = path.resolve(__dirname, "fixtures", "proof-packages", "tampered");
    const loaded = await loadProofPackageFromDir(dir);
    expect(loaded.proof.verification?.status).toBe("BROKEN_CHAIN");
  });
});
