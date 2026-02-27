import { describe, it, expect, beforeAll, afterAll } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { readArtifactBytes } from "../src/proof/artifactStore.js";

describe("readArtifactBytes", () => {
  let tmpDir: string;

  beforeAll(async () => {
    tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "artifact-store-test-"));
    const artifactsDir = path.join(tmpDir, "artifacts");
    await fs.mkdir(artifactsDir, { recursive: true });
    await fs.writeFile(path.join(artifactsDir, "data.csv"), "x,y\n1,2\n");
  });

  afterAll(async () => {
    await fs.rm(tmpDir, { recursive: true, force: true });
  });

  it("reads a valid artifact and returns bytes with hash", async () => {
    const result = await readArtifactBytes(tmpDir, "artifacts/data.csv");
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.bytes).toBeInstanceOf(Buffer);
    expect(result.hash).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(new TextDecoder().decode(result.bytes)).toBe("x,y\n1,2\n");
  });

  it("rejects path traversal with ../", async () => {
    const result = await readArtifactBytes(tmpDir, "../../etc/passwd");
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.reason).toContain("Path traversal rejected");
  });

  it("rejects absolute path traversal", async () => {
    const result = await readArtifactBytes(tmpDir, "/etc/passwd");
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.reason).toContain("Path traversal rejected");
  });

  it("rejects double-dot in the middle of a path", async () => {
    const result = await readArtifactBytes(tmpDir, "artifacts/../../../etc/shadow");
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.reason).toContain("Path traversal rejected");
  });

  it("returns missing reason for nonexistent file", async () => {
    const result = await readArtifactBytes(tmpDir, "artifacts/no-such-file.bin");
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.reason).toBe("Artifact file missing");
  });

  it("allows valid nested relative URI", async () => {
    const result = await readArtifactBytes(tmpDir, "artifacts/data.csv");
    expect(result.ok).toBe(true);
  });
});
