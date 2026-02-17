import fs from "node:fs/promises";
import path from "node:path";
import { sha256Prefixed } from "../util/hash.js";

export type ArtifactReadResult =
  | { ok: true; bytes: Uint8Array; hash: `sha256:${string}` }
  | { ok: false; reason: string };

export async function readArtifactBytes(bundleDir: string, uri: string): Promise<ArtifactReadResult> {
  const p = path.resolve(bundleDir, uri);
  let bytes: Buffer;
  try {
    bytes = await fs.readFile(p);
  } catch {
    return { ok: false, reason: "Artifact file missing" };
  }
  return { ok: true, bytes, hash: sha256Prefixed(bytes) };
}
