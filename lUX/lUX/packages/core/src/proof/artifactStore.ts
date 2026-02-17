import fs from "node:fs";
import path from "node:path";
import { sha256Prefixed } from "../util/hash.js";

export type ArtifactReadResult =
  | { ok: true; bytes: Uint8Array; hash: `sha256:${string}` }
  | { ok: false; reason: string };

export function readArtifactBytes(bundleDir: string, uri: string): ArtifactReadResult {
  const p = path.resolve(bundleDir, uri);
  if (!fs.existsSync(p)) return { ok: false, reason: "Artifact file missing" };
  const bytes = fs.readFileSync(p);
  return { ok: true, bytes, hash: sha256Prefixed(bytes) };
}
