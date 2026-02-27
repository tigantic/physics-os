import crypto from "node:crypto";

export function sha256Hex(data: Uint8Array): string {
  return crypto.createHash("sha256").update(data).digest("hex");
}

export function sha256Prefixed(data: Uint8Array): `sha256:${string}` {
  return `sha256:${sha256Hex(data)}`;
}
