import "server-only";

import { createHash } from "node:crypto";

/**
 * Compute a weak ETag from a JSON-serialisable body.
 *
 * Uses SHA-256 truncated to 16 hex characters (64-bit collision space),
 * wrapped in the weak validator syntax `W/"..."`.
 *
 * @param body — The response body (string or Buffer). Objects are JSON-stringified.
 * @returns ETag value suitable for the `ETag` response header.
 */
export function computeETag(body: string | Buffer | Record<string, unknown>): string {
  const raw = typeof body === "string" ? body : body instanceof Buffer ? body : JSON.stringify(body);
  const hash = createHash("sha256").update(raw).digest("hex").slice(0, 16);
  return `W/"${hash}"`;
}

/**
 * Check if the client's `If-None-Match` header matches the current ETag.
 *
 * @param request — The incoming request.
 * @param etag — The computed ETag value.
 * @returns `true` if the client already has a fresh copy (respond with 304).
 */
export function isNotModified(request: Request, etag: string): boolean {
  const ifNoneMatch = request.headers.get("if-none-match");
  if (!ifNoneMatch) return false;
  // Handle both exact match and comma-separated list
  return ifNoneMatch.split(",").some((t) => t.trim() === etag);
}
