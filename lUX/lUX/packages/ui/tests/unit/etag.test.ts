import { describe, it, expect, vi } from "vitest";

// Mock server-only
vi.mock("server-only", () => ({}));

import { computeETag, isNotModified } from "@/lib/etag";

describe("computeETag", () => {
  it("returns a weak ETag string", () => {
    const etag = computeETag("hello world");
    expect(etag).toMatch(/^W\/"[a-f0-9]{16}"$/);
  });

  it("produces deterministic output for same input", () => {
    const a = computeETag("test data");
    const b = computeETag("test data");
    expect(a).toBe(b);
  });

  it("produces different tags for different input", () => {
    const a = computeETag("data-1");
    const b = computeETag("data-2");
    expect(a).not.toBe(b);
  });

  it("accepts an object and JSON-stringifies it", () => {
    const etag = computeETag({ key: "value", num: 42 });
    expect(etag).toMatch(/^W\/"[a-f0-9]{16}"$/);
  });

  it("accepts a Buffer", () => {
    const etag = computeETag(Buffer.from("binary data"));
    expect(etag).toMatch(/^W\/"[a-f0-9]{16}"$/);
  });

  it("object order affects ETag (JSON.stringify is deterministic)", () => {
    const a = computeETag({ x: 1, y: 2 });
    const b = computeETag({ y: 2, x: 1 });
    // JSON.stringify preserves insertion order, so these differ
    expect(a).not.toBe(b);
  });
});

describe("isNotModified", () => {
  it("returns false when no If-None-Match header present", () => {
    const req = new Request("http://localhost/api/test");
    expect(isNotModified(req, 'W/"abc123"')).toBe(false);
  });

  it("returns true when If-None-Match matches ETag exactly", () => {
    const req = new Request("http://localhost/api/test", {
      headers: { "If-None-Match": 'W/"abc123"' },
    });
    expect(isNotModified(req, 'W/"abc123"')).toBe(true);
  });

  it("returns false when If-None-Match does not match", () => {
    const req = new Request("http://localhost/api/test", {
      headers: { "If-None-Match": 'W/"different"' },
    });
    expect(isNotModified(req, 'W/"abc123"')).toBe(false);
  });

  it("handles comma-separated If-None-Match values", () => {
    const req = new Request("http://localhost/api/test", {
      headers: { "If-None-Match": 'W/"first", W/"abc123", W/"third"' },
    });
    expect(isNotModified(req, 'W/"abc123"')).toBe(true);
  });

  it("returns false for non-matching comma-separated list", () => {
    const req = new Request("http://localhost/api/test", {
      headers: { "If-None-Match": 'W/"first", W/"second"' },
    });
    expect(isNotModified(req, 'W/"abc123"')).toBe(false);
  });
});
