import { describe, it, expect } from "vitest";
import { sha256Hex, sha256Prefixed } from "../src/util/hash.js";

describe("sha256Hex", () => {
  it("returns correct hash for known input", () => {
    // SHA-256 of empty buffer is well-known
    const empty = new Uint8Array(0);
    expect(sha256Hex(empty)).toBe("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  });

  it("returns correct hash for 'hello'", () => {
    const data = new TextEncoder().encode("hello");
    expect(sha256Hex(data)).toBe("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
  });

  it("returns a 64-character hex string", () => {
    const data = new TextEncoder().encode("test data");
    const result = sha256Hex(data);
    expect(result).toHaveLength(64);
    expect(result).toMatch(/^[0-9a-f]{64}$/);
  });

  it("produces different hashes for different inputs", () => {
    const a = sha256Hex(new TextEncoder().encode("alpha"));
    const b = sha256Hex(new TextEncoder().encode("beta"));
    expect(a).not.toBe(b);
  });
});

describe("sha256Prefixed", () => {
  it("returns sha256: prefix", () => {
    const data = new Uint8Array(0);
    const result = sha256Prefixed(data);
    expect(result).toMatch(/^sha256:[0-9a-f]{64}$/);
  });

  it("matches sha256Hex output with prefix", () => {
    const data = new TextEncoder().encode("verify");
    expect(sha256Prefixed(data)).toBe(`sha256:${sha256Hex(data)}`);
  });
});
