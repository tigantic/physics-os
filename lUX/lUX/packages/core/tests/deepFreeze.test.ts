import { describe, it, expect } from "vitest";
import { deepFreeze } from "../src/util/deepFreeze.js";

describe("deepFreeze", () => {
  it("freezes a flat object", () => {
    const obj = { a: 1, b: "hello" };
    const result = deepFreeze(obj);
    expect(Object.isFrozen(result)).toBe(true);
    expect(result).toBe(obj); // same reference
  });

  it("freezes nested objects recursively", () => {
    const obj = { a: { b: { c: 42 } } };
    deepFreeze(obj);
    expect(Object.isFrozen(obj)).toBe(true);
    expect(Object.isFrozen(obj.a)).toBe(true);
    expect(Object.isFrozen(obj.a.b)).toBe(true);
  });

  it("freezes arrays and their contents", () => {
    const obj = { items: [{ id: 1 }, { id: 2 }] };
    deepFreeze(obj);
    expect(Object.isFrozen(obj.items)).toBe(true);
    expect(Object.isFrozen(obj.items[0])).toBe(true);
    expect(Object.isFrozen(obj.items[1])).toBe(true);
  });

  it("handles already-frozen input without error", () => {
    const inner = Object.freeze({ x: 10 });
    const obj = { nested: inner };
    expect(() => deepFreeze(obj)).not.toThrow();
    expect(Object.isFrozen(obj)).toBe(true);
  });

  it("returns primitives unchanged", () => {
    expect(deepFreeze(42)).toBe(42);
    expect(deepFreeze("str")).toBe("str");
    expect(deepFreeze(null)).toBe(null);
    expect(deepFreeze(undefined)).toBe(undefined);
    expect(deepFreeze(true)).toBe(true);
  });

  it("returns empty object frozen", () => {
    const obj = {};
    deepFreeze(obj);
    expect(Object.isFrozen(obj)).toBe(true);
  });

  it("prevents mutation after freezing", () => {
    const obj = deepFreeze({ a: 1, nested: { b: 2 } });
    expect(() => {
      (obj as Record<string, unknown>).a = 999;
    }).toThrow();
    expect(() => {
      (obj.nested as Record<string, unknown>).b = 999;
    }).toThrow();
  });
});
