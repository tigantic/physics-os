import { describe, it, expect } from "vitest";
import { env } from "@/config/env";

describe("env", () => {
  it("env.fixturesRoot is a non-empty string", () => {
    expect(typeof env.fixturesRoot).toBe("string");
    expect(env.fixturesRoot.length).toBeGreaterThan(0);
  });

  it('env.baseUrl defaults to "http://localhost:3000"', () => {
    expect(env.baseUrl).toBe("http://localhost:3000");
  });

  it("env.revalidate defaults to 0", () => {
    expect(env.revalidate).toBe(0);
  });

  it("env.isCI is a boolean", () => {
    expect(typeof env.isCI).toBe("boolean");
  });

  it("env object is frozen", () => {
    expect(Object.isFrozen(env)).toBe(true);
  });
});
