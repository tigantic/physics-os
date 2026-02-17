import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// Mock server-only globally so any module that imports it (env.ts, provider.ts,
// logger.ts, metrics.ts, etc.) can be loaded in vitest without throwing.
vi.mock("server-only", () => ({}));

afterEach(() => {
  cleanup();
});
