import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// Mock server-only globally so any module that imports it (env.ts, provider.ts,
// logger.ts, metrics.ts, etc.) can be loaded in vitest without throwing.
vi.mock("server-only", () => ({}));

// Mock window.matchMedia for jsdom (used by ThemeToggle and media queries)
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

afterEach(() => {
  cleanup();
});
