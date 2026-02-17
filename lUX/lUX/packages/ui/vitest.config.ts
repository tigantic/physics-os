import { defineConfig } from "vitest/config";
import path from "node:path";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
      "@luxury/core": path.resolve(__dirname, "..", "core", "src", "index.ts"),
    },
  },
  test: {
    environment: "jsdom",
    include: ["tests/unit/**/*.test.ts", "tests/unit/**/*.test.tsx"],
    setupFiles: ["tests/setup.ts"],
  },
});
