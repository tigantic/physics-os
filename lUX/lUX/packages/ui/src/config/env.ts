import path from "node:path";

/**
 * Typed, validated environment configuration for the UI package.
 * All env vars are read once at module level and validated.
 * Import as: import { env } from "@/config/env";
 */
function requireEnv() {
  const fixturesRoot =
    process.env.LUX_FIXTURES_ROOT?.trim() || path.resolve(process.cwd(), "..", "core", "tests", "fixtures");

  const baseUrl = process.env.LUX_BASE_URL?.trim() || "http://localhost:3000";

  const revalidate = parseInt(process.env.LUX_REVALIDATE ?? "0", 10);

  const apiBaseUrl = process.env.LUX_API_BASE_URL?.trim() || undefined;

  return Object.freeze({
    /** Absolute path to the fixtures directory containing proof-packages/ and domain-packs/. */
    fixturesRoot: path.resolve(fixturesRoot),
    /** Base URL for canonical links and Open Graph tags. */
    baseUrl,
    /** ISR revalidation interval in seconds. 0 = no cache. */
    revalidate: Number.isFinite(revalidate) && revalidate >= 0 ? revalidate : 0,
    /** Whether running in CI. */
    isCI: process.env.CI === "true",
    /** API base URL for HttpProvider. Undefined = use FilesystemProvider. */
    apiBaseUrl,
  });
}

export const env = requireEnv();
