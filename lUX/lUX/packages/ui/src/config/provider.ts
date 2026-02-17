import "server-only";

import type { ProofDataProvider } from "@luxury/core";
import { createProvider } from "@luxury/core";
import { env } from "./env";

/**
 * Lazily-initialized server-side ProofDataProvider singleton.
 *
 * Shared across all server component renders within the same
 * Node.js process. The provider is created once on first access,
 * using environment configuration from `env.ts`.
 *
 * Import as: import { getProvider } from "@/config/provider";
 */
let _provider: ProofDataProvider | null = null;
let _providerPromise: Promise<ProofDataProvider> | null = null;

/**
 * Check if the data provider has been successfully initialized.
 * Used by /api/health and /api/ready for orchestrator probes.
 */
export function isProviderReady(): boolean {
  return _provider !== null;
}

export function getProvider(): Promise<ProofDataProvider> {
  if (_provider) return Promise.resolve(_provider);
  if (_providerPromise) return _providerPromise;

  _providerPromise = createProvider({
    fixturesRoot: env.fixturesRoot,
    apiBaseUrl: env.apiBaseUrl,
  }).then((p) => {
    _provider = p;
    return p;
  });

  return _providerPromise;
}

/**
 * Reset the provider singleton. Only use in tests.
 * @internal
 */
export function resetProvider(): void {
  _provider = null;
  _providerPromise = null;
}
