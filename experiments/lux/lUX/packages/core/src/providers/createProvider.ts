import type { ProofDataProvider } from "./types.js";

export type ProviderKind = "filesystem" | "http";

export interface ProviderConfig {
  /** Provider type. Defaults to "filesystem". */
  kind?: ProviderKind;
  /** Absolute path to fixtures root (required for "filesystem"). */
  fixturesRoot?: string;
  /** API base URL (required for "http"). */
  apiBaseUrl?: string;
  /** Optional HTTP headers for the "http" provider (e.g. Authorization). */
  httpHeaders?: Record<string, string>;
}

/**
 * Create a ProofDataProvider based on configuration.
 *
 * Provider selection priority:
 *   1. Explicit `config.kind`
 *   2. `LUX_DATA_PROVIDER` environment variable
 *   3. Default: "filesystem"
 *
 * @throws {Error} If required configuration is missing for the selected provider.
 */
export async function createProvider(config: ProviderConfig = {}): Promise<ProofDataProvider> {
  const kind = resolveKind(config.kind);

  switch (kind) {
    case "filesystem": {
      const root = config.fixturesRoot ?? process.env.LUX_FIXTURES_ROOT?.trim();
      if (!root) {
        throw new Error(
          "FilesystemProvider requires fixturesRoot. " + "Set LUX_FIXTURES_ROOT env var or pass config.fixturesRoot.",
        );
      }
      const { FilesystemProvider } = await import("./FilesystemProvider.js");
      return new FilesystemProvider(root);
    }
    case "http": {
      const baseUrl = config.apiBaseUrl ?? process.env.LUX_API_BASE_URL?.trim();
      if (!baseUrl) {
        throw new Error(
          "HttpProvider requires apiBaseUrl. " + "Set LUX_API_BASE_URL env var or pass config.apiBaseUrl.",
        );
      }
      const { HttpProvider } = await import("./HttpProvider.js");
      return new HttpProvider(baseUrl, config.httpHeaders);
    }
    default: {
      const _exhaustive: never = kind;
      throw new Error(`Unknown provider kind: ${_exhaustive}`);
    }
  }
}

function resolveKind(explicit?: ProviderKind): ProviderKind {
  if (explicit) return explicit;
  const env = process.env.LUX_DATA_PROVIDER?.trim().toLowerCase();
  if (env === "http") return "http";
  return "filesystem";
}
