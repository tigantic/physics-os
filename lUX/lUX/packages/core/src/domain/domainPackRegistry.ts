import fs from "node:fs/promises";
import path from "node:path";
import { DomainPackSchema } from "../schema/domainPack.zod.js";
import type { DomainPack } from "../schema/domainPack.zod.js";
import { deepFreeze } from "../util/deepFreeze.js";

/** LRU cache for loaded domain packs — avoids repeated disk IO + Zod parse. */
const packCache = new Map<string, Promise<DomainPack>>();

/**
 * Load and validate a single DomainPack from a JSON file.
 * Result is deep-frozen and cached by absolute path.
 */
export async function loadDomainPackFromFile(filePath: string): Promise<DomainPack> {
  const abs = path.resolve(filePath);
  const cached = packCache.get(abs);
  if (cached) return cached;

  const promise = (async () => {
    const text = await fs.readFile(abs, "utf8");
    let raw: unknown;
    try {
      raw = JSON.parse(text);
    } catch (err) {
      throw new Error(`Invalid JSON in domain pack ${abs}: ${err instanceof Error ? err.message : String(err)}`, {
        cause: err,
      });
    }
    return deepFreeze(DomainPackSchema.parse(raw));
  })();
  packCache.set(abs, promise);
  try {
    return await promise;
  } catch (err) {
    packCache.delete(abs);
    throw err;
  }
}

/** Validate that a pack/domain ID contains only safe filesystem characters. */
function assertSafeId(id: string, label: string): void {
  if (/\.\.[\\/]/.test(id) || /[\\/]/.test(id)) {
    throw new Error(`${label} contains path traversal characters: ${id}`);
  }
  if (!/^[a-zA-Z0-9._-]+$/.test(id)) {
    throw new Error(`${label} contains invalid characters: ${id}`);
  }
}

/**
 * Load a domain pack by its pack ID (e.g. "com.physics.euler_3d").
 * Searches <fixturesRoot>/domain-packs/<packId>.json.
 */
export async function loadDomainPackById(fixturesRoot: string, packId: string): Promise<DomainPack> {
  assertSafeId(packId, "packId");
  const p = path.resolve(fixturesRoot, "domain-packs", `${packId}.json`);
  try {
    await fs.access(p);
  } catch {
    throw new Error(`DomainPack not found: ${packId}`);
  }
  return loadDomainPackFromFile(p);
}

/**
 * Load a domain pack by TPC domain_id (e.g. "II.2").
 * Uses _manifest.json to resolve domain_id → pack ID.
 */
export async function loadDomainPackForDomain(fixturesRoot: string, domainId: string): Promise<DomainPack> {
  assertSafeId(domainId, "domainId");
  const manifest = await loadManifest(fixturesRoot);
  const packId = manifest[domainId];
  if (!packId) {
    // Fallback: try using domainId directly as pack ID
    return loadDomainPackById(fixturesRoot, domainId);
  }
  return loadDomainPackById(fixturesRoot, packId);
}

/**
 * Load the _manifest.json mapping TPC domain_ids → pack IDs.
 * Cached after first load.
 */
let manifestCache: Record<string, string> | null = null;
let manifestRoot: string | null = null;

import { z } from "zod";

const ManifestSchema = z.record(z.string(), z.string());

async function loadManifest(fixturesRoot: string): Promise<Record<string, string>> {
  const root = path.resolve(fixturesRoot);
  if (manifestCache && manifestRoot === root) return manifestCache;

  const manifestPath = path.join(root, "domain-packs", "_manifest.json");
  let rawText: string;
  try {
    rawText = await fs.readFile(manifestPath, "utf8");
  } catch {
    return {};
  }

  let raw: unknown;
  try {
    raw = JSON.parse(rawText);
  } catch {
    throw new Error(`Invalid JSON in _manifest.json at ${manifestPath}`);
  }

  const result = ManifestSchema.safeParse(raw);
  if (!result.success) {
    throw new Error(`Invalid manifest schema in _manifest.json: ${result.error.message}`);
  }

  manifestCache = result.data;
  manifestRoot = root;
  return manifestCache;
}

/**
 * List all available domain pack IDs from the domain-packs directory.
 */
export async function listDomainPackIds(fixturesRoot: string): Promise<string[]> {
  const dir = path.resolve(fixturesRoot, "domain-packs");
  let entries: string[];
  try {
    entries = await fs.readdir(dir);
  } catch {
    return [];
  }
  return entries.filter((f) => f.endsWith(".json") && !f.startsWith("_")).map((f) => f.replace(/\.json$/, ""));
}

/**
 * Clear all caches. Useful in tests.
 */
export function clearDomainPackCaches(): void {
  packCache.clear();
  manifestCache = null;
  manifestRoot = null;
}
