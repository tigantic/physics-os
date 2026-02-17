import fs from "node:fs";
import path from "node:path";
import { DomainPackSchema } from "../schema/domainPack.zod.js";
import type { DomainPack } from "../schema/domainPack.zod.js";

function deepFreeze<T>(obj: T): T {
  if (obj && typeof obj === "object") {
    Object.freeze(obj);
    for (const v of Object.values(obj as Record<string, unknown>)) {
      if (v && typeof v === "object" && !Object.isFrozen(v)) deepFreeze(v);
    }
  }
  return obj;
}

/** LRU cache for loaded domain packs — avoids repeated disk IO + Zod parse. */
const packCache = new Map<string, DomainPack>();

/**
 * Load and validate a single DomainPack from a JSON file.
 * Result is deep-frozen and cached by absolute path.
 */
export function loadDomainPackFromFile(filePath: string): DomainPack {
  const abs = path.resolve(filePath);
  const cached = packCache.get(abs);
  if (cached) return cached;

  const raw: unknown = JSON.parse(fs.readFileSync(abs, "utf8"));
  const pack = deepFreeze(DomainPackSchema.parse(raw));
  packCache.set(abs, pack);
  return pack;
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
export function loadDomainPackById(fixturesRoot: string, packId: string): DomainPack {
  assertSafeId(packId, "packId");
  const p = path.resolve(fixturesRoot, "domain-packs", `${packId}.json`);
  if (!fs.existsSync(p)) throw new Error(`DomainPack not found: ${packId}`);
  return loadDomainPackFromFile(p);
}

/**
 * Load a domain pack by TPC domain_id (e.g. "II.2").
 * Uses _manifest.json to resolve domain_id → pack ID.
 */
export function loadDomainPackForDomain(fixturesRoot: string, domainId: string): DomainPack {
  assertSafeId(domainId, "domainId");
  const manifest = loadManifest(fixturesRoot);
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

function loadManifest(fixturesRoot: string): Record<string, string> {
  const root = path.resolve(fixturesRoot);
  if (manifestCache && manifestRoot === root) return manifestCache;

  const manifestPath = path.join(root, "domain-packs", "_manifest.json");
  if (!fs.existsSync(manifestPath)) return {};

  const raw: unknown = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  if (typeof raw !== "object" || raw === null) return {};

  manifestCache = raw as Record<string, string>;
  manifestRoot = root;
  return manifestCache;
}

/**
 * List all available domain pack IDs from the domain-packs directory.
 */
export function listDomainPackIds(fixturesRoot: string): string[] {
  const dir = path.resolve(fixturesRoot, "domain-packs");
  if (!fs.existsSync(dir)) return [];
  return fs
    .readdirSync(dir)
    .filter((f) => f.endsWith(".json") && !f.startsWith("_"))
    .map((f) => f.replace(/\.json$/, ""));
}

/**
 * Clear all caches. Useful in tests.
 */
export function clearDomainPackCaches(): void {
  packCache.clear();
  manifestCache = null;
  manifestRoot = null;
}