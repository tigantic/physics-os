import "server-only";

import path from "node:path";
import type { ProofMode } from "@luxury/core";
import { loadProofPackageFromDir, loadDomainPackForDomain } from "@luxury/core";
import { ProofWorkspace } from "@/features/proof/ProofWorkspace";

const VALID_MODES = new Set<ProofMode>(["EXECUTIVE", "REVIEW", "AUDIT", "PUBLICATION"]);

function parseMode(raw?: string): ProofMode {
  const m = (raw ?? "REVIEW").toUpperCase() as ProofMode;
  return VALID_MODES.has(m) ? m : "REVIEW";
}

/**
 * Fixtures path — resolved relative to the UI package root.
 * Next.js guarantees process.cwd() = the directory containing next.config.*.
 * In this monorepo: packages/ui → ../core/tests/fixtures.
 */
const FIXTURES_ROOT = path.resolve(process.cwd(), "..", "core", "tests", "fixtures");

const VALID_FIXTURES = new Set(["pass", "fail", "warn", "incomplete", "tampered"]);

export default async function GalleryPage({
  searchParams,
}: {
  searchParams: Record<string, string | string[] | undefined>;
}) {
  const rawFixture = String(searchParams.fixture ?? "pass");
  const rawBaseline = String(searchParams.baseline ?? "pass");
  const fixture = VALID_FIXTURES.has(rawFixture) ? rawFixture : "pass";
  const baseline = VALID_FIXTURES.has(rawBaseline) ? rawBaseline : "pass";
  const mode = parseMode(searchParams.mode as string | undefined);

  const bundleDir = path.resolve(FIXTURES_ROOT, "proof-packages", fixture);
  const loaded = await loadProofPackageFromDir(bundleDir);
  const domain = await loadDomainPackForDomain(FIXTURES_ROOT, loaded.proof.meta.domain_id);

  const baselineDir = path.resolve(FIXTURES_ROOT, "proof-packages", baseline);
  const baselineLoaded = await loadProofPackageFromDir(baselineDir);

  return (
    <ProofWorkspace
      proof={loaded.proof}
      baseline={baselineLoaded.proof}
      domain={domain}
      fixture={fixture}
      mode={mode}
      bundleDir={bundleDir}
    />
  );
}
