import "server-only";

import path from "node:path";
import type { Metadata } from "next";
import type { ProofMode } from "@luxury/core";
import { loadProofPackageFromDir, loadDomainPackForDomain } from "@luxury/core";
import { ProofWorkspace } from "@/features/proof/ProofWorkspace";
import { env } from "@/config/env";

const VALID_MODES = new Set<ProofMode>(["EXECUTIVE", "REVIEW", "AUDIT", "PUBLICATION"]);

function isProofMode(s: string): s is ProofMode {
  return VALID_MODES.has(s as ProofMode);
}

function parseMode(raw?: string): ProofMode {
  const m = (raw ?? "REVIEW").toUpperCase();
  return isProofMode(m) ? m : "REVIEW";
}

/** ISR revalidation — configurable via LUX_REVALIDATE env var. */
export const revalidate = env.revalidate;

const VALID_FIXTURES = new Set(["pass", "fail", "warn", "incomplete", "tampered"]);

export async function generateMetadata({
  searchParams,
}: {
  searchParams: Record<string, string | string[] | undefined>;
}): Promise<Metadata> {
  const fixture = String(searchParams.fixture ?? "pass");
  const mode = parseMode(searchParams.mode as string | undefined);
  const title = `${fixture} · ${mode}`;
  return {
    title,
    openGraph: {
      title: `${title} · Luxury Physics Viewer`,
      description: `Proof package "${fixture}" viewed in ${mode} mode`,
    },
  };
}

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

  const bundleDir = path.resolve(env.fixturesRoot, "proof-packages", fixture);
  const loaded = await loadProofPackageFromDir(bundleDir);
  const domain = await loadDomainPackForDomain(env.fixturesRoot, loaded.proof.meta.domain_id);

  const baselineDir = path.resolve(env.fixturesRoot, "proof-packages", baseline);
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
