import "server-only";

import type { Metadata } from "next";
import type { ProofMode } from "@luxury/core";
import { ProofWorkspace } from "@/features/proof/ProofWorkspace";
import { getProvider } from "@/config/provider";
import { env } from "@/config/env";
import { logger } from "@/lib/logger";
import { startTimer } from "@/lib/timing";

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
  const rawFixture = String(searchParams.fixture ?? "pass");
  // Sanitize: only allow known fixture names in metadata to prevent XSS via OG tags
  const fixture = VALID_FIXTURES.has(rawFixture) ? rawFixture : "pass";
  const mode = parseMode(Array.isArray(searchParams.mode) ? searchParams.mode[0] : (searchParams.mode ?? undefined));
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
  const mode = parseMode(Array.isArray(searchParams.mode) ? searchParams.mode[0] : (searchParams.mode ?? undefined));

  const timer = startTimer("gallery_render");
  const provider = await getProvider();
  const proof = await provider.loadPackage(fixture);
  const domain = await provider.loadDomainPack(proof.meta.domain_id);
  const baselineProof = await provider.loadPackage(baseline);
  const timing = timer.stop();

  logger.info("gallery.render", {
    fixture,
    baseline,
    mode,
    domain: proof.meta.domain_id,
    durationMs: timing.durationMs,
  });

  return (
    <ProofWorkspace
      proof={proof}
      baseline={baselineProof}
      domain={domain}
      fixture={fixture}
      mode={mode}
      packageId={fixture}
    />
  );
}
