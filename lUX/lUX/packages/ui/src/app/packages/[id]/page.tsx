import "server-only";

import type { Metadata } from "next";
import { notFound } from "next/navigation";
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

export const revalidate = env.revalidate;

export async function generateMetadata({
  params,
  searchParams,
}: {
  params: { id: string };
  searchParams: Record<string, string | string[] | undefined>;
}): Promise<Metadata> {
  const id = decodeURIComponent(params.id);
  const mode = parseMode(Array.isArray(searchParams.mode) ? searchParams.mode[0] : (searchParams.mode ?? undefined));
  const title = `${id} · ${mode}`;
  return {
    title: `${title} · lUX Proof Viewer`,
    openGraph: {
      title: `${title} · lUX Proof Viewer`,
      description: `Proof package "${id}" viewed in ${mode} mode`,
    },
  };
}

export default async function PackageWorkspacePage({
  params,
  searchParams,
}: {
  params: { id: string };
  searchParams: Record<string, string | string[] | undefined>;
}) {
  const id = decodeURIComponent(params.id);
  const rawBaseline = searchParams.baseline ? String(searchParams.baseline) : undefined;
  const mode = parseMode(Array.isArray(searchParams.mode) ? searchParams.mode[0] : (searchParams.mode ?? undefined));

  const timer = startTimer("package_render");
  const provider = await getProvider();

  let proof;
  try {
    proof = await provider.loadPackage(id);
  } catch {
    notFound();
  }

  const domain = await provider.loadDomainPack(proof.meta.domain_id);

  let baselineProof;
  if (rawBaseline) {
    try {
      baselineProof = await provider.loadPackage(rawBaseline);
    } catch {
      // Baseline load failure is non-fatal — render without comparison
      logger.warn("package.baseline_load_failed", { packageId: id, baseline: rawBaseline });
    }
  }

  const timing = timer.stop();
  logger.info("package.render", {
    packageId: id,
    baseline: rawBaseline,
    mode,
    domain: proof.meta.domain_id,
    durationMs: timing.durationMs,
  });

  return (
    <ProofWorkspace
      proof={proof}
      baseline={baselineProof}
      domain={domain}
      fixture={id}
      mode={mode}
      packageId={id}
    />
  );
}
