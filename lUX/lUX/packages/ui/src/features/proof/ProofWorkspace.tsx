import { Suspense } from "react";
import type { ProofPackage, DomainPack, ProofMode } from "@luxury/core";
import { IdentityStrip } from "./IdentityStrip";
import { LeftRail } from "./LeftRail";
import { RightRail } from "./RightRail";
import { CenterCanvas } from "./CenterCanvas";
import { ModeDial } from "./ModeDial";
import { ResponsiveShell } from "./ResponsiveShell";

function CenterSkeleton() {
  return (
    <div className="space-y-4">
      <div className="lux-shimmer-bg h-40 animate-lux-shimmer rounded-[var(--radius-outer)]" />
      <div className="lux-shimmer-bg h-60 animate-lux-shimmer rounded-[var(--radius-outer)]" />
    </div>
  );
}

export function ProofWorkspace({
  proof,
  baseline,
  domain,
  fixture,
  mode,
  packageId,
}: {
  proof: ProofPackage;
  baseline?: ProofPackage;
  domain: DomainPack;
  fixture: string;
  mode: ProofMode;
  packageId: string;
}) {
  return (
    <div className="min-h-screen bg-[var(--color-bg-base)]">
      <ResponsiveShell
        header={<IdentityStrip proof={proof} />}
        leftRail={<LeftRail proof={proof} fixture={fixture} mode={mode} />}
        rightRail={<RightRail proof={proof} />}
      >
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-2 px-4 pt-4 md:px-6 2xl:max-w-[1600px]">
          <div className="text-xs text-[var(--color-text-tertiary)]">
            Fixture: <span className="font-mono">{fixture}</span>
          </div>
          <ModeDial />
        </div>
        <main id="main-content" className="min-w-0 flex-1 px-4 py-4 md:px-6 md:py-6">
          <Suspense fallback={<CenterSkeleton />}>
            <CenterCanvas proof={proof} baseline={baseline} domain={domain} mode={mode} packageId={packageId} />
          </Suspense>
        </main>
      </ResponsiveShell>
    </div>
  );
}
