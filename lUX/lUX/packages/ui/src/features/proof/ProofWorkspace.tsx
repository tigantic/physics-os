import type { ProofPackage, DomainPack, ProofMode } from "@luxury/core";
import { IdentityStrip } from "./IdentityStrip";
import { LeftRail } from "./LeftRail";
import { RightRail } from "./RightRail";
import { CenterCanvas } from "./CenterCanvas";
import { ModeDial } from "./ModeDial";

export function ProofWorkspace({
  proof,
  baseline,
  domain,
  fixture,
  mode,
  bundleDir
}: {
  proof: ProofPackage;
  baseline?: ProofPackage;
  domain: DomainPack;
  fixture: string;
  mode: ProofMode;
  bundleDir: string;
}) {
  return (
    <div className="min-h-screen bg-[var(--color-bg-base)]">
      <IdentityStrip proof={proof} />
      <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-2 px-4 pt-4 md:px-6">
        <div className="text-xs text-[var(--color-text-tertiary)]">Fixture: <span className="font-mono">{fixture}</span></div>
        <ModeDial />
      </div>
      <div className="mx-auto flex max-w-[1400px] flex-col gap-0 md:flex-row">
        <LeftRail proof={proof} fixture={fixture} mode={mode} />
        <main className="min-w-0 flex-1 px-4 py-4 md:px-6 md:py-6">
          <CenterCanvas proof={proof} baseline={baseline} domain={domain} mode={mode} bundleDir={bundleDir} />
        </main>
        <RightRail proof={proof} />
      </div>
    </div>
  );
}
