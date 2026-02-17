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
      <div className="mx-auto flex max-w-[1400px] items-center justify-between px-6 pt-4">
        <div className="text-xs text-[var(--color-text-tertiary)]">Fixture: <span className="font-mono">{fixture}</span></div>
        <ModeDial />
      </div>
      <div className="mx-auto flex max-w-[1400px] gap-0">
        <LeftRail proof={proof} fixture={fixture} mode={mode} />
        <main className="flex-1 px-6 py-6">
          <CenterCanvas proof={proof} baseline={baseline} domain={domain} mode={mode} bundleDir={bundleDir} />
        </main>
        <RightRail proof={proof} />
      </div>
    </div>
  );
}
