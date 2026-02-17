import type { ProofPackage } from "@luxury/core";
import { VerdictSeal } from "@/ds/components/VerdictSeal";
import { CopyField } from "@/ds/components/CopyField";

export function IdentityStrip({ proof }: { proof: ProofPackage }) {
  const verification = proof.verification?.status ?? "UNVERIFIED";
  return (
    <div className="sticky top-0 z-50 border-b bg-[var(--color-bg-base)]">
      <div className="mx-auto flex max-w-[1400px] items-center justify-between gap-6 px-6 py-4">
        <div className="min-w-0">
          <div className="text-xs uppercase tracking-wider text-[var(--color-text-tertiary)]">Luxury Physics Viewer</div>
          <div className="truncate text-lg font-semibold text-[var(--color-text-primary)]">{proof.meta.project_id} · {proof.meta.domain_id}</div>
        </div>
        <VerdictSeal status={proof.verdict.status} verification={verification} />
        <div className="hidden w-[420px] lg:block">
          <CopyField label="Run ID" value={proof.meta.id} />
        </div>
      </div>
    </div>
  );
}
