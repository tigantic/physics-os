import type { ProofPackage } from "@luxury/core";
import { VerdictSeal } from "@/ds/components/VerdictSeal";
import { CopyField } from "@/ds/components/CopyField";
import { ThemeToggle } from "@/ds/components/ThemeToggle";
import { HamburgerButton } from "./HamburgerButton";

export function IdentityStrip({ proof }: { proof: ProofPackage }) {
  const verification = proof.verification?.status ?? "UNVERIFIED";
  return (
    <header
      aria-label="Proof identity"
      className="bg-[var(--color-bg-base)]/95 sticky top-0 z-50 border-b border-b-[var(--color-border-base)] backdrop-blur-sm"
    >
      <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-3 px-4 py-3 md:gap-6 md:px-6 md:py-4 2xl:max-w-[1600px]">
        <div className="flex items-center gap-3">
          <HamburgerButton />
          <div className="min-w-0">
            <div className="hidden text-xs uppercase tracking-wider text-[var(--color-text-tertiary)] sm:block">
              lUX Proof Viewer
            </div>
            <h1
              className="animate-lux-slide-up truncate text-base font-semibold text-[var(--color-text-primary)] md:text-lg"
              title={`${proof.meta.project_id} · ${proof.meta.domain_id}`}
            >
              {proof.meta.project_id} · {proof.meta.domain_id}
            </h1>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <VerdictSeal status={proof.verdict.status} verification={verification} />
          <ThemeToggle className="hidden sm:flex" />
        </div>
        <div className="hidden w-[420px] lg:block xl:w-[480px]">
          <CopyField label="Run ID" value={proof.meta.id} />
        </div>
      </div>
    </header>
  );
}
