import type { ProofPackage } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { CopyField } from "@/ds/components/CopyField";
import { cn } from "@/config/utils";

export function RightRail({ proof }: { proof: ProofPackage }) {
  const v = proof.verification?.status ?? "UNVERIFIED";
  return (
    <aside
      aria-label="Integrity details"
      className="w-full px-4 py-4 lg:w-[360px] lg:shrink-0 lg:px-6 lg:py-6 xl:w-[400px]"
    >
      <Card>
        <CardHeader>
          <div className="text-sm text-[var(--color-text-primary)]">Integrity</div>
          <div className="flex items-center gap-2 text-xs text-[var(--color-text-tertiary)]">
            <span
              className={cn(
                "inline-block h-1.5 w-1.5 rounded-full",
                v === "VERIFIED"
                  ? "bg-[var(--color-verdict-pass)]"
                  : v === "BROKEN_CHAIN"
                    ? "bg-[var(--color-verdict-fail)]"
                    : "bg-[var(--color-text-tertiary)]",
              )}
              aria-hidden="true"
            />
            {v}
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <CopyField label="Merkle Root" value={proof.attestation.merkle_root} />
          <CopyField label="Container Digest" value={proof.meta.environment.container_digest} />
          <CopyField label="Commit" value={proof.meta.solver.commit_hash} />
          {proof.verification?.failures?.length ? (
            <div className="rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] px-3 py-2">
              <div className="text-xs uppercase tracking-wide text-[var(--color-text-tertiary)]">Failures</div>
              <ul className="mt-2 space-y-1 text-xs text-[var(--color-verdict-fail)]">
                {proof.verification.failures.map((f) => (
                  <li key={`${f.code}-${f.artifact_id ?? "global"}`} className="font-mono">
                    {f.code}
                    {f.artifact_id ? ` (${f.artifact_id})` : ""}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </CardContent>
      </Card>
    </aside>
  );
}
