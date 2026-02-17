import { ProofPackage } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { CopyField } from "@/ds/components/CopyField";

export function RightRail({ proof }: { proof: ProofPackage }) {
  const v = proof.verification?.status ?? "UNVERIFIED";
  return (
    <div className="w-[360px] shrink-0 px-6 py-6">
      <Card>
        <CardHeader>
          <div className="text-sm text-[var(--color-text-primary)]">Integrity</div>
          <div className="text-xs text-[var(--color-text-tertiary)]">{v}</div>
        </CardHeader>
        <CardContent className="space-y-3">
          <CopyField label="Merkle Root" value={proof.attestation.merkle_root} />
          <CopyField label="Container Digest" value={proof.meta.environment.container_digest} />
          <CopyField label="Commit" value={proof.meta.solver.commit_hash} />
          {proof.verification?.failures?.length ? (
            <div className="rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] px-3 py-2">
              <div className="text-xs uppercase tracking-wide text-[var(--color-text-tertiary)]">Failures</div>
              <ul className="mt-2 space-y-1 text-xs text-[var(--color-verdict-fail)]">
                {proof.verification.failures.map((f, i) => (
                  <li key={i} className="font-mono">{f.code}{f.artifact_id ? ` (${f.artifact_id})` : ""}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}
