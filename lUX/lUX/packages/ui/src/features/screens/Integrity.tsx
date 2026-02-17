import type { ProofPackage } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { Chip } from "@/ds/components/Chip";

export function IntegrityScreen({ proof }: { proof: ProofPackage }) {
  const v = proof.verification?.status ?? "UNVERIFIED";
  return (
    <Card>
      <CardHeader>
        <div className="text-sm text-[var(--color-text-primary)]">Verification</div>
        <div className="text-xs text-[var(--color-text-tertiary)]">{v}</div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] px-3 py-2">
          <div className="text-xs text-[var(--color-text-tertiary)]">Merkle Root</div>
          <div className="font-mono text-xs text-[var(--color-text-secondary)]">{proof.attestation.merkle_root}</div>
        </div>
        {proof.verification?.failures?.length ? (
          <div className="space-y-2">
            {proof.verification.failures.map((f, i) => (
              <div key={i} className="flex items-center justify-between rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] px-3 py-2">
                <div className="font-mono text-xs text-[var(--color-text-secondary)]">{f.code}</div>
                <Chip tone="fail">FAIL</Chip>
              </div>
            ))}
          </div>
        ) : (
          <Chip tone="gold">Chain Intact</Chip>
        )}
      </CardContent>
    </Card>
  );
}
