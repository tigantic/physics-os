import { memo } from "react";
import type { ProofPackage } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { KeyValueGrid } from "@/ds/components/KeyValueGrid";
import { Chip } from "@/ds/components/Chip";

export const IntegrityScreen = memo(function IntegrityScreen({ proof }: { proof: ProofPackage }) {
  const v = proof.verification?.status ?? "UNVERIFIED";
  return (
    <Card>
      <CardHeader>
        <h2 className="text-sm text-[var(--color-text-primary)]">Verification</h2>
        <div className="text-xs text-[var(--color-text-tertiary)]">{v}</div>
      </CardHeader>
      <CardContent className="space-y-4">
        <KeyValueGrid
          entries={[
            { label: "Merkle Root", value: proof.attestation.merkle_root, mono: true },
            ...(proof.verification?.verifier_version
              ? [{ label: "Verifier", value: proof.verification.verifier_version, mono: true }]
              : []),
          ]}
        />
        {proof.verification?.failures?.length ? (
          <div className="space-y-2">
            {proof.verification.failures.map((f) => (
              <div
                key={`${f.code}-${f.artifact_id ?? "global"}`}
                className="flex items-center justify-between rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] px-3 py-2"
              >
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
});

IntegrityScreen.displayName = "IntegrityScreen";
