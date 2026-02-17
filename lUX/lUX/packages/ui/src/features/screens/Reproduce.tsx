import type { ProofPackage } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { CopyField } from "@/ds/components/CopyField";
import { Chip } from "@/ds/components/Chip";

/** Validates container digest format: sha256:<64 hex chars> (matches Zod SHA256 schema). */
const DIGEST_RE = /^sha256:[a-f0-9]{64}$/i;

export function ReproduceScreen({ proof }: { proof: ProofPackage }) {
  const { container_digest: digest, seed } = proof.meta.environment;

  if (!DIGEST_RE.test(digest) || !Number.isSafeInteger(seed)) {
    return (
      <Card>
        <CardHeader>
          <div className="text-sm text-[var(--color-text-primary)]">Reproduce</div>
        </CardHeader>
        <CardContent>
          <Chip tone="fail">Invalid reproduction metadata</Chip>
        </CardContent>
      </Card>
    );
  }

  const cmd = `docker run --rm ${digest} --seed ${seed}`;
  return (
    <Card>
      <CardHeader>
        <div className="text-sm text-[var(--color-text-primary)]">Reproduce</div>
        <div className="text-xs text-[var(--color-text-tertiary)]">Deterministic command</div>
      </CardHeader>
      <CardContent className="space-y-3">
        <CopyField label="Command" value={cmd} />
      </CardContent>
    </Card>
  );
}
