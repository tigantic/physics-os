import type { ProofPackage } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { CopyField } from "@/ds/components/CopyField";

export function ReproduceScreen({ proof }: { proof: ProofPackage }) {
  const cmd = `docker run --rm ${proof.meta.environment.container_digest} --seed ${proof.meta.environment.seed}`;
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
