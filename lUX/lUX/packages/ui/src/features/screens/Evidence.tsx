import type { ProofPackage } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { Disclosure } from "@/ds/components/Disclosure";

export function EvidenceScreen({ proof }: { proof: ProofPackage }) {
  return (
    <Card>
      <CardHeader>
        <div className="text-sm text-[var(--color-text-primary)]">Evidence</div>
        <div className="text-xs text-[var(--color-text-tertiary)]">{Object.keys(proof.artifacts).length} artifacts</div>
      </CardHeader>
      <CardContent className="space-y-3">
        {Object.values(proof.artifacts).map((a) => (
          <Disclosure key={a.id} title={`${a.id} · ${a.type}`}>
            <div className="space-y-2">
              <div className="font-mono text-xs text-[var(--color-text-tertiary)]">{a.hash}</div>
              <div className="text-xs text-[var(--color-text-secondary)]">URI: {a.uri}</div>
              <div className="text-xs text-[var(--color-text-secondary)]">MIME: {a.mime_type}</div>
            </div>
          </Disclosure>
        ))}
      </CardContent>
    </Card>
  );
}
