import type { ProofPackage, DomainPack } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { MarginBar } from "@/ds/components/MarginBar";
import { Chip } from "@/ds/components/Chip";

export function GatesScreen({ proof }: { proof: ProofPackage; domain: DomainPack }) {
  const results = Object.values(proof.gate_results);
  return (
    <Card>
      <CardHeader>
        <div className="text-sm text-[var(--color-text-primary)]">Gates</div>
        <div className="text-xs text-[var(--color-text-tertiary)]">{results.length} evaluated</div>
      </CardHeader>
      <CardContent className="space-y-3">
        {results.map((r) => (
          <div key={r.gate_id} className="rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] p-3">
            <div className="flex items-center justify-between">
              <div className="font-mono text-xs text-[var(--color-text-tertiary)]">
                {r.gate_id} · {r.metric_id}
              </div>
              {r.passed.status === "ok" ? (
                <Chip tone={r.passed.value ? "gold" : "fail"}>{r.passed.value ? "PASS" : "FAIL"}</Chip>
              ) : r.passed.status === "missing" ? (
                <Chip tone="warn">Data Unavailable</Chip>
              ) : (
                <Chip tone="fail">Invalid</Chip>
              )}
            </div>
            <div className="mt-3">
              <MarginBar margin={r.margin} />
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
