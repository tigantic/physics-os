import type { ProofPackage, DomainPack } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { DataValueNumberView } from "@/features/proof/DataValueView";

export function TimelineScreen({ proof, domain }: { proof: ProofPackage; domain: DomainPack }) {
  return (
    <Card>
      <CardHeader>
        <div className="text-sm text-[var(--color-text-primary)]">Timeline</div>
        <div className="text-xs text-[var(--color-text-tertiary)]">{proof.timeline.step_count} steps</div>
      </CardHeader>
      <CardContent className="space-y-2">
        {proof.timeline.steps.map((s) => (
          <div key={s.step_index} className="grid grid-cols-1 gap-1 rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] px-3 py-2 md:grid-cols-12 md:gap-2">
            <div className="font-mono text-xs text-[var(--color-text-tertiary)] md:col-span-2">Step {s.step_index}</div>
            <div className="hidden truncate font-mono text-xs text-[var(--color-text-tertiary)] md:col-span-4 md:block">{s.state_hash}</div>
            <div className="md:col-span-3">
              <DataValueNumberView dv={s.metrics.conservation_residual ?? { status: "missing", reason: "missing" }} metricId="conservation_residual" domain={domain} />
            </div>
            <div className="md:col-span-3">
              <DataValueNumberView dv={s.metrics.l2_drift ?? { status: "missing", reason: "missing" }} metricId="l2_drift" domain={domain} />
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
