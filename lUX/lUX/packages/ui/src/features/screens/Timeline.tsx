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
          <div key={s.step_index} className="grid grid-cols-12 gap-2 rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] px-3 py-2">
            <div className="col-span-2 font-mono text-xs text-[var(--color-text-tertiary)]">Step {s.step_index}</div>
            <div className="col-span-4 truncate font-mono text-xs text-[var(--color-text-tertiary)]">{s.state_hash}</div>
            <div className="col-span-3">
              <DataValueNumberView dv={s.metrics.conservation_residual ?? { status: "missing", reason: "missing" }} metricId="conservation_residual" domain={domain} />
            </div>
            <div className="col-span-3">
              <DataValueNumberView dv={s.metrics.l2_drift ?? { status: "missing", reason: "missing" }} metricId="l2_drift" domain={domain} />
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
