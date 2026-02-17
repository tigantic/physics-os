import type { ProofPackage, DomainPack } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { DataValueNumberView } from "@/features/proof/DataValueView";
import { Chip } from "@/ds/components/Chip";

function lastStepMetric(proof: ProofPackage, metricId: string) {
  const step = proof.timeline.steps[proof.timeline.steps.length - 1];
  return step.metrics[metricId] ?? { status: "missing" as const, reason: "Metric missing" };
}

export function CompareScreen({
  proof,
  baseline,
  domain,
}: {
  proof: ProofPackage;
  baseline?: ProofPackage;
  domain: DomainPack;
}) {
  if (!baseline) {
    return (
      <Card>
        <CardHeader>
          <div className="text-sm text-[var(--color-text-primary)]">Compare</div>
          <div className="text-xs text-[var(--color-text-tertiary)]">No baseline selected</div>
        </CardHeader>
        <CardContent>
          <Chip tone="warn">Data Unavailable</Chip>
        </CardContent>
      </Card>
    );
  }

  const mids = domain.templates.executive_summary_metric_ids;

  return (
    <Card>
      <CardHeader>
        <div className="text-sm text-[var(--color-text-primary)]">Compare</div>
        <div className="text-xs text-[var(--color-text-tertiary)]">
          Baseline: <span className="font-mono">{baseline.meta.id}</span>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        {mids.map((mid) => (
          <div
            key={mid}
            className="grid grid-cols-1 items-center gap-1 rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] px-3 py-2 md:grid-cols-12 md:gap-2"
          >
            <div className="text-xs font-semibold text-[var(--color-text-tertiary)] md:col-span-3 md:font-normal">
              {domain.metrics[mid]?.label ?? mid}
            </div>
            <div className="md:col-span-4">
              <DataValueNumberView dv={lastStepMetric(proof, mid)} metricId={mid} domain={domain} />
            </div>
            <div className="hidden text-xs text-[var(--color-text-tertiary)] md:col-span-1 md:block">vs</div>
            <div className="md:col-span-4">
              <DataValueNumberView dv={lastStepMetric(baseline, mid)} metricId={mid} domain={domain} />
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
