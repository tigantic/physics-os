import type { ProofPackage, DomainPack } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { DataValueNumberView } from "@/features/proof/DataValueView";
import { MathBlock } from "@/features/math/MathBlock";

export function SummaryScreen({ proof, domain, mode }: { proof: ProofPackage; domain: DomainPack; mode: string }) {
  const metrics = domain.templates.executive_summary_metric_ids;
  const step = proof.timeline.steps[proof.timeline.steps.length - 1];

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <h2 className="text-sm text-[var(--color-text-primary)]">Overview</h2>
          <div className="text-xs text-[var(--color-text-tertiary)]">
            {proof.verdict.reason || "No additional notes"}
          </div>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          {metrics.map((mid) => (
            <div key={mid} className="rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] px-4 py-3">
              <div className="text-xs text-[var(--color-text-tertiary)]">{domain.metrics[mid]?.label ?? mid}</div>
              <div className="mt-2">
                <DataValueNumberView
                  dv={step.metrics[mid] ?? { status: "missing", reason: "Metric missing" }}
                  metricId={mid}
                  domain={domain}
                />
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {mode === "PUBLICATION" ? (
        <Card>
          <CardHeader>
            <h2 className="text-sm text-[var(--color-text-primary)]">Paper View</h2>
            <div className="text-xs text-[var(--color-text-tertiary)]">Deterministic SVG math</div>
          </CardHeader>
          <CardContent>
            <MathBlock latex={"\\int f(\\mathbf{x},\\mathbf{v})\\,d\\mathbf{x}\\,d\\mathbf{v} = M"} />
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
