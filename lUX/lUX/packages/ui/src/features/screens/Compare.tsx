import { memo, useMemo } from "react";
import type { ProofPackage, DomainPack } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { DataTable, type DataTableColumn } from "@/ds/components/DataTable";
import { DataValueNumberView } from "@/features/proof/DataValueView";
import { KeyValueGrid } from "@/ds/components/KeyValueGrid";
import { Chip } from "@/ds/components/Chip";
import { EmptyState } from "@/ds/components/EmptyState";

interface CompareRow {
  metricId: string;
  label: string;
  current: ProofPackage["timeline"]["steps"][number]["metrics"][string];
  baseline: ProofPackage["timeline"]["steps"][number]["metrics"][string];
}

function lastStepMetric(proof: ProofPackage, metricId: string) {
  const step = proof.timeline.steps[proof.timeline.steps.length - 1];
  return step.metrics[metricId] ?? { status: "missing" as const, reason: "Metric missing" };
}

function makeColumns(domain: DomainPack): DataTableColumn<CompareRow>[] {
  return [
    {
      key: "metric",
      header: "Metric",
      cell: (r) => <span className="text-xs font-semibold">{r.label}</span>,
      className: "w-[25%]",
    },
    {
      key: "current",
      header: "Current",
      cell: (r) => <DataValueNumberView dv={r.current} metricId={r.metricId} domain={domain} />,
      numeric: true,
    },
    {
      key: "baseline",
      header: "Baseline",
      cell: (r) => <DataValueNumberView dv={r.baseline} metricId={r.metricId} domain={domain} />,
      numeric: true,
    },
  ];
}

export const CompareScreen = memo(function CompareScreen({
  proof,
  baseline,
  domain,
}: {
  proof: ProofPackage;
  baseline?: ProofPackage;
  domain: DomainPack;
}) {
  const columns = useMemo(() => makeColumns(domain), [domain]);
  const mids = domain.templates.executive_summary_metric_ids;

  const rows: CompareRow[] = useMemo(
    () =>
      baseline
        ? mids.map((mid) => ({
            metricId: mid,
            label: domain.metrics[mid]?.label ?? mid,
            current: lastStepMetric(proof, mid),
            baseline: lastStepMetric(baseline, mid),
          }))
        : [],
    [baseline, mids, domain, proof],
  );

  if (!baseline) {
    return (
      <Card>
        <CardHeader>
          <div className="text-sm text-[var(--color-text-primary)]">Compare</div>
          <div className="text-xs text-[var(--color-text-tertiary)]">No baseline selected</div>
        </CardHeader>
        <CardContent>
          <EmptyState title="No baseline" description="Select a baseline proof to enable comparison." action={<Chip tone="warn">Data Unavailable</Chip>} />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="text-sm text-[var(--color-text-primary)]">Compare</div>
      </CardHeader>
      <CardContent className="space-y-4">
        <KeyValueGrid
          entries={[
            { label: "Current", value: proof.meta.id, mono: true },
            { label: "Baseline", value: baseline.meta.id, mono: true },
          ]}
          columns={2}
        />
        <DataTable<CompareRow>
          columns={columns}
          data={rows}
          rowKey={(r) => r.metricId}
          caption="Metric comparison: current vs baseline"
          emptyState={<EmptyState title="No metrics" description="Domain has no metrics configured." />}
        />
      </CardContent>
    </Card>
  );
});

CompareScreen.displayName = "CompareScreen";
