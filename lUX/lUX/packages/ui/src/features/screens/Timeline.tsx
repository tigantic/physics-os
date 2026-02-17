import { memo, useMemo } from "react";
import type { ProofPackage, DomainPack } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { DataTable, type DataTableColumn } from "@/ds/components/DataTable";
import { DataValueNumberView } from "@/features/proof/DataValueView";
import { EmptyState } from "@/ds/components/EmptyState";

type StepRow = ProofPackage["timeline"]["steps"][number];

function makeColumns(domain: DomainPack): DataTableColumn<StepRow>[] {
  return [
    {
      key: "step",
      header: "Step",
      cell: (s) => <span className="font-mono text-xs">{s.step_index}</span>,
      className: "w-16",
      numeric: true,
    },
    {
      key: "hash",
      header: "State Hash",
      cell: (s) => (
        <span className="hidden truncate font-mono text-xs text-[var(--color-text-tertiary)] md:inline" title={s.state_hash}>
          {s.state_hash}
        </span>
      ),
      className: "hidden md:table-cell max-w-[200px]",
    },
    {
      key: "conservation",
      header: "Conservation Residual",
      cell: (s) => (
        <DataValueNumberView
          dv={s.metrics.conservation_residual ?? { status: "missing", reason: "missing" }}
          metricId="conservation_residual"
          domain={domain}
        />
      ),
      numeric: true,
    },
    {
      key: "drift",
      header: "L₂ Drift",
      cell: (s) => (
        <DataValueNumberView
          dv={s.metrics.l2_drift ?? { status: "missing", reason: "missing" }}
          metricId="l2_drift"
          domain={domain}
        />
      ),
      numeric: true,
    },
  ];
}

export const TimelineScreen = memo(function TimelineScreen({ proof, domain }: { proof: ProofPackage; domain: DomainPack }) {
  const columns = useMemo(() => makeColumns(domain), [domain]);

  return (
    <Card>
      <CardHeader>
        <h2 className="text-sm text-[var(--color-text-primary)]">Timeline</h2>
        <div className="text-xs text-[var(--color-text-tertiary)]">{proof.timeline.step_count} steps</div>
      </CardHeader>
      <CardContent>
        <DataTable<StepRow>
          columns={columns}
          data={proof.timeline.steps}
          rowKey={(s) => String(s.step_index)}
          caption="Simulation timeline steps"
          compact
          maxRows={200}
          emptyState={<EmptyState title="No steps" description="Timeline contains no simulation steps." />}
        />
      </CardContent>
    </Card>
  );
});

TimelineScreen.displayName = "TimelineScreen";
