"use client";

import { memo, useMemo } from "react";
import type { ProofPackage } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { DataTable, type DataTableColumn } from "@/ds/components/DataTable";
import { MarginBar } from "@/ds/components/MarginBar";
import { Chip } from "@/ds/components/Chip";
import { EmptyState } from "@/ds/components/EmptyState";

type GateRow = ProofPackage["gate_results"][string];

const COLUMNS: DataTableColumn<GateRow>[] = [
  {
    key: "gate",
    header: "Gate",
    cell: (r) => <span className="font-mono text-xs">{r.gate_id}</span>,
    className: "w-[30%]",
  },
  {
    key: "metric",
    header: "Metric",
    cell: (r) => <span className="font-mono text-xs text-[var(--color-text-tertiary)]">{r.metric_id}</span>,
    className: "w-[25%]",
  },
  {
    key: "status",
    header: "Status",
    cell: (r) => {
      if (r.passed.status === "ok") {
        return <Chip tone={r.passed.value ? "gold" : "fail"}>{r.passed.value ? "PASS" : "FAIL"}</Chip>;
      }
      if (r.passed.status === "missing") {
        return <Chip tone="warn">Data Unavailable</Chip>;
      }
      return <Chip tone="fail">Invalid</Chip>;
    },
    className: "w-[15%]",
  },
  {
    key: "margin",
    header: "Margin",
    cell: (r) => <MarginBar margin={r.margin} />,
    className: "w-[30%]",
  },
];

export const GatesScreen = memo(function GatesScreen({ proof }: { proof: ProofPackage }) {
  const results = useMemo(() => Object.values(proof.gate_results), [proof.gate_results]);
  return (
    <Card>
      <CardHeader>
        <h2 className="text-sm text-[var(--color-text-primary)]">Gates</h2>
        <div className="text-xs text-[var(--color-text-tertiary)]">{results.length} evaluated</div>
      </CardHeader>
      <CardContent>
        <DataTable<GateRow>
          columns={COLUMNS}
          data={results}
          rowKey={(r) => r.gate_id}
          caption="Gate evaluation results"
          emptyState={<EmptyState title="No gates evaluated" description="This proof has no gate results." />}
        />
      </CardContent>
    </Card>
  );
});

GatesScreen.displayName = "GatesScreen";
