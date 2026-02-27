"use client";

import { memo, useMemo, useState, useCallback, useTransition } from "react";
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

/**
 * BaselineSelector — optimistic selection widget for choosing a baseline
 * proof package for comparison.
 *
 * Optimistic UI pattern:
 *   1. User selects a baseline from the dropdown
 *   2. Selected ID is shown immediately (optimistic) while the parent
 *      transitions to the new data
 *   3. Uses React useTransition to keep the current view visible while
 *      loading the new baseline data
 *   4. Visual feedback shows loading state during transition
 */
const BaselineSelector = memo(function BaselineSelector({
  availableIds,
  selectedId,
  onSelect,
  isPending,
}: {
  availableIds: readonly string[];
  selectedId: string | undefined;
  onSelect: (id: string) => void;
  isPending: boolean;
}) {
  if (availableIds.length === 0) return null;

  return (
    <div className="flex items-center gap-3">
      <label
        htmlFor="baseline-select"
        className="text-2xs uppercase tracking-wide text-[var(--color-text-tertiary)]"
      >
        Baseline
      </label>
      <div className="relative">
        <select
          id="baseline-select"
          value={selectedId ?? ""}
          onChange={(e) => onSelect(e.target.value)}
          className="rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] px-2 py-1 font-mono text-xs text-[var(--color-text-secondary)] transition-colors duration-fast ease-lux-out focus-visible:ring-2 focus-visible:ring-[var(--color-accent)]"
        >
          <option value="">Select baseline…</option>
          {availableIds.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
        {isPending && (
          <div
            className="absolute right-2 top-1/2 h-3 w-3 -translate-y-1/2 animate-spin rounded-full border-2 border-[var(--color-accent)] border-t-transparent"
            role="status"
            aria-label="Loading baseline"
          />
        )}
      </div>
    </div>
  );
});

export const CompareScreen = memo(function CompareScreen({
  proof,
  baseline,
  domain,
  availableBaselineIds = [],
  onBaselineSelect,
}: {
  proof: ProofPackage;
  baseline?: ProofPackage;
  domain: DomainPack;
  /** IDs of available proof packages for baseline selection */
  availableBaselineIds?: readonly string[];
  /** Callback when user selects a baseline — parent responsible for data loading */
  onBaselineSelect?: (id: string) => void;
}) {
  const columns = useMemo(() => makeColumns(domain), [domain]);
  const mids = domain.templates.executive_summary_metric_ids;

  const [isPending, startTransition] = useTransition();
  const [optimisticId, setOptimisticId] = useState<string | undefined>(
    baseline?.meta.id,
  );

  const handleSelect = useCallback(
    (id: string) => {
      // Optimistic: update the displayed selection immediately
      setOptimisticId(id);
      // Wrap the actual data-loading callback in startTransition so the
      // current view remains visible while the new baseline loads
      startTransition(() => {
        onBaselineSelect?.(id);
      });
    },
    [onBaselineSelect],
  );

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
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-[var(--color-text-primary)]">Compare</div>
              <div className="text-xs text-[var(--color-text-tertiary)]">No baseline selected</div>
            </div>
            {availableBaselineIds.length > 0 && (
              <BaselineSelector
                availableIds={availableBaselineIds}
                selectedId={optimisticId}
                onSelect={handleSelect}
                isPending={isPending}
              />
            )}
          </div>
        </CardHeader>
        <CardContent>
          <EmptyState
            title="No baseline"
            description="Select a baseline proof to enable comparison."
            action={<Chip tone="warn">Data Unavailable</Chip>}
          />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="text-sm text-[var(--color-text-primary)]">Compare</div>
          {availableBaselineIds.length > 0 && (
            <BaselineSelector
              availableIds={availableBaselineIds}
              selectedId={optimisticId ?? baseline.meta.id}
              onSelect={handleSelect}
              isPending={isPending}
            />
          )}
        </div>
      </CardHeader>
      <CardContent className={`space-y-4 ${isPending ? "opacity-70 transition-opacity duration-base" : ""}`}>
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
