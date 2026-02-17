"use client";

import Link from "next/link";
import { useMemo, useState, useCallback } from "react";
import type { PackageSummary } from "@luxury/core";
import { DataTable, type DataTableColumn } from "@/ds/components/DataTable";
import { Chip } from "@/ds/components/Chip";
import { EmptyState } from "@/ds/components/EmptyState";

/* ── Verdict → Chip tone mapping ──────────────────────────────────── */

function verdictTone(status: string): "gold" | "fail" | "warn" {
  switch (status.toUpperCase()) {
    case "PASS":
      return "gold";
    case "FAIL":
      return "fail";
    default:
      return "warn";
  }
}

/* ── Columns ──────────────────────────────────────────────────────── */

const COLUMNS: DataTableColumn<PackageSummary>[] = [
  {
    key: "id",
    header: "Package",
    cell: (row) => (
      <Link
        href={`/packages/${encodeURIComponent(row.id)}`}
        className="font-mono text-xs text-[var(--color-accent)] underline-offset-2 hover:underline"
      >
        {row.id}
      </Link>
    ),
    className: "w-[22%]",
  },
  {
    key: "domain",
    header: "Domain",
    cell: (row) => (
      <span className="truncate font-mono text-xs text-[var(--color-text-tertiary)]" title={row.domain_id}>
        {row.domain_id}
      </span>
    ),
    className: "w-[22%] max-w-[200px]",
  },
  {
    key: "verdict",
    header: "Verdict",
    cell: (row) => <Chip tone={verdictTone(row.verdict_status)}>{row.verdict_status}</Chip>,
    className: "w-[12%]",
  },
  {
    key: "quality",
    header: "Quality",
    cell: (row) => <span>{(row.quality_score * 100).toFixed(0)}%</span>,
    numeric: true,
    className: "w-[10%]",
  },
  {
    key: "solver",
    header: "Solver",
    cell: (row) => <span className="text-xs text-[var(--color-text-secondary)]">{row.solver_name}</span>,
    className: "w-[16%]",
  },
  {
    key: "timestamp",
    header: "Created",
    cell: (row) => {
      if (!row.timestamp) return <span className="text-[var(--color-text-tertiary)]">—</span>;
      try {
        const d = new Date(row.timestamp);
        return (
          <time dateTime={row.timestamp} className="text-xs text-[var(--color-text-tertiary)]">
            {d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" })}
          </time>
        );
      } catch {
        return <span className="font-mono text-xs text-[var(--color-text-tertiary)]">{row.timestamp}</span>;
      }
    },
    className: "w-[18%]",
  },
];

/* ── Stable row key extractor ─────────────────────────────────────── */

const packageRowKey = (r: PackageSummary) => r.id;

/* ── Component ────────────────────────────────────────────────────── */

export function PackageList({ packages }: { packages: readonly PackageSummary[] }) {
  const [query, setQuery] = useState("");
  const onQueryChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => setQuery(e.target.value), []);

  const filtered = useMemo(() => {
    if (!query.trim()) return packages;
    const q = query.toLowerCase();
    return packages.filter(
      (p) =>
        p.id.toLowerCase().includes(q) ||
        p.domain_id.toLowerCase().includes(q) ||
        p.solver_name.toLowerCase().includes(q) ||
        p.verdict_status.toLowerCase().includes(q),
    );
  }, [packages, query]);

  return (
    <div className="space-y-4">
      {/* Search bar */}
      <div className="relative">
        <input
          type="search"
          placeholder="Search packages…"
          value={query}
          onChange={onQueryChange}
          className="w-full rounded-[var(--radius-control)] border border-[var(--color-border-base)] bg-[var(--color-bg-surface)] px-4 py-2.5 text-sm text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)] focus-visible:border-[var(--color-accent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent-border)]"
          aria-label="Search proof packages"
        />
      </div>

      {/* Table */}
      <DataTable<PackageSummary>
        columns={COLUMNS}
        data={filtered}
        rowKey={packageRowKey}
        caption="Available proof packages"
        emptyState={
          query.trim() ? (
            <EmptyState
              title="No matches"
              description={`No packages match "${query}". Try a different search term.`}
            />
          ) : (
            <EmptyState
              title="No packages found"
              description="No proof packages are available. Check your data directory configuration."
            />
          )
        }
      />
    </div>
  );
}
