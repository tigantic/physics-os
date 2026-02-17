"use client";

import * as React from "react";
import { cn } from "@/config/utils";

/* ── Types ─────────────────────────────────────────────────────────── */

export interface DataTableColumn<T> {
  /** Unique key for React & header id */
  key: string;
  /** Column header label */
  header: string;
  /** Render the cell content for a given row */
  cell: (row: T, index: number) => React.ReactNode;
  /** Tailwind className applied to both <th> and <td> for the column */
  className?: string;
  /** Right-align numeric columns (adds text-right + tabular-nums) */
  numeric?: boolean;
}

export interface DataTableProps<T> {
  columns: DataTableColumn<T>[];
  data: readonly T[];
  /** Function returning a unique React key for each row */
  rowKey: (row: T, index: number) => string;
  /** Optional caption for a11y */
  caption?: string;
  /** Stripe alternate rows. Default: true */
  striped?: boolean;
  /** Compact density. Default: false */
  compact?: boolean;
  /** Class applied to the wrapper div */
  className?: string;
  /** Rendered when data is empty */
  emptyState?: React.ReactNode;
  /** Max rows to render before showing a "Show all" button. Default: unlimited */
  maxRows?: number;
}

/* ── Component ─────────────────────────────────────────────────────── */

function DataTableInner<T>(
  {
    columns,
    data,
    rowKey,
    caption,
    striped = true,
    compact = false,
    className,
    emptyState,
    maxRows,
  }: DataTableProps<T>,
  ref: React.ForwardedRef<HTMLDivElement>,
) {
  const [expanded, setExpanded] = React.useState(false);
  const cellPad = compact ? "px-3 py-1.5" : "px-4 py-2.5";

  const isCapped = maxRows !== undefined && !expanded && data.length > maxRows;
  const visibleData = isCapped ? data.slice(0, maxRows) : data;
  const hiddenCount = data.length - (maxRows ?? data.length);

  if (data.length === 0 && emptyState) {
    return (
      <div ref={ref} className={className}>
        {emptyState}
      </div>
    );
  }

  return (
    <div
      ref={ref}
      className={cn(
        "overflow-x-auto rounded-[var(--radius-inner)] border border-[var(--color-border-base)]",
        className,
      )}
    >
      <table className="w-full border-collapse text-left text-fluid-sm">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr className="border-b border-[var(--color-border-base)] bg-[var(--color-bg-surface)]">
            {columns.map((col) => (
              <th
                key={col.key}
                scope="col"
                className={cn(
                  cellPad,
                  "text-xs font-medium uppercase tracking-wider text-[var(--color-text-tertiary)]",
                  col.numeric && "text-right tabular-nums",
                  col.className,
                )}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {visibleData.map((row, i) => (
            <tr
              key={rowKey(row, i)}
              className={cn(
                "border-b border-[var(--color-border-base)] last:border-b-0 transition-colors duration-hover ease-lux-out hover:bg-[var(--color-bg-hover)]",
                striped && i % 2 === 1 && "bg-[var(--color-bg-surface)]/50",
              )}
            >
              {columns.map((col) => (
                <td
                  key={col.key}
                  className={cn(
                    cellPad,
                    "text-[var(--color-text-secondary)]",
                    col.numeric && "text-right font-mono tabular-nums",
                    col.className,
                  )}
                >
                  {col.cell(row, i)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {isCapped && (
        <button
          type="button"
          onClick={() => setExpanded(true)}
          className="flex w-full items-center justify-center gap-1.5 border-t border-[var(--color-border-base)] bg-[var(--color-bg-surface)] px-4 py-2 text-xs font-medium text-[var(--color-accent)] transition-colors duration-hover ease-lux-out hover:bg-[var(--color-bg-hover)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[var(--color-accent-border)]"
        >
          Show all {data.length} rows ({hiddenCount} more)
        </button>
      )}
    </div>
  );
}

export const DataTable = React.forwardRef(DataTableInner) as <T>(
  props: DataTableProps<T> & { ref?: React.Ref<HTMLDivElement> },
) => React.ReactElement;
