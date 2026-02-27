"use client";

import * as React from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
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
  /**
   * Row count above which the table body is virtualised. Rows outside the
   * visible scroll viewport (plus overscan) are not mounted in the DOM,
   * keeping large datasets performant. Set to `Infinity` to disable.
   * @default 50
   */
  virtualizeThreshold?: number;
  /**
   * CSS max-height applied to the scroll container when virtualisation is
   * active. Accepts any valid CSS length value.
   * @default "24rem"
   */
  virtualHeight?: string;
}

/* ── Constants ─────────────────────────────────────────────────────── */

const ROW_HEIGHT_DEFAULT = 38;
const ROW_HEIGHT_COMPACT = 30;
const VIRTUAL_OVERSCAN = 10;
const SHOW_ALL_BTN =
  "flex w-full items-center justify-center gap-1.5 border-t border-[var(--color-border-base)] bg-[var(--color-bg-surface)] px-4 py-2 text-xs font-medium text-[var(--color-accent)] transition-colors duration-hover ease-lux-out hover:bg-[var(--color-bg-hover)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[var(--color-accent-border)]";

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
    virtualizeThreshold = 50,
    virtualHeight = "24rem",
  }: DataTableProps<T>,
  ref: React.ForwardedRef<HTMLDivElement>,
) {
  const [expanded, setExpanded] = React.useState(false);
  const scrollContainerRef = React.useRef<HTMLDivElement>(null);

  const cellPad = compact ? "px-3 py-1.5" : "px-4 py-2.5";
  const isCapped = maxRows !== undefined && !expanded && data.length > maxRows;
  const visibleData = isCapped ? data.slice(0, maxRows) : data;
  const hiddenCount = data.length - (maxRows ?? data.length);

  const shouldVirtualize = visibleData.length > virtualizeThreshold;

  const virtualizer = useVirtualizer({
    count: shouldVirtualize ? visibleData.length : 0,
    getScrollElement: () => scrollContainerRef.current,
    estimateSize: () => (compact ? ROW_HEIGHT_COMPACT : ROW_HEIGHT_DEFAULT),
    overscan: VIRTUAL_OVERSCAN,
  });

  /* ── Empty state ─────────────────────────────────────────────────── */

  if (data.length === 0 && emptyState) {
    return (
      <div ref={ref} className={className}>
        {emptyState}
      </div>
    );
  }

  /* ── Shared head row ─────────────────────────────────────────────── */

  const headRow = (
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
  );

  /* ── Row renderer ────────────────────────────────────────────────── */

  const renderCells = (row: T, index: number) =>
    columns.map((col) => (
      <td
        key={col.key}
        className={cn(
          cellPad,
          "text-[var(--color-text-secondary)]",
          col.numeric && "text-right font-mono tabular-nums",
          col.className,
        )}
      >
        {col.cell(row, index)}
      </td>
    ));

  const rowCn = (index: number) =>
    cn(
      "border-b border-[var(--color-border-base)] last:border-b-0 transition-colors duration-hover ease-lux-out hover:bg-[var(--color-bg-hover)]",
      striped && index % 2 === 1 && "bg-[var(--color-bg-surface)]/50",
    );

  /* ── Virtualised path ────────────────────────────────────────────── */

  if (shouldVirtualize) {
    const virtualItems = virtualizer.getVirtualItems();
    const totalSize = virtualizer.getTotalSize();
    const paddingTop = virtualItems[0]?.start ?? 0;
    const paddingBottom = Math.max(
      0,
      totalSize - (virtualItems[virtualItems.length - 1]?.end ?? 0),
    );

    return (
      <div
        ref={ref}
        className={cn(
          "rounded-[var(--radius-inner)] border border-[var(--color-border-base)]",
          className,
        )}
      >
        <div
          ref={scrollContainerRef}
          className="overflow-auto"
          style={{ maxHeight: virtualHeight }}
          role="region"
          aria-label={caption ? `${caption} (scrollable)` : "Scrollable table region"}
          tabIndex={0}
        >
          <table className="w-full border-collapse text-left text-fluid-sm">
            {caption && <caption className="sr-only">{caption}</caption>}
            <thead className="sticky top-0 z-10">{headRow}</thead>
            <tbody>
              {paddingTop > 0 && (
                <tr aria-hidden="true">
                  <td
                    colSpan={columns.length}
                    style={{ height: paddingTop, padding: 0, border: "none" }}
                  />
                </tr>
              )}
              {virtualItems.map((vRow) => {
                const row = visibleData[vRow.index];
                return (
                  <tr
                    key={rowKey(row, vRow.index)}
                    ref={virtualizer.measureElement}
                    data-index={vRow.index}
                    className={rowCn(vRow.index)}
                  >
                    {renderCells(row, vRow.index)}
                  </tr>
                );
              })}
              {paddingBottom > 0 && (
                <tr aria-hidden="true">
                  <td
                    colSpan={columns.length}
                    style={{ height: paddingBottom, padding: 0, border: "none" }}
                  />
                </tr>
              )}
            </tbody>
          </table>
        </div>
        {isCapped && (
          <button type="button" onClick={() => setExpanded(true)} className={SHOW_ALL_BTN}>
            Show all {data.length} rows ({hiddenCount} more)
          </button>
        )}
      </div>
    );
  }

  /* ── Standard (non-virtualised) path ─────────────────────────────── */

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
        <thead>{headRow}</thead>
        <tbody>
          {visibleData.map((row, i) => (
            <tr key={rowKey(row, i)} className={rowCn(i)}>
              {renderCells(row, i)}
            </tr>
          ))}
        </tbody>
      </table>
      {isCapped && (
        <button type="button" onClick={() => setExpanded(true)} className={SHOW_ALL_BTN}>
          Show all {data.length} rows ({hiddenCount} more)
        </button>
      )}
    </div>
  );
}

export const DataTable = React.forwardRef(DataTableInner) as <T>(
  props: DataTableProps<T> & { ref?: React.Ref<HTMLDivElement> },
) => React.ReactElement;
