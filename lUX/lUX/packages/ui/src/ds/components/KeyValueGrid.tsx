import * as React from "react";
import { cn } from "@/config/utils";

export interface KVEntry {
  label: string;
  value: React.ReactNode;
  /** Render value in monospace font (hashes, IDs, etc.) */
  mono?: boolean;
  /** Optional className for the value cell */
  className?: string;
}

export interface KeyValueGridProps {
  entries: KVEntry[];
  /** Number of columns at md+ breakpoint. Default: 1 */
  columns?: 1 | 2 | 3;
  /** Class applied to the grid wrapper */
  className?: string;
}

const colsClass = {
  1: "md:grid-cols-1",
  2: "md:grid-cols-2",
  3: "md:grid-cols-3",
} as const;

export const KeyValueGrid = React.memo(function KeyValueGrid({
  entries,
  columns = 1,
  className,
}: KeyValueGridProps) {
  return (
    <dl
      className={cn(
        "grid grid-cols-1 gap-x-6 gap-y-3",
        colsClass[columns],
        className,
      )}
    >
      {entries.map((e) => (
        <div key={e.label} className="min-w-0">
          <dt className="text-xs uppercase tracking-wider text-[var(--color-text-tertiary)]">
            {e.label}
          </dt>
          <dd
            className={cn(
              "mt-0.5 truncate text-sm text-[var(--color-text-secondary)]",
              e.mono && "font-mono text-xs",
              e.className,
            )}
            title={typeof e.value === "string" ? e.value : undefined}
          >
            {e.value}
          </dd>
        </div>
      ))}
    </dl>
  );
});
