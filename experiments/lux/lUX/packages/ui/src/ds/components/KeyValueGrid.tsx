"use client";
import * as React from "react";
import { cn } from "@/config/utils";

export interface KVEntry {
  label: string;
  value: React.ReactNode;
  /** Render value in monospace font (hashes, IDs, etc.) */
  mono?: boolean;
  /** Show a copy-to-clipboard button for this entry (only works when value is a string) */
  copyable?: boolean;
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
      {entries.map((e, i) => (
        <div key={`${e.label}-${i}`} className="min-w-0">
          <dt className="text-xs uppercase tracking-wider text-[var(--color-text-tertiary)]">
            {e.label}
          </dt>
          <dd
            className={cn(
              "mt-0.5 flex items-center gap-1.5 text-sm text-[var(--color-text-secondary)]",
              e.mono && "font-mono text-xs",
              e.className,
            )}
          >
            <span
              className="min-w-0 truncate"
              title={typeof e.value === "string" ? e.value : undefined}
            >
              {e.value}
            </span>
            {e.copyable && typeof e.value === "string" && (
              <CopyButton value={e.value} label={e.label} />
            )}
          </dd>
        </div>
      ))}
    </dl>
  );
});

/* ── Inline copy button ────────────────────────────────────────────── */

function CopyButton({ value, label }: { value: string; label: string }) {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = React.useCallback(async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 900);
    } catch {
      /* silently fail — clipboard may be unavailable */
    }
  }, [value]);

  return (
    <button
      type="button"
      onClick={handleCopy}
      aria-label={copied ? `${label} copied` : `Copy ${label}`}
      className="shrink-0 rounded-[var(--radius-control)] p-0.5 text-[var(--color-text-tertiary)] transition-colors duration-hover ease-lux-out hover:text-[var(--color-text-secondary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent-border)]"
    >
      {copied ? (
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
          <path d="M3.5 7.5l2 2 5-5" stroke="var(--color-accent)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      ) : (
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
          <rect x="5" y="5" width="7" height="7" rx="1.5" stroke="currentColor" strokeWidth="1.2" />
          <path d="M9 5V3.5A1.5 1.5 0 0 0 7.5 2h-4A1.5 1.5 0 0 0 2 3.5v4A1.5 1.5 0 0 0 3.5 9H5" stroke="currentColor" strokeWidth="1.2" />
        </svg>
      )}
    </button>
  );
}
