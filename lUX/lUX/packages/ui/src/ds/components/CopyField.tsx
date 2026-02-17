"use client";
import * as React from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/config/utils";

export const CopyField = React.memo(function CopyField({ label, value }: { label: string; value: string }) {
  const [copied, setCopied] = React.useState(false);
  const [error, setError] = React.useState(false);

  async function onCopy() {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setError(false);
      window.setTimeout(() => setCopied(false), 900);
    } catch {
      setError(true);
      window.setTimeout(() => setError(false), 1500);
    }
  }

  return (
    <div className="flex items-center justify-between gap-3 rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] px-3 py-2">
      <div className="min-w-0">
        <div className="text-[11px] uppercase tracking-wide text-[var(--color-text-tertiary)]">{label}</div>
        <div className="truncate font-mono text-xs text-[var(--color-text-secondary)]">{value}</div>
      </div>
      <Button
        variant="ghost"
        size="sm"
        onClick={onCopy}
        aria-label={`Copy ${label}`}
        className={cn(copied ? "text-[var(--color-accent-gold)]" : error ? "text-[var(--color-verdict-fail)]" : "")}
      >
        <span aria-live="polite">{copied ? "Copied" : error ? "Failed" : "Copy"}</span>
      </Button>
    </div>
  );
});
