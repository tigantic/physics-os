"use client";
import * as React from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/config/utils";

export const CopyField = React.memo(function CopyField({ label, value }: { label: string; value: string }) {
  const [copied, setCopied] = React.useState(false);
  const [error, setError] = React.useState(false);
  const buttonRef = React.useRef<HTMLButtonElement>(null);

  const onCopy = React.useCallback(async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setError(false);
      window.setTimeout(() => setCopied(false), 900);
    } catch {
      setError(true);
      window.setTimeout(() => setError(false), 1500);
    }
    // Re-focus the button after the async clipboard operation so keyboard
    // users don't lose their place in the tab order.
    buttonRef.current?.focus();
  }, [value]);

  return (
    <div className="flex flex-col gap-2 rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] px-3 py-2 sm:flex-row sm:items-center sm:justify-between sm:gap-3">
      <div className="min-w-0">
        <div className="text-2xs uppercase tracking-wide text-[var(--color-text-tertiary)]">{label}</div>
        <div className="truncate font-mono text-xs text-[var(--color-text-secondary)]" title={value}>{value}</div>
      </div>
      <Button
        ref={buttonRef}
        variant="ghost"
        size="sm"
        onClick={onCopy}
        aria-label={`Copy ${label}`}
        className={cn(
          "transition-colors duration-hover ease-lux-out",
          copied ? "text-[var(--color-accent)]" : error ? "text-[var(--color-status-fail)]" : "",
        )}
      >
        <span aria-live="polite">{copied ? "Copied" : error ? "Failed" : "Copy"}</span>
      </Button>
    </div>
  );
});
