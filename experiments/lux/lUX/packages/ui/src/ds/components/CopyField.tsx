"use client";
import * as React from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/config/utils";
import { addBreadcrumb } from "@/lib/reportError";

/**
 * CopyField — inline label + value + copy button with optimistic feedback.
 *
 * Optimistic UI pattern:
 *   1. User clicks Copy → button immediately shows checkmark + "Copied"
 *   2. Clipboard write happens async in background
 *   3. If write fails, reverts to error state with "Failed"
 *   4. Focus returns to the button after action completes
 *   5. A breadcrumb is logged for observability
 */
export const CopyField = React.memo(function CopyField({ label, value }: { label: string; value: string }) {
  const [state, setState] = React.useState<"idle" | "copied" | "error">("idle");
  const buttonRef = React.useRef<HTMLButtonElement>(null);
  const timerRef = React.useRef<ReturnType<typeof setTimeout>>();

  const onCopy = React.useCallback(async () => {
    // Optimistic: show "Copied" immediately before clipboard write completes
    setState("copied");
    if (timerRef.current) clearTimeout(timerRef.current);

    try {
      await navigator.clipboard.writeText(value);
      addBreadcrumb("action", `Copied ${label}`);
      timerRef.current = setTimeout(() => setState("idle"), 1200);
    } catch {
      // Revert optimistic state on failure
      setState("error");
      timerRef.current = setTimeout(() => setState("idle"), 1800);
    }
    // Re-focus the button after the async clipboard operation so keyboard
    // users don't lose their place in the tab order.
    buttonRef.current?.focus();
  }, [value, label]);

  // Cleanup timer on unmount
  React.useEffect(() => () => {
    if (timerRef.current) clearTimeout(timerRef.current);
  }, []);

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
          "inline-flex items-center gap-1.5 transition-all duration-fast ease-lux-out",
          state === "copied"
            ? "text-[var(--color-verdict-pass)]"
            : state === "error"
              ? "text-[var(--color-status-fail)]"
              : "",
        )}
      >
        {state === "copied" && (
          <svg
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-3.5 w-3.5 animate-lux-scale-in"
            aria-hidden="true"
          >
            <path d="M3.5 8.5 6.5 11.5 12.5 4.5" />
          </svg>
        )}
        {state === "error" && (
          <svg
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            className="h-3.5 w-3.5"
            aria-hidden="true"
          >
            <path d="M4 4 12 12M12 4 4 12" />
          </svg>
        )}
        <span aria-live="polite">
          {state === "copied" ? "Copied" : state === "error" ? "Failed" : "Copy"}
        </span>
      </Button>
    </div>
  );
});
