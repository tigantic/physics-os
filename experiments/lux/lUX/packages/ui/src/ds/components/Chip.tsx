import * as React from "react";
import { cn } from "@/config/utils";

export const Chip = React.memo(function Chip({
  children,
  tone = "default",
}: {
  children: React.ReactNode;
  tone?: "default" | "gold" | "fail" | "warn";
}) {
  const cls =
    tone === "gold"
      ? "border-[var(--color-accent-border)] bg-[var(--color-accent-dim)] text-[var(--color-accent)]"
      : tone === "fail"
        ? "border-[var(--color-status-fail)]/40 bg-[var(--color-bg-raised)] text-[var(--color-status-fail)]"
        : tone === "warn"
          ? "border-[var(--color-status-warn)]/40 bg-[var(--color-bg-raised)] text-[var(--color-status-warn)]"
          : "border-[var(--color-border-base)] bg-[var(--color-bg-raised)] text-[var(--color-text-secondary)]";
  return (
    <span
      role="status"
      className={cn(
        "inline-flex animate-lux-fade-in items-center rounded-md border px-2 py-0.5 text-xs font-medium",
        cls,
      )}
    >
      {children}
    </span>
  );
});
