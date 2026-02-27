import * as React from "react";
import { cn } from "@/config/utils";

export interface EmptyStateProps {
  /** Primary heading text */
  title: string;
  /** Optional descriptive text */
  description?: string;
  /** Optional icon (rendered above title) */
  icon?: React.ReactNode;
  /** Optional action button/link */
  action?: React.ReactNode;
  /** Class applied to the wrapper */
  className?: string;
}

export const EmptyState = React.memo(function EmptyState({
  title,
  description,
  icon,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center rounded-[var(--radius-outer)] border border-dashed border-[var(--color-border-base)] px-6 py-12 text-center",
        className,
      )}
    >
      {icon && (
        <div className="mb-4 text-[var(--color-text-tertiary)]" aria-hidden="true">
          {icon}
        </div>
      )}
      <h3 className="text-sm font-medium text-[var(--color-text-primary)]">{title}</h3>
      {description && (
        <p className="mt-1 max-w-[320px] text-xs text-[var(--color-text-tertiary)]">{description}</p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
});
