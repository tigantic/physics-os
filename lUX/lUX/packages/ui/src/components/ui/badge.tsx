import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/config/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium transition-colors duration-hover ease-lux-out",
  {
    variants: {
      variant: {
        default: "border-[var(--color-border-base)] bg-[var(--color-bg-raised)] text-[var(--color-text-secondary)]",
        gold: "border-[var(--color-accent-border)] bg-[var(--color-accent-dim)] text-[var(--color-accent)]",
        pass: "border-[var(--color-status-pass-border)] bg-[var(--color-bg-raised)] text-[var(--color-status-pass)]",
        fail: "border-[var(--color-status-fail-border)] bg-[var(--color-bg-raised)] text-[var(--color-status-fail)]",
        warn: "border-[var(--color-status-warn-border)] bg-[var(--color-bg-raised)] text-[var(--color-status-warn)]",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof badgeVariants> {}

export const Badge = React.memo(function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
});
