import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/config/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium transition-colors duration-fast ease-lux-out",
  {
    variants: {
      variant: {
        default: "border-[var(--color-border-base)] bg-[var(--color-bg-raised)] text-[var(--color-text-secondary)]",
        gold: "border-[var(--color-accent-goldBorder)] bg-[var(--color-accent-goldDim)] text-[var(--color-accent-gold)]",
        pass: "border-[var(--color-verdict-passBorder)] bg-[var(--color-bg-raised)] text-[var(--color-verdict-pass)]",
        fail: "border-[var(--color-verdict-failBorder)] bg-[var(--color-bg-raised)] text-[var(--color-verdict-fail)]",
        warn: "border-[var(--color-verdict-warnBorder)] bg-[var(--color-bg-raised)] text-[var(--color-verdict-warn)]",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof badgeVariants> {}

export const Badge = React.memo(function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
});
