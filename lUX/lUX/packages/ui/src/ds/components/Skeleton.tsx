import * as React from "react";
import { cn } from "@/config/utils";

export interface SkeletonProps {
  /** Width — CSS value or Tailwind class like "w-full". Default: "w-full" */
  className?: string;
  /** Height — e.g. "h-4", "h-8". Default: "h-4" */
  heightClass?: string;
  /** Rounded corners. Default: "rounded-md" */
  roundedClass?: string;
  /** Number of skeleton rows to render. Default: 1 */
  rows?: number;
  /** Gap between rows. Default: "gap-2" */
  gapClass?: string;
}

export const Skeleton = React.memo(function Skeleton({
  className,
  heightClass = "h-4",
  roundedClass = "rounded-md",
  rows = 1,
  gapClass = "gap-2",
}: SkeletonProps) {
  if (rows === 1) {
    return (
      <div
        role="status"
        aria-label="Loading"
        className={cn(
          "lux-shimmer-bg animate-lux-shimmer",
          heightClass,
          roundedClass,
          className ?? "w-full",
        )}
      />
    );
  }

  return (
    <div role="status" aria-label="Loading" className={cn("flex flex-col", gapClass)}>
      {Array.from({ length: rows }, (_, i) => (
        <div
          key={i}
          className={cn(
            "lux-shimmer-bg animate-lux-shimmer",
            heightClass,
            roundedClass,
            className ?? "w-full",
            /* Stagger last row shorter for visual variety */
            i === rows - 1 && rows > 1 && "w-3/4",
          )}
        />
      ))}
    </div>
  );
});
