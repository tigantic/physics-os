import * as React from "react";
import { cn } from "@/config/utils";

export function Card({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        "rounded-[var(--radius-outer)] border bg-[var(--color-bg-raised)] shadow-[var(--shadow-raised)] transition-shadow duration-base ease-lux-out hover:shadow-[var(--shadow-floating)]",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function CardHeader({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("px-4 pb-3 pt-4 md:px-6 md:pt-5", className)}>{children}</div>;
}

export function CardContent({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("px-4 pb-4 md:px-6 md:pb-5", className)}>{children}</div>;
}
