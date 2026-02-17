import * as React from "react";
import { cn } from "@/lib/utils";

export function Card({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("rounded-[var(--radius-outer)] border bg-[var(--color-bg-raised)] shadow-[var(--shadow-raised)]", className)}>{children}</div>;
}

export function CardHeader({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("px-6 pt-5 pb-3", className)}>{children}</div>;
}

export function CardContent({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("px-6 pb-5", className)}>{children}</div>;
}
