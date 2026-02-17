"use client";
import * as React from "react";
import { Button } from "@/components/ui/button";

export function Disclosure({ title, children }: { title: string; children: React.ReactNode }) {
  const [open, setOpen] = React.useState(false);
  const panelId = React.useId();
  return (
    <div className="rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)]">
      <div className="flex items-center justify-between px-3 py-2">
        <div className="text-sm text-[var(--color-text-primary)]">{title}</div>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-controls={panelId}
        >
          {open ? "Hide" : "Show"}
        </Button>
      </div>
      {open ? (
        <div id={panelId} role="region" aria-label={title} className="px-3 pb-3">
          {children}
        </div>
      ) : null}
    </div>
  );
}
