"use client";
import * as React from "react";
import { Button } from "@/components/ui/button";

export const Disclosure = React.memo(function Disclosure({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = React.useState(false);
  const panelId = React.useId();
  const triggerRef = React.useRef<HTMLButtonElement>(null);

  const close = React.useCallback(() => {
    setOpen(false);
    // Return focus to the trigger button after closing
    triggerRef.current?.focus();
  }, []);

  const onPanelKeyDown = React.useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        close();
      }
    },
    [close],
  );

  return (
    <div className="rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] transition-shadow duration-base ease-lux-out hover:shadow-[var(--shadow-raised)]">
      <div className="flex items-center justify-between px-3 py-2">
        <div className="text-sm text-[var(--color-text-primary)]">{title}</div>
        <Button
          ref={triggerRef}
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
        <div
          id={panelId}
          role="region"
          aria-label={title}
          className="lux-disclosure-enter px-3 pb-3"
          onKeyDown={onPanelKeyDown}
        >
          {children}
        </div>
      ) : null}
    </div>
  );
});
