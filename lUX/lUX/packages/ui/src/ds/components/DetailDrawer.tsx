"use client";
import * as React from "react";
import { cn } from "@/config/utils";

export interface DetailDrawerProps {
  /** Whether the drawer is open */
  open: boolean;
  /** Callback when the drawer should close */
  onClose: () => void;
  /** Drawer title */
  title: string;
  /** Optional subtitle */
  subtitle?: string;
  /** Drawer content */
  children: React.ReactNode;
  /** Width class. Default: "w-[420px]" */
  widthClass?: string;
}

/**
 * DetailDrawer — slides in from the right to show contextual detail.
 * Traps focus, closes on Escape. Overlays on mobile, pushes on desktop.
 */
export function DetailDrawer({
  open,
  onClose,
  title,
  subtitle,
  children,
  widthClass = "w-[420px]",
}: DetailDrawerProps) {
  const drawerRef = React.useRef<HTMLDivElement>(null);
  const closeRef = React.useRef<HTMLButtonElement>(null);

  /* Focus trap: on open, focus the close button */
  React.useEffect(() => {
    if (open) {
      closeRef.current?.focus();
    }
  }, [open]);

  /* Close on Escape */
  React.useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <>
      {/* Backdrop (mobile) */}
      <div
        className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm lg:hidden"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Drawer panel */}
      <aside
        ref={drawerRef}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className={cn(
          "fixed right-0 top-0 z-50 flex h-full flex-col border-l border-[var(--color-border-base)] bg-[var(--color-bg-raised)] shadow-[var(--shadow-floating)] animate-lux-drawer-in",
          widthClass,
          "max-w-[90vw]",
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[var(--color-border-base)] px-4 py-3">
          <div className="min-w-0">
            <h2 className="truncate text-sm font-semibold text-[var(--color-text-primary)]">{title}</h2>
            {subtitle && (
              <p className="truncate text-xs text-[var(--color-text-tertiary)]">{subtitle}</p>
            )}
          </div>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close drawer"
            className="flex h-8 w-8 items-center justify-center rounded-[var(--radius-control)] text-[var(--color-text-secondary)] transition-colors duration-hover ease-lux-out hover:bg-[var(--color-bg-hover)] hover:text-[var(--color-text-primary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent-border)]"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
              <path d="M1 1l12 12M13 1L1 13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-4 py-4">
          {children}
        </div>
      </aside>
    </>
  );
}
