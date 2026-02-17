"use client";
import * as React from "react";
import { cn } from "@/config/utils";

/**
 * MobileDrawer — accessible slide-in overlay panel for mobile navigation.
 *
 * Features:
 * - Focus trap: tab cycles within drawer when open
 * - Escape closes
 * - Click-outside closes (overlay backdrop)
 * - Scroll lock on body while open
 * - Reduced-motion safe (CSS-driven transitions)
 * - Renders nothing on server (client-only portal-free approach)
 */
export function MobileDrawer({
  open,
  onClose,
  side = "left",
  label,
  children,
}: {
  open: boolean;
  onClose: () => void;
  side?: "left" | "right";
  label: string;
  children: React.ReactNode;
}) {
  const panelRef = React.useRef<HTMLDivElement>(null);
  const previousFocusRef = React.useRef<HTMLElement | null>(null);

  // Lock body scroll while open, compensate for scrollbar width
  React.useEffect(() => {
    if (open) {
      previousFocusRef.current = document.activeElement as HTMLElement | null;
      const scrollbarWidth = window.innerWidth - document.documentElement.clientWidth;
      document.body.style.overflow = "hidden";
      if (scrollbarWidth > 0) {
        document.body.style.paddingRight = `${scrollbarWidth}px`;
      }
      // Focus the panel after paint
      requestAnimationFrame(() => {
        panelRef.current?.focus();
      });
    } else {
      document.body.style.overflow = "";
      document.body.style.paddingRight = "";
      // Restore previous focus
      previousFocusRef.current?.focus();
    }
    return () => {
      document.body.style.overflow = "";
      document.body.style.paddingRight = "";
    };
  }, [open]);

  // Escape key handler
  React.useEffect(() => {
    if (!open) return;
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  // Focus trap: cycle tab within panel
  React.useEffect(() => {
    if (!open) return;
    function handleTab(e: KeyboardEvent) {
      if (e.key !== "Tab" || !panelRef.current) return;
      const focusable = panelRef.current.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
    document.addEventListener("keydown", handleTab);
    return () => document.removeEventListener("keydown", handleTab);
  }, [open]);

  return (
    <>
      {/* Backdrop overlay */}
      <div
        className={cn(
          "fixed inset-0 z-[100] bg-black/60 backdrop-blur-[2px] transition-opacity duration-base ease-lux-out",
          open ? "opacity-100" : "pointer-events-none opacity-0",
        )}
        aria-hidden="true"
        onClick={onClose}
      />

      {/* Drawer panel */}
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={label}
        tabIndex={-1}
        className={cn(
          "fixed top-0 z-[101] flex h-full w-[85vw] max-w-[360px] flex-col overflow-y-auto bg-[var(--color-bg-base)] shadow-[var(--shadow-floating)] transition-transform duration-base ease-lux-out",
          side === "left" && "left-0",
          side === "right" && "right-0",
          side === "left" && (open ? "translate-x-0" : "-translate-x-full"),
          side === "right" && (open ? "translate-x-0" : "translate-x-full"),
        )}
      >
        {/* Close button */}
        <div className="flex items-center justify-between border-b border-[var(--color-border-base)] px-4 py-3">
          <span className="text-sm font-medium text-[var(--color-text-primary)]">{label}</span>
          <button
            type="button"
            onClick={onClose}
            aria-label={`Close ${label}`}
            className="flex h-9 w-9 items-center justify-center rounded-md text-[var(--color-text-secondary)] transition-colors duration-hover ease-lux-out hover:bg-[var(--color-bg-hover)] hover:text-[var(--color-text-primary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent-border)]"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">{children}</div>
      </div>
    </>
  );
}
