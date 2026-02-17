"use client";
import * as React from "react";
import { useDrawerToggle } from "@/features/proof/ResponsiveShell";

/**
 * HamburgerButton — renders a mobile menu trigger that opens the LeftRail drawer.
 * Only visible below md breakpoint. Automatically connects to ResponsiveShell context.
 */
export function HamburgerButton() {
  const toggle = useDrawerToggle();
  if (!toggle) return null;

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Open navigation menu"
      className="flex h-10 w-10 items-center justify-center rounded-md text-[var(--color-text-secondary)] transition-colors duration-hover ease-lux-out hover:bg-[var(--color-bg-hover)] hover:text-[var(--color-text-primary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent-border)] md:hidden"
    >
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
        <path d="M3 5h14M3 10h14M3 15h14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    </button>
  );
}
