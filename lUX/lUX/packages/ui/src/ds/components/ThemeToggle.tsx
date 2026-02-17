"use client";

import * as React from "react";
import { cn } from "@/config/utils";

/**
 * ThemeToggle — switches between dark and light themes via data-theme attribute.
 * Persists choice to localStorage and respects prefers-color-scheme on first visit.
 */
export function ThemeToggle({ className }: { className?: string }) {
  const [theme, setTheme] = React.useState<"dark" | "light">("dark");
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => {
    const stored = localStorage.getItem("lux-theme");
    if (stored === "light" || stored === "dark") {
      setTheme(stored);
    } else if (window.matchMedia("(prefers-color-scheme: light)").matches) {
      setTheme("light");
    }
    setMounted(true);
  }, []);

  React.useEffect(() => {
    if (!mounted) return;
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("lux-theme", theme);
  }, [theme, mounted]);

  function toggle() {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  }

  /* Render invisible placeholder before hydration to avoid layout shift */
  if (!mounted) {
    return <div className={cn("h-9 w-9", className)} aria-hidden="true" />;
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
      className={cn(
        "flex h-9 w-9 items-center justify-center rounded-[var(--radius-control)] text-[var(--color-text-secondary)] transition-colors duration-hover ease-lux-out hover:bg-[var(--color-bg-hover)] hover:text-[var(--color-text-primary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent-border)]",
        className,
      )}
    >
      {theme === "dark" ? (
        /* Sun icon */
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <circle cx="8" cy="8" r="3" stroke="currentColor" strokeWidth="1.5" />
          <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.17 3.17l1.42 1.42M11.41 11.41l1.42 1.42M3.17 12.83l1.42-1.42M11.41 4.59l1.42-1.42" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      ) : (
        /* Moon icon */
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path d="M14 9.27A6 6 0 116.73 2 4.5 4.5 0 0014 9.27z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )}
    </button>
  );
}
