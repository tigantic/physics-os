"use client";
import * as React from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { cn } from "@/config/utils";

const MODES = ["EXECUTIVE", "REVIEW", "AUDIT", "PUBLICATION"] as const;
type Mode = (typeof MODES)[number];

function isMode(s: string): s is Mode {
  return (MODES as readonly string[]).includes(s);
}

export function ModeDial() {
  const sp = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const raw = sp.get("mode") ?? "REVIEW";
  const mode: Mode = isMode(raw) ? raw : "REVIEW";
  const fixture = sp.get("fixture") ?? "pass";
  const baseline = sp.get("baseline") ?? "pass";
  const tabsRef = React.useRef<(HTMLButtonElement | null)[]>([]);

  function setMode(m: string) {
    const next = new URLSearchParams(sp.toString());
    next.set("mode", m);
    next.set("fixture", fixture);
    next.set("baseline", baseline);
    router.push(`${pathname}?${next.toString()}`);
  }

  /** Prefetch adjacent mode on hover/focus so transition is near-instant. */
  function prefetchMode(m: string) {
    if (m === mode) return;
    const next = new URLSearchParams(sp.toString());
    next.set("mode", m);
    next.set("fixture", fixture);
    next.set("baseline", baseline);
    router.prefetch(`${pathname}?${next.toString()}`);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
    const currentIdx = MODES.indexOf(mode);
    if (currentIdx === -1) return;

    let nextIdx: number | null = null;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") {
      nextIdx = (currentIdx + 1) % MODES.length;
    } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
      nextIdx = (currentIdx - 1 + MODES.length) % MODES.length;
    } else if (e.key === "Home") {
      nextIdx = 0;
    } else if (e.key === "End") {
      nextIdx = MODES.length - 1;
    }

    if (nextIdx !== null) {
      e.preventDefault();
      tabsRef.current[nextIdx]?.focus();
      setMode(MODES[nextIdx]);
    }
  }

  return (
    <div
      role="tablist"
      aria-label="Proof viewing mode"
      className="flex items-center gap-2 overflow-x-auto"
      onKeyDown={handleKeyDown}
    >
      {MODES.map((m, i) => (
        <Button
          key={m}
          ref={(el) => {
            tabsRef.current[i] = el;
          }}
          role="tab"
          id={`mode-tab-${m}`}
          aria-selected={m === mode}
          aria-controls="mode-tabpanel"
          tabIndex={m === mode ? 0 : -1}
          variant={m === mode ? "gold" : "default"}
          size="sm"
          onClick={() => setMode(m)}
          onMouseEnter={() => prefetchMode(m)}
          onFocus={() => prefetchMode(m)}
          className={cn(
            "h-10 sm:h-8",
            m === mode && "shadow-[0_0_12px_rgba(201,169,110,0.15)] ring-1 ring-[var(--color-accent-goldBorder)]",
          )}
        >
          {m}
        </Button>
      ))}
    </div>
  );
}
