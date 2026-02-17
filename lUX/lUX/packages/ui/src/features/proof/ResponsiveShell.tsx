"use client";
import * as React from "react";
import { MobileDrawer } from "@/ds/components/MobileDrawer";

/**
 * ResponsiveShell — orchestrates mobile drawer (LeftRail) and
 * collapsible RightRail for sub-lg viewports.
 *
 * Layout strategy:
 *   xs-sm : single column, LeftRail in drawer, RightRail collapsed
 *   md    : two-column (LeftRail + Center), RightRail collapsed
 *   lg+   : three-column, everything visible
 */
export function ResponsiveShell({
  header,
  leftRail,
  rightRail,
  children,
}: {
  header: React.ReactNode;
  leftRail: React.ReactNode;
  rightRail: React.ReactNode;
  children: React.ReactNode;
}) {
  const [drawerOpen, setDrawerOpen] = React.useState(false);
  const [rightExpanded, setRightExpanded] = React.useState(false);

  const openDrawer = React.useCallback(() => setDrawerOpen(true), []);
  const closeDrawer = React.useCallback(() => setDrawerOpen(false), []);
  const toggleRight = React.useCallback(() => setRightExpanded((v) => !v), []);

  return (
    <>
      {/* Header (IdentityStrip) — pass drawer toggle via context */}
      <DrawerToggleContext.Provider value={openDrawer}>{header}</DrawerToggleContext.Provider>

      {/* Mobile drawer for LeftRail — visible only below md */}
      <MobileDrawer open={drawerOpen} onClose={closeDrawer} side="left" label="Navigation">
        <div onClick={closeDrawer}>{leftRail}</div>
      </MobileDrawer>

      {/* Main 3-column layout */}
      <div className="mx-auto flex max-w-[1400px] flex-col lg:flex-row 2xl:max-w-[1600px]">
        {/* LeftRail — hidden below md, inline from md+ */}
        <div className="hidden md:block">{leftRail}</div>

        {/* CenterCanvas */}
        <div className="min-w-0 flex-1">{children}</div>

        {/* RightRail — always visible at lg+, collapsible below lg */}
        <div className="hidden lg:block">{rightRail}</div>
        <div className="lg:hidden">
          <div className="border-t border-[var(--color-border-base)]">
            <button
              type="button"
              onClick={toggleRight}
              aria-expanded={rightExpanded}
              aria-controls="right-rail-panel"
              aria-label={rightExpanded ? "Collapse integrity details" : "Expand integrity details"}
              className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium text-[var(--color-text-primary)] transition-colors duration-hover ease-lux-out hover:bg-[var(--color-bg-hover)]"
            >
              <span>Integrity Details</span>
              <svg
                width="16"
                height="16"
                viewBox="0 0 16 16"
                fill="none"
                aria-hidden="true"
                className={`transition-transform duration-hover ease-lux-out ${rightExpanded ? "rotate-180" : ""}`}
              >
                <path
                  d="M4 6l4 4 4-4"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
            {rightExpanded && <div id="right-rail-panel" className="lux-disclosure-enter">{rightRail}</div>}
          </div>
        </div>
      </div>
    </>
  );
}

/** Context to allow IdentityStrip to trigger the mobile drawer open */
export const DrawerToggleContext = React.createContext<(() => void) | null>(null);

/** Hook consumed by IdentityStrip to get the drawer toggle callback */
export function useDrawerToggle(): (() => void) | null {
  return React.useContext(DrawerToggleContext);
}
