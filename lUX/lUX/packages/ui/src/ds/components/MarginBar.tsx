import React from "react";
import type { DataValue } from "@luxury/core";
import { Chip } from "./Chip";

/**
 * MarginBar — visualises how much headroom a gate result has.
 *
 * `margin` is expected as a fraction in [0, 1] where:
 *   0   = right at the threshold (no headroom)
 *   1   = 100 % margin (maximum headroom)
 *   <0  = beyond threshold (failure)
 *
 * The bar clamps display to [0, 1] and color-codes:
 *   >50 % → gold (comfortable)   10-50 % → warn   <10 % → fail
 */
export const MarginBar = React.memo(function MarginBar({ margin }: { margin: DataValue<number> }) {
  if (margin.status === "missing") return <Chip tone="warn">Data Unavailable</Chip>;
  if (margin.status === "invalid") return <Chip tone="fail">Invalid</Chip>;

  const val = margin.value;
  const pct = Math.max(0, Math.min(1, val));
  const tone =
    val >= 0.5 ? "var(--color-accent)" : val >= 0.1 ? "var(--color-status-warn)" : "var(--color-status-fail)";

  return (
    <div className="w-full">
      <div
        role="progressbar"
        aria-valuenow={Math.round(pct * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
        className="h-2 w-full overflow-hidden rounded-full border border-[var(--color-border-base)] bg-[var(--color-bg-surface)]"
      >
        <div
          className="h-full transition-all duration-base ease-lux-out"
          style={{ width: `${pct * 100}%`, backgroundColor: tone }}
        />
      </div>
      <div className="mt-1 flex items-center justify-between">
        <span className="font-mono text-xs text-[var(--color-text-tertiary)]">{(val * 100).toFixed(1)}%</span>
        {val < 0.1 && <Chip tone="fail">Low margin</Chip>}
      </div>
    </div>
  );
});
