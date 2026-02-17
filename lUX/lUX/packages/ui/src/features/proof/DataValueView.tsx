import type { DataValue, DomainPack } from "@luxury/core";
import { Chip } from "@/ds/components/Chip";

function formatEngineering(n: number, precision: number): string {
  if (n === 0) return n.toFixed(precision);
  const exp = Math.floor(Math.log10(Math.abs(n)) / 3) * 3;
  const scaled = n / Math.pow(10, exp);
  return `${scaled.toFixed(precision)}e${exp}`;
}

export function DataValueNumberView({ dv, metricId, domain }: { dv: DataValue<number>; metricId: string; domain: DomainPack }) {
  if (dv.status === "missing") return <Chip tone="warn">Data Unavailable</Chip>;
  if (dv.status === "invalid") return <Chip tone="fail">Invalid</Chip>;
  const meta = domain.metrics[metricId];
  const precision = meta?.precision ?? 4;
  const fmt = meta?.format ?? "fixed";
  let s = "";
  if (fmt === "fixed") s = dv.value.toFixed(precision);
  else if (fmt === "scientific") s = dv.value.toExponential(precision);
  else s = formatEngineering(dv.value, precision);

  if (meta?.validity_range) {
    const [lo, hi] = meta.validity_range;
    if (dv.value < lo || dv.value > hi) return <Chip tone="fail">Invalid</Chip>;
  }
  return <span className="font-mono text-sm text-[var(--color-text-primary)]">{s}</span>;
}
