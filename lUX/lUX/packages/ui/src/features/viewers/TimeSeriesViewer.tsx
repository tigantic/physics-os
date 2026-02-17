import "server-only";

import fs from "node:fs/promises";
import path from "node:path";
import type { ProofPackage } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { Chip } from "@/ds/components/Chip";

function parseCsv(bytes: Uint8Array): Array<{ x: number; y: number }> {
  const txt = new TextDecoder().decode(bytes).trim();
  const lines = txt.split("\n").slice(1);
  const pts: Array<{ x: number; y: number }> = [];
  for (const line of lines) {
    const [a, b] = line.split(",");
    const x = Number(a);
    const y = Number(b);
    if (Number.isFinite(x) && Number.isFinite(y)) pts.push({ x, y });
  }
  return pts;
}

function sparkline(points: Array<{ x: number; y: number }>, w = 560, h = 120) {
  if (points.length < 2) return "";
  const ys = points.map((p) => p.y);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const dx = points.length - 1;
  const scaleY = (v: number) => {
    if (maxY === minY) return h / 2;
    return h - ((v - minY) / (maxY - minY)) * h;
  };
  const pathD = points
    .map((p, i) => {
      const x = (i / dx) * w;
      const y = scaleY(p.y);
      return `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
  return pathD;
}

export async function TimeSeriesViewer({
  proof,
  bundleDir,
  artifactId,
}: {
  proof: ProofPackage;
  bundleDir: string;
  artifactId: string;
}) {
  const art = proof.artifacts[artifactId];
  if (!art) return <Chip tone="warn">Data Unavailable</Chip>;

  const fp = path.resolve(bundleDir, art.uri);

  // Prevent path traversal — artifact URI must resolve inside bundleDir
  const resolved = path.resolve(fp);
  if (!resolved.startsWith(path.resolve(bundleDir) + path.sep)) {
    return <Chip tone="fail">Invalid Path</Chip>;
  }

  let bytes: Buffer;
  try {
    bytes = await fs.readFile(resolved);
  } catch {
    return <Chip tone="fail">Read Error</Chip>;
  }

  const pts = parseCsv(bytes);
  const d = sparkline(pts);

  return (
    <Card>
      <CardHeader>
        <div className="text-sm text-[var(--color-text-primary)]">Primary Artifact</div>
        <div className="text-xs text-[var(--color-text-tertiary)]">
          {artifactId} · {art.type}
        </div>
      </CardHeader>
      <CardContent>
        <div className="rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] p-3">
          <svg width="100%" viewBox="0 0 560 120" preserveAspectRatio="none">
            <path d={d} fill="none" stroke="var(--color-accent-gold)" strokeWidth="2" />
          </svg>
        </div>
        <div className="mt-2 font-mono text-xs text-[var(--color-text-tertiary)]">{art.hash}</div>
      </CardContent>
    </Card>
  );
}
