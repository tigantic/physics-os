import "server-only";

import type { ProofPackage } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { Chip } from "@/ds/components/Chip";
import { getProvider } from "@/config/provider";
import { parseCsv, sparkline } from "./sparkline";

export async function TimeSeriesViewer({
  proof,
  packageId,
  artifactId,
}: {
  proof: ProofPackage;
  packageId: string;
  artifactId: string;
}) {
  const art = proof.artifacts[artifactId];
  if (!art) return <Chip tone="warn">Data Unavailable</Chip>;

  const provider = await getProvider();
  const result = await provider.readArtifact(packageId, art.uri);

  if (!result.ok) {
    return <Chip tone="fail">Read Error</Chip>;
  }

  const pts = parseCsv(result.bytes);
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
          <svg
            role="img"
            aria-label={`Time-series sparkline for ${artifactId}`}
            width="100%"
            viewBox="0 0 560 120"
            preserveAspectRatio="none"
          >
            <path d={d} fill="none" stroke="var(--color-accent)" strokeWidth="2" />
          </svg>
        </div>
        <div className="mt-2 font-mono text-xs text-[var(--color-text-tertiary)]">{art.hash}</div>
      </CardContent>
    </Card>
  );
}
