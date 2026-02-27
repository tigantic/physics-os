import { Suspense } from "react";
import type { ProofPackage } from "@luxury/core";
import { TimeSeriesViewer } from "./TimeSeriesViewer";
import { Chip } from "@/ds/components/Chip";
import { Skeleton } from "@/ds/components/Skeleton";

function ViewerSkeleton() {
  return (
    <div className="space-y-3 rounded-[var(--radius-outer)] border border-[var(--color-border-base)] bg-[var(--color-bg-raised)] p-4">
      <Skeleton heightClass="h-4" className="w-1/3" />
      <Skeleton heightClass="h-32" />
      <Skeleton heightClass="h-3" className="w-2/3" />
    </div>
  );
}

export function PrimaryViewer({ proof, packageId }: { proof: ProofPackage; packageId: string }) {
  if (proof.verification?.status === "BROKEN_CHAIN") return <Chip tone="fail">BROKEN_CHAIN</Chip>;
  const first = Object.values(proof.artifacts).find((a) => a.type === "time_series");
  if (!first) return <Chip tone="warn">Data Unavailable</Chip>;
  return (
    <Suspense fallback={<ViewerSkeleton />}>
      <TimeSeriesViewer proof={proof} packageId={packageId} artifactId={first.id} />
    </Suspense>
  );
}
