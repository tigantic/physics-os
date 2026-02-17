import type { ProofPackage } from "@luxury/core";
import { TimeSeriesViewer } from "./TimeSeriesViewer";
import { Chip } from "@/ds/components/Chip";

export function PrimaryViewer({ proof, packageId }: { proof: ProofPackage; packageId: string }) {
  if (proof.verification?.status === "BROKEN_CHAIN") return <Chip tone="fail">BROKEN_CHAIN</Chip>;
  const first = Object.values(proof.artifacts).find((a) => a.type === "time_series");
  if (!first) return <Chip tone="warn">Data Unavailable</Chip>;
  return <TimeSeriesViewer proof={proof} packageId={packageId} artifactId={first.id} />;
}
