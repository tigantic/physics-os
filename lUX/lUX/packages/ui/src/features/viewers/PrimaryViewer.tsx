import type { ProofPackage } from "@luxury/core";
import { TimeSeriesViewer } from "./TimeSeriesViewer";
import { Chip } from "@/ds/components/Chip";

export function PrimaryViewer({ proof, bundleDir }: { proof: ProofPackage; bundleDir: string }) {
  if (proof.verification?.status === "BROKEN_CHAIN") return <Chip tone="fail">BROKEN_CHAIN</Chip>;
  const first = Object.values(proof.artifacts).find((a) => a.type === "time_series");
  if (!first) return <Chip tone="warn">Data Unavailable</Chip>;
  return <TimeSeriesViewer proof={proof} bundleDir={bundleDir} artifactId={first.id} />;
}
