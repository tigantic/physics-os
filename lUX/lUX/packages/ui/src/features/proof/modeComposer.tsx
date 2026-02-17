import * as React from "react";
import dynamic from "next/dynamic";
import type { ProofMode, ProofPackage, DomainPack } from "@luxury/core";
import { ModeMap } from "@luxury/core";
import { Card, CardHeader, CardContent } from "@/ds/components/Card";
import { Skeleton } from "@/ds/components/Skeleton";
import { ScreenErrorBoundary } from "@/ds/components/ScreenErrorBoundary";

/* ── Shared loading fallback for dynamically-imported screens ─────────── */

function ScreenSkeleton() {
  return <Skeleton rows={4} heightClass="h-8" gapClass="gap-3" />;
}

/* ── Dynamic imports — each screen is a separate chunk ────────────────── */

const SummaryScreen = dynamic(
  () => import("@/features/screens/Summary").then((m) => m.SummaryScreen),
  { loading: ScreenSkeleton },
);
const TimelineScreen = dynamic(
  () => import("@/features/screens/Timeline").then((m) => m.TimelineScreen),
  { loading: ScreenSkeleton },
);
const GatesScreen = dynamic(
  () => import("@/features/screens/Gates").then((m) => m.GatesScreen),
  { loading: ScreenSkeleton },
);
const EvidenceScreen = dynamic(
  () => import("@/features/screens/Evidence").then((m) => m.EvidenceScreen),
  { loading: ScreenSkeleton },
);
const IntegrityScreen = dynamic(
  () => import("@/features/screens/Integrity").then((m) => m.IntegrityScreen),
  { loading: ScreenSkeleton },
);
const CompareScreen = dynamic(
  () => import("@/features/screens/Compare").then((m) => m.CompareScreen),
  { loading: ScreenSkeleton },
);
const PrimaryViewer = dynamic(
  () => import("@/features/viewers/PrimaryViewer").then((m) => m.PrimaryViewer),
  { loading: ScreenSkeleton },
);

export interface CenterCtx {
  proof: ProofPackage;
  baseline?: ProofPackage;
  domain: DomainPack;
  packageId: string;
  mode: ProofMode;
}

export function renderCenterScreens(ctx: CenterCtx) {
  const layout = ModeMap[ctx.mode];
  const keys = layout.center;
  return keys.map((k, i) => {
    const key = `${k}-${i}`;

    /** Wrap each screen in an error boundary so a single failure
     *  shows an inline error card instead of crashing the entire page. */
    function wrap(screenName: string, node: React.ReactNode) {
      return (
        <ScreenErrorBoundary key={key} screenName={screenName}>
          {node}
        </ScreenErrorBoundary>
      );
    }

    if (k === "HeroMetrics")
      return wrap("Summary", <SummaryScreen proof={ctx.proof} domain={ctx.domain} mode={ctx.mode} />);
    if (k === "ExecutiveNarrative")
      return wrap("Summary", <SummaryScreen proof={ctx.proof} domain={ctx.domain} mode={ctx.mode} />);
    if (k === "Timeline")
      return wrap("Timeline", <TimelineScreen proof={ctx.proof} domain={ctx.domain} />);
    if (k === "ClaimCards")
      return wrap("Gates", <GatesScreen proof={ctx.proof} />);
    if (k === "PrimaryViewer")
      return wrap("Viewer", <PrimaryViewer proof={ctx.proof} packageId={ctx.packageId} />);
    if (k === "RawArtifactViewer")
      return wrap("Evidence", <EvidenceScreen proof={ctx.proof} />);
    if (k === "ManifestViewer") {
      return wrap(
        "Manifests",
        <Card>
          <CardHeader>
            <h2 className="text-sm text-[var(--color-text-primary)]">Manifests</h2>
            <div className="text-xs text-[var(--color-text-tertiary)]">Gate manifests</div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] p-3">
              <pre className="text-xs leading-relaxed">
                {Object.entries(ctx.proof.gate_manifests ?? {}).map(([gateId, manifest]) => (
                  <div key={gateId} className="mb-3 last:mb-0">
                    <span className="font-semibold text-[var(--color-accent)]">{gateId}</span>
                    {"\n"}
                    {Object.entries(manifest as Record<string, unknown>).map(([k2, v]) => (
                      <div key={k2} className="ml-4">
                        <span className="text-[var(--color-text-tertiary)]">{k2}</span>
                        <span className="text-[var(--color-text-tertiary)]">: </span>
                        <span className="text-[var(--color-text-secondary)]">
                          {typeof v === "string" ? `"${v}"` : JSON.stringify(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                ))}
              </pre>
            </div>
          </CardContent>
        </Card>,
      );
    }
    if (k === "DiffViewer")
      return wrap("Compare", <CompareScreen proof={ctx.proof} baseline={ctx.baseline} domain={ctx.domain} />);
    if (k === "PaperView")
      return wrap("Paper", <SummaryScreen proof={ctx.proof} domain={ctx.domain} mode={ctx.mode} />);
    if (k === "FigureStaging")
      return wrap("Figures", <PrimaryViewer proof={ctx.proof} packageId={ctx.packageId} />);
    return wrap("Integrity", <IntegrityScreen proof={ctx.proof} />);
  });
}
