import type { ProofMode, ProofPackage, DomainPack } from "@luxury/core";
import { ModeMap } from "@luxury/core";
import { SummaryScreen } from "@/features/screens/Summary";
import { TimelineScreen } from "@/features/screens/Timeline";
import { GatesScreen } from "@/features/screens/Gates";
import { EvidenceScreen } from "@/features/screens/Evidence";
import { IntegrityScreen } from "@/features/screens/Integrity";
import { CompareScreen } from "@/features/screens/Compare";
import { PrimaryViewer } from "@/features/viewers/PrimaryViewer";
import { Card, CardHeader, CardContent } from "@/ds/components/Card";

export interface CenterCtx {
  proof: ProofPackage;
  baseline?: ProofPackage;
  domain: DomainPack;
  bundleDir: string;
  mode: ProofMode;
}

export function renderCenterScreens(ctx: CenterCtx) {
  const layout = ModeMap[ctx.mode];
  const keys = layout.center;
  return keys.map((k, i) => {
    const key = `${k}-${i}`;
    if (k === "HeroMetrics") return <SummaryScreen key={key} proof={ctx.proof} domain={ctx.domain} mode={ctx.mode} />;
    if (k === "ExecutiveNarrative") return <SummaryScreen key={key} proof={ctx.proof} domain={ctx.domain} mode={ctx.mode} />;
    if (k === "Timeline") return <TimelineScreen key={key} proof={ctx.proof} domain={ctx.domain} />;
    if (k === "ClaimCards") return <GatesScreen key={key} proof={ctx.proof} domain={ctx.domain} />;
    if (k === "PrimaryViewer") return <PrimaryViewer key={key} proof={ctx.proof} bundleDir={ctx.bundleDir} />;
    if (k === "RawArtifactViewer") return <EvidenceScreen key={key} proof={ctx.proof} />;
    if (k === "ManifestViewer") {
      return (
        <Card key={key}>
          <CardHeader>
            <div className="text-sm text-[var(--color-text-primary)]">Manifests</div>
            <div className="text-xs text-[var(--color-text-tertiary)]">Gate manifests</div>
          </CardHeader>
          <CardContent>
            <pre className="overflow-x-auto rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] p-3 text-xs text-[var(--color-text-secondary)]">
              {JSON.stringify(ctx.proof.gate_manifests, null, 2)}
            </pre>
          </CardContent>
        </Card>
      );
    }
    if (k === "DiffViewer") return <CompareScreen key={key} proof={ctx.proof} baseline={ctx.baseline} domain={ctx.domain} />;
    if (k === "PaperView") return <SummaryScreen key={key} proof={ctx.proof} domain={ctx.domain} mode={ctx.mode} />;
    if (k === "FigureStaging") return <PrimaryViewer key={key} proof={ctx.proof} bundleDir={ctx.bundleDir} />;
    return <IntegrityScreen key={key} proof={ctx.proof} />;
  });
}
