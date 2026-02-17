export type ProofMode = "EXECUTIVE" | "REVIEW" | "AUDIT" | "PUBLICATION";

export type RailVariant = "collapsed" | "claimsTree" | "flatClaims" | "chapters";

export interface ModeLayout {
  readonly leftRail: { readonly variant: RailVariant; readonly includeArtifactBrowserEntry?: boolean };
  readonly center: readonly string[];
  readonly rightRail: readonly string[];
}

export const ModeMap: Readonly<Record<ProofMode, ModeLayout>> = {
  EXECUTIVE: {
    leftRail: { variant: "collapsed" },
    center: ["HeroMetrics", "ExecutiveNarrative"],
    rightRail: ["ActionPanel"]
  },
  REVIEW: {
    leftRail: { variant: "claimsTree" },
    center: ["Timeline", "ClaimCards", "PrimaryViewer"],
    rightRail: ["GateInspector", "AnomalyExplainer", "ReproduceMini"]
  },
  AUDIT: {
    leftRail: { variant: "flatClaims", includeArtifactBrowserEntry: true },
    center: ["RawArtifactViewer", "ManifestViewer", "DiffViewer"],
    rightRail: ["ChainInspector", "VerificationFailures", "ReproduceFull"]
  },
  PUBLICATION: {
    leftRail: { variant: "chapters" },
    center: ["PaperView", "FigureStaging"],
    rightRail: ["ExportTools", "CitationTools"]
  }
};
