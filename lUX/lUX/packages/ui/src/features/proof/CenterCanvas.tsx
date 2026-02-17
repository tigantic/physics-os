import type { ProofMode, ProofPackage, DomainPack } from "@luxury/core";
import { renderCenterScreens } from "./modeComposer";

export function CenterCanvas({
  proof,
  baseline,
  domain,
  mode,
  packageId,
}: {
  proof: ProofPackage;
  baseline?: ProofPackage;
  domain: DomainPack;
  mode: ProofMode;
  packageId: string;
}) {
  return (
    <div id="mode-tabpanel" role="tabpanel" aria-labelledby={`mode-tab-${mode}`} className="space-y-6">
      {renderCenterScreens({ proof, baseline, domain, packageId, mode })}
    </div>
  );
}
