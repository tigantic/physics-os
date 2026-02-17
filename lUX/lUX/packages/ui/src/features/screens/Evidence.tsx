import { memo } from "react";
import type { ProofPackage } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { Disclosure } from "@/ds/components/Disclosure";
import { KeyValueGrid } from "@/ds/components/KeyValueGrid";

export const EvidenceScreen = memo(function EvidenceScreen({ proof }: { proof: ProofPackage }) {
  return (
    <Card>
      <CardHeader>
        <div className="text-sm text-[var(--color-text-primary)]">Evidence</div>
        <div className="text-xs text-[var(--color-text-tertiary)]">{Object.keys(proof.artifacts).length} artifacts</div>
      </CardHeader>
      <CardContent className="space-y-3">
        {Object.values(proof.artifacts).map((a) => (
          <Disclosure key={a.id} title={`${a.id} · ${a.type}`}>
            <KeyValueGrid
              entries={[
                { label: "Hash", value: a.hash, mono: true },
                { label: "URI", value: a.uri, mono: true },
                { label: "MIME", value: a.mime_type },
              ]}
            />
          </Disclosure>
        ))}
      </CardContent>
    </Card>
  );
});

EvidenceScreen.displayName = "EvidenceScreen";
