import { memo } from "react";
import type { ProofPackage } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { CodeBlock } from "@/ds/components/CodeBlock";
import { KeyValueGrid } from "@/ds/components/KeyValueGrid";
import { Chip } from "@/ds/components/Chip";

/** Validates container digest format: sha256:<64 hex chars> (matches Zod SHA256 schema). */
const DIGEST_RE = /^sha256:[a-f0-9]{64}$/i;

export const ReproduceScreen = memo(function ReproduceScreen({ proof }: { proof: ProofPackage }) {
  const { container_digest: digest, seed, arch } = proof.meta.environment;

  if (!DIGEST_RE.test(digest) || !Number.isSafeInteger(seed)) {
    return (
      <Card>
        <CardHeader>
          <div className="text-sm text-[var(--color-text-primary)]">Reproduce</div>
        </CardHeader>
        <CardContent>
          <Chip tone="fail">Invalid reproduction metadata</Chip>
        </CardContent>
      </Card>
    );
  }

  const cmd = `docker run --rm ${digest} --seed ${seed}`;
  return (
    <Card>
      <CardHeader>
        <div className="text-sm text-[var(--color-text-primary)]">Reproduce</div>
        <div className="text-xs text-[var(--color-text-tertiary)]">Deterministic command</div>
      </CardHeader>
      <CardContent className="space-y-4">
        <CodeBlock language="bash" copyable>{cmd}</CodeBlock>
        <KeyValueGrid
          entries={[
            { label: "Container Digest", value: digest, mono: true },
            { label: "Seed", value: String(seed), mono: true },
            { label: "Architecture", value: arch, mono: true },
          ]}
          columns={2}
        />
      </CardContent>
    </Card>
  );
});

ReproduceScreen.displayName = "ReproduceScreen";
