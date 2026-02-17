import type { ProofPackage } from "../schema/proofPackage.zod.js";
import { readArtifactBytes } from "./artifactStore.js";
import { sha256Prefixed } from "../util/hash.js";

export type VerificationStatus = "VERIFIED" | "UNVERIFIED" | "BROKEN_CHAIN" | "UNSUPPORTED";

export function computeMerkleRootFromArtifactHashes(artifactHashes: Array<`sha256:${string}`>): `sha256:${string}` {
  const sorted = [...artifactHashes].sort();
  const joined = new TextEncoder().encode(sorted.join("|"));
  return sha256Prefixed(joined);
}

export async function verifyProofPackageArtifacts(pkg: ProofPackage, bundleDir: string) {
  const failures: Array<{ code: string; message: string; artifact_id?: string }> = [];
  const computedHashes: Array<`sha256:${string}`> = [];

  for (const [id, art] of Object.entries(pkg.artifacts)) {
    const res = await readArtifactBytes(bundleDir, art.uri);
    if (!res.ok) {
      failures.push({ code: "MISSING_ARTIFACT", message: res.reason, artifact_id: id });
      continue;
    }
    computedHashes.push(res.hash);
    if (res.hash !== art.hash) {
      failures.push({ code: "HASH_MISMATCH", message: "Artifact hash mismatch", artifact_id: id });
    }
  }

  const merkle = computeMerkleRootFromArtifactHashes(computedHashes);
  if (merkle !== pkg.attestation.merkle_root) {
    failures.push({ code: "MERKLE_MISMATCH", message: "Merkle root mismatch" });
  }

  const status: VerificationStatus = failures.length ? "BROKEN_CHAIN" : "VERIFIED";
  return { status, failures, merkle };
}
