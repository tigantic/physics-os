import fs from "node:fs/promises";
import path from "node:path";
import { ProofPackageSchema } from "../schema/proofPackage.zod.js";
import type { ProofPackage } from "../schema/proofPackage.zod.js";
import { verifyProofPackageArtifacts } from "./integrity.js";
import { deepFreeze } from "../util/deepFreeze.js";

export type LoadedProofPackage = {
  bundleDir: string;
  proof: ProofPackage;
};

export async function loadProofPackageFromDir(bundleDir: string): Promise<LoadedProofPackage> {
  const jsonPath = path.resolve(bundleDir, "proofPackage.json");
  let rawText: string;
  try {
    rawText = await fs.readFile(jsonPath, "utf8");
  } catch {
    throw new Error("proofPackage.json missing");
  }
  const raw = JSON.parse(rawText);
  const parsed = ProofPackageSchema.parse(raw);
  const verification = await verifyProofPackageArtifacts(parsed, bundleDir);

  const proof: ProofPackage = deepFreeze({
    ...parsed,
    verification: {
      status: verification.status,
      failures: verification.failures.map((f) => ({ code: f.code, message: f.message, artifact_id: f.artifact_id })),
      verifier_version: "0.1.0",
    },
  });
  return { bundleDir, proof };
}
