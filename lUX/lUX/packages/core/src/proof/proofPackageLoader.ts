import fs from "node:fs";
import path from "node:path";
import { ProofPackageSchema } from "../schema/proofPackage.zod.js";
import type { ProofPackage } from "../schema/proofPackage.zod.js";
import { verifyProofPackageArtifacts } from "./integrity.js";

function deepFreeze<T>(obj: T): T {
  if (obj && typeof obj === "object") {
    Object.freeze(obj);
    for (const v of Object.values(obj as Record<string, unknown>)) {
      if (v && typeof v === "object" && !Object.isFrozen(v)) deepFreeze(v);
    }
  }
  return obj;
}

export type LoadedProofPackage = {
  bundleDir: string;
  proof: ProofPackage;
};

export function loadProofPackageFromDir(bundleDir: string): LoadedProofPackage {
  const jsonPath = path.resolve(bundleDir, "proofPackage.json");
  if (!fs.existsSync(jsonPath)) throw new Error("proofPackage.json missing");
  const raw = JSON.parse(fs.readFileSync(jsonPath, "utf8"));
  const parsed = ProofPackageSchema.parse(raw);
  const verification = verifyProofPackageArtifacts(parsed, bundleDir);

  const proof: ProofPackage = deepFreeze({
    ...parsed,
    verification: {
      status: verification.status,
      failures: verification.failures.map(f => ({ code: f.code, message: f.message, artifact_id: f.artifact_id })),
      verifier_version: "0.1.0"
    }
  });
  return { bundleDir, proof };
}