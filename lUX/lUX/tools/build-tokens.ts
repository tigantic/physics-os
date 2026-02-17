import fs from "node:fs";
import path from "node:path";

type AnyRecord = Record<string, unknown>;

function isObject(v: unknown): v is AnyRecord {
  return !!v && typeof v === "object" && !Array.isArray(v);
}

function flatten(obj: AnyRecord, prefix: string[] = [], out: Record<string, string> = {}) {
  const keys = Object.keys(obj).sort();
  for (const k of keys) {
    const v = obj[k];
    if (isObject(v)) flatten(v, [...prefix, k], out);
    else out[[...prefix, k].join("-")] = String(v);
  }
  return out;
}

const root = process.cwd();
const tokensPath = path.resolve(root, "design", "tokens.json");
const outCss = path.resolve(root, "packages", "ui", "src", "ds", "tokens.css");
const outTs = path.resolve(root, "packages", "ui", "src", "ds", "tokens.ts");

const tokens = JSON.parse(fs.readFileSync(tokensPath, "utf8")) as AnyRecord;
const flat = flatten(tokens);

const lines: string[] = [];
lines.push(":root {");
for (const k of Object.keys(flat).sort()) {
  lines.push(`  --${k}: ${flat[k]};`);
}
lines.push("}");
lines.push("");

fs.writeFileSync(outCss, lines.join("\n"), "utf8");
fs.writeFileSync(outTs, `export const TOKENS = ${JSON.stringify(tokens, null, 2)} as const;\n`, "utf8");

console.log(`Wrote ${outCss}`);
console.log(`Wrote ${outTs}`);
