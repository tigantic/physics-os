import katex from "katex";
import { mathjax } from "mathjax-full/js/mathjax.js";
import { MathML } from "mathjax-full/js/input/mathml.js";
import { SVG } from "mathjax-full/js/output/svg.js";
import { liteAdaptor } from "mathjax-full/js/adaptors/liteAdaptor.js";
import { RegisterHTMLHandler } from "mathjax-full/js/handlers/html.js";
import type { LiteAdaptor } from "mathjax-full/js/adaptors/liteAdaptor.js";
import { normalizeSvg } from "./normalizeSvg.js";
import { sanitizeSvg } from "./sanitizeSvg.js";

let initialized = false;
let adaptor: LiteAdaptor;
let mathDocument: ReturnType<typeof mathjax.document>;

function init(): void {
  if (initialized) return;
  adaptor = liteAdaptor();
  RegisterHTMLHandler(adaptor);
  const input = new MathML();
  const output = new SVG({ fontCache: "none" });
  mathDocument = mathjax.document("", { InputJax: input, OutputJax: output });
  initialized = true;
}

export function renderLatexToSvg(latex: string, displayMode = true): string {
  init();
  const mathml = katex.renderToString(latex, { output: "mathml", throwOnError: false });
  const node = mathDocument.convert(mathml, { display: displayMode });
  const svg = adaptor.outerHTML(node);
  return sanitizeSvg(normalizeSvg(svg));
}
