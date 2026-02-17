import { renderLatexToSvg } from "@luxury/core";

export function MathBlock({ latex }: { latex: string }) {
  const svg = renderLatexToSvg(latex, true);
  return <div role="img" aria-label={`Math equation: ${latex}`} className="overflow-x-auto" dangerouslySetInnerHTML={{ __html: svg }} />;
}
