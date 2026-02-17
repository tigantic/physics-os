import { renderLatexToSvg } from "@luxury/core";

export function MathBlock({ latex }: { latex: string }) {
  const svg = renderLatexToSvg(latex, true);
  return <div className="overflow-x-auto" dangerouslySetInnerHTML={{ __html: svg }} />;
}
