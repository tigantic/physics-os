import { renderLatexToSvg } from "@luxury/core";

/**
 * Renders a LaTeX expression as SVG.
 *
 * Safety: `renderLatexToSvg` in @luxury/core runs the output through
 * `sanitizeSvg()` (strips <script>, on* handlers, javascript: URIs,
 * <foreignObject>, and external <use> references) before returning.
 * The LaTeX input must originate from trusted domain templates — never
 * from unsanitised user input.
 */
export function MathBlock({ latex }: { latex: string }) {
  const svg = renderLatexToSvg(latex, true);
  return (
    <div
      role="img"
      aria-label={`Math equation: ${latex}`}
      className="overflow-x-auto"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
