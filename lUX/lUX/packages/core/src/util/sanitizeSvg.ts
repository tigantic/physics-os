/**
 * Defense-in-depth SVG sanitizer.
 *
 * Strips known-dangerous patterns from SVG output to guard against XSS
 * if the upstream MathJax/KaTeX pipeline is ever compromised.
 *
 * This targets the specific threat model of server-generated SVG from
 * trusted libraries. For user-supplied SVG, use a full DOM sanitizer
 * such as DOMPurify.
 */
export function sanitizeSvg(svg: string): string {
  // 1. Strip <script> elements (including self-closing and nested variants).
  let clean = svg.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script\s*>/gi, "");
  clean = clean.replace(/<script\b[^>]*\/>/gi, "");

  // 2. Strip on* event-handler attributes (onclick, onerror, onload, etc.).
  clean = clean.replace(/\s+on[a-z]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]*)/gi, "");

  // 3. Strip javascript: and vbscript: protocol in href / xlink:href.
  clean = clean.replace(/((?:xlink:)?href\s*=\s*["'])\s*(?:javascript|vbscript)\s*:/gi, "$1#blocked:");

  // 4. Strip data: URIs in href except safe image types.
  clean = clean.replace(
    /((?:xlink:)?href\s*=\s*["'])\s*data:(?!image\/(?:png|jpeg|gif|svg\+xml)[;,])/gi,
    "$1#blocked:",
  );

  // 5. Remove <foreignObject> (can embed arbitrary HTML).
  clean = clean.replace(/<foreignObject\b[^<]*(?:(?!<\/foreignObject>)<[^<]*)*<\/foreignObject\s*>/gi, "");
  clean = clean.replace(/<foreignObject\b[^>]*\/>/gi, "");

  // 6. Remove <use> with external references (can pull in remote SVG).
  clean = clean.replace(/<use\b[^>]*\bhref\s*=\s*["'][^#][^"']*["'][^>]*\/?>/gi, "");

  return clean;
}
