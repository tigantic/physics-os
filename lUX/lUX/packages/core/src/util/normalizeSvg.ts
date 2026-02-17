export function normalizeSvg(svg: string): string {
  return svg
    .replace(/\r\n/g, "\n")
    .replace(/\s+/g, " ")
    .replace(/>\s+</g, "><")
    .trim();
}
