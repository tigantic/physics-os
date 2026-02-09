#!/usr/bin/env python3
"""Convert PWA_REPLICATION_NOTE.md to publication-quality PDF.

Uses markdown + weasyprint with KaTeX-style math rendering via CSS.
Produces paper/PWA_REPLICATION_NOTE.pdf alongside the source .md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown
from weasyprint import HTML


# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MD_PATH = SCRIPT_DIR / "PWA_REPLICATION_NOTE.md"
PDF_PATH = SCRIPT_DIR / "PWA_REPLICATION_NOTE.pdf"


# ── LaTeX → MathML (lightweight, no external service) ────────────────
# We render $...$ and $$...$$ as styled <span>/<div> with the raw LaTeX
# preserved, then use weasyprint CSS to style them.  For a full MathML
# pipeline we would need a JS engine; instead we keep the LaTeX notation
# legible in a monospace math font, which is standard for technical PDFs
# distributed as supplementary material.

def _latex_display_to_html(match: re.Match) -> str:
    """Convert $$...$$ to a centered display-math block."""
    latex = match.group(1).strip()
    escaped = latex.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<div class="display-math">{escaped}</div>'


def _latex_inline_to_html(match: re.Match) -> str:
    """Convert $...$ to an inline math span."""
    latex = match.group(1)
    escaped = latex.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<span class="inline-math">{escaped}</span>'


def preprocess_math(md_text: str) -> str:
    """Replace LaTeX math delimiters with styled HTML before markdown parsing."""
    # Display math first (greedy across newlines)
    md_text = re.sub(r'\$\$(.*?)\$\$', _latex_display_to_html, md_text, flags=re.DOTALL)
    # Inline math (non-greedy, single line)
    md_text = re.sub(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)', _latex_inline_to_html, md_text)
    return md_text


# ── CSS ───────────────────────────────────────────────────────────────
CSS = """
@page {
    size: A4;
    margin: 2cm 2.2cm;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9pt;
        color: #666;
        font-family: "Noto Sans", "DejaVu Sans", sans-serif;
    }
    @top-center {
        content: "PWA Compute Engine — Replication Note";
        font-size: 8pt;
        color: #999;
        font-family: "Noto Sans", "DejaVu Sans", sans-serif;
    }
}

body {
    font-family: "Noto Serif", "DejaVu Serif", "Times New Roman", serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #1a1a1a;
    max-width: 100%;
}

h1 {
    font-family: "Noto Sans", "DejaVu Sans", sans-serif;
    font-size: 18pt;
    font-weight: 700;
    color: #111;
    border-bottom: 2.5pt solid #222;
    padding-bottom: 6pt;
    margin-top: 0;
    margin-bottom: 12pt;
}

h2 {
    font-family: "Noto Sans", "DejaVu Sans", sans-serif;
    font-size: 14pt;
    font-weight: 600;
    color: #222;
    border-bottom: 1pt solid #ccc;
    padding-bottom: 4pt;
    margin-top: 24pt;
    margin-bottom: 10pt;
    page-break-after: avoid;
}

h3 {
    font-family: "Noto Sans", "DejaVu Sans", sans-serif;
    font-size: 11.5pt;
    font-weight: 600;
    color: #333;
    margin-top: 18pt;
    margin-bottom: 6pt;
    page-break-after: avoid;
}

p {
    margin: 6pt 0;
    text-align: justify;
}

strong {
    font-weight: 700;
}

code {
    font-family: "Noto Mono", "DejaVu Sans Mono", "Courier New", monospace;
    font-size: 9pt;
    background-color: #f4f4f4;
    padding: 1pt 3pt;
    border-radius: 2pt;
    border: 0.5pt solid #ddd;
}

pre {
    font-family: "Noto Mono", "DejaVu Sans Mono", "Courier New", monospace;
    font-size: 8.5pt;
    background-color: #f8f8f8;
    border: 1pt solid #ddd;
    border-radius: 3pt;
    padding: 8pt 10pt;
    line-height: 1.4;
    overflow-x: auto;
    page-break-inside: avoid;
}

pre code {
    background: none;
    border: none;
    padding: 0;
    font-size: inherit;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 10pt 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

th {
    background-color: #f0f0f0;
    font-family: "Noto Sans", "DejaVu Sans", sans-serif;
    font-weight: 600;
    font-size: 9pt;
    text-align: left;
    padding: 5pt 8pt;
    border: 1pt solid #ccc;
}

td {
    padding: 4pt 8pt;
    border: 1pt solid #ddd;
    vertical-align: top;
}

tr:nth-child(even) {
    background-color: #fafafa;
}

hr {
    border: none;
    border-top: 1pt solid #ccc;
    margin: 20pt 0;
}

/* Math styling */
.display-math {
    font-family: "Noto Mono", "DejaVu Sans Mono", "Courier New", monospace;
    font-size: 9.5pt;
    text-align: center;
    margin: 12pt 0;
    padding: 10pt;
    background-color: #fafafa;
    border-left: 3pt solid #4a90d9;
    border-radius: 2pt;
    white-space: pre-wrap;
    line-height: 1.6;
    page-break-inside: avoid;
    color: #1a1a1a;
}

.inline-math {
    font-family: "Noto Mono", "DejaVu Sans Mono", "Courier New", monospace;
    font-size: 9pt;
    color: #2c3e50;
    white-space: nowrap;
}

/* Lists */
ul, ol {
    margin: 6pt 0;
    padding-left: 22pt;
}

li {
    margin: 3pt 0;
}

/* Blockquote */
blockquote {
    border-left: 3pt solid #ddd;
    margin: 10pt 0;
    padding: 4pt 12pt;
    color: #555;
    font-style: italic;
}

/* Footer citation */
em:last-child {
    font-size: 9.5pt;
    color: #555;
}
"""


def convert() -> None:
    """Read markdown, convert to HTML, render to PDF."""
    if not MD_PATH.exists():
        print(f"ERROR: {MD_PATH} not found", file=sys.stderr)
        sys.exit(1)

    md_text = MD_PATH.read_text(encoding="utf-8")

    # Pre-process LaTeX math
    md_text = preprocess_math(md_text)

    # Convert markdown → HTML
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists"],
        extension_configs={
            "codehilite": {"guess_lang": False, "css_class": "highlight"},
        },
    )

    # Wrap in full HTML document
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    # Render to PDF
    HTML(string=html_doc).write_pdf(str(PDF_PATH))
    print(f"PDF written: {PDF_PATH}")
    print(f"  Size: {PDF_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    convert()
