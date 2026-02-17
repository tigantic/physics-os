/**
 * Critical CSS inline injection for eliminating FOUC (Flash of Unstyled Content).
 *
 * Inlines the minimal set of design tokens needed for above-the-fold rendering
 * directly in the <head>, ensuring:
 *   - Correct background/text colors on first paint (no white flash)
 *   - Font family declarations available before external CSS loads
 *   - Theme-critical custom properties set immediately
 *
 * Next.js bundles all @import CSS, but the JS-driven injection can cause a
 * single-frame FOUC on cold loads. This inline style eliminates that gap.
 *
 * Covers ROADMAP item:
 *   - Phase 6: "Preload critical CSS (tokens.css, typography.css) via <link rel='preload'>"
 */
export function CriticalCSS() {
  return (
    <style
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{
        __html: `
/* Critical path — inlined to prevent FOUC */
:root,[data-theme="dark"]{
  --color-bg-base:#0e0e13;
  --color-bg-surface:#16161d;
  --color-bg-raised:#1e1e28;
  --color-text-primary:#eeeef0;
  --color-text-secondary:#b0afb8;
  --color-text-tertiary:#8c8798;
  --color-accent:#4b7bf5;
  --color-border-base:#2a2a36;
  --radius-inner:0.375rem;
  --radius-card:0.5rem;
  --font-sans:var(--font-sans),ui-sans-serif,system-ui,sans-serif;
  --font-mono:var(--font-mono),ui-monospace,monospace;
  color-scheme:dark;
}
[data-theme="light"]{
  --color-bg-base:#fafafa;
  --color-bg-surface:#ffffff;
  --color-bg-raised:#f4f4f5;
  --color-text-primary:#18181b;
  --color-text-secondary:#52525b;
  --color-text-tertiary:#71717a;
  --color-accent:#3b63c9;
  --color-border-base:#e4e4e7;
  color-scheme:light;
}
html,body{
  background:var(--color-bg-base);
  color:var(--color-text-primary);
  -webkit-font-smoothing:antialiased;
  -moz-osx-font-smoothing:grayscale;
}
`.trim(),
      }}
    />
  );
}
