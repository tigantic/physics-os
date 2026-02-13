// ============================================================
// CSP FIX — Add this to your svelte.config.js under kit: {}
// ============================================================
// The browser is blocking eval() because SvelteKit's server-side
// CSP header overrides the <meta> tag in app.html.
// The meta tag says 'unsafe-eval' but the server header doesn't.
// Server header wins. Fix the server config.
//
// In svelte.config.js, inside the `kit` object:

kit: {
  csp: {
    mode: 'auto',
    directives: {
      'script-src': ['self', 'unsafe-inline', 'unsafe-eval'],
      'style-src': ['self', 'unsafe-inline'],
      'img-src': ['self', 'data:', 'blob:'],
      'connect-src': ['self', 'ws:', 'wss:', 'http:', 'https:'],
      'worker-src': ['self', 'blob:'],
    }
  }
}

// Also update app.html — replace the existing CSP meta tag with:
// <meta http-equiv="Content-Security-Policy"
//   content="default-src 'self';
//            script-src 'self' 'unsafe-inline' 'unsafe-eval';
//            style-src 'self' 'unsafe-inline';
//            img-src 'self' data: blob:;
//            connect-src 'self' ws: wss: http: https:;
//            worker-src 'self' blob:;" />
