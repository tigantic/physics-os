"use client";

import * as React from "react";
import { reportError } from "@/lib/reportError";

/**
 * Global error boundary — catches root layout errors.
 * Must render its own <html>/<body> since the root layout may have crashed.
 *
 * IMPORTANT: All styles are inline because CSS custom properties and Tailwind
 * classes are unavailable — the root layout (and therefore all CSS imports)
 * may have crashed. Colors approximate the dark-mode design tokens.
 */
export default function GlobalError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  const retryRef = React.useRef<HTMLButtonElement>(null);

  React.useEffect(() => {
    retryRef.current?.focus();
  }, []);

  React.useEffect(() => {
    reportError(error, "GlobalError");
  }, [error]);

  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          fontFamily: "system-ui, sans-serif",
          background: "#0a0a0a",
          color: "#e5e5e5",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          minHeight: "100vh",
        }}
      >
        <div
          role="alert"
          aria-live="assertive"
          style={{
            maxWidth: 600,
            padding: 32,
            border: "1px solid #333",
            borderRadius: 12,
            background: "#141414",
          }}
        >
          <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 1, color: "#999" }}>
            Fatal Render Error
          </div>
          <h1 style={{ marginTop: 8, fontSize: 18, fontWeight: 600 }}>Viewer Unrecoverable</h1>
          <pre
            style={{
              marginTop: 16,
              fontFamily: "monospace",
              fontSize: 12,
              color: "#ef4444",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {error.message}
          </pre>
          {error.digest && <div style={{ marginTop: 8, fontSize: 11, color: "#777" }}>Digest: {error.digest}</div>}
          <button
            ref={retryRef}
            type="button"
            onClick={reset}
            style={{
              marginTop: 20,
              padding: "8px 20px",
              fontSize: 13,
              fontWeight: 500,
              border: "1px solid #444",
              borderRadius: 6,
              background: "#1a1a1a",
              color: "#e5e5e5",
              cursor: "pointer",
            }}
          >
            Retry
          </button>
        </div>
      </body>
    </html>
  );
}
