"use client";
import * as React from "react";
import { cn } from "@/config/utils";

export interface CodeBlockProps {
  /** The source code or structured text to display */
  children: string;
  /** Programming language — used for aria-label, not syntax highlighting */
  language?: string;
  /** Show line numbers. Default: false */
  lineNumbers?: boolean;
  /** Maximum height before scroll. Default: 400 */
  maxHeight?: number;
  /** Class applied to the wrapper */
  className?: string;
  /** Enable copy button. Default: true */
  copyable?: boolean;
}

export const CodeBlock = React.memo(function CodeBlock({
  children,
  language,
  lineNumbers = false,
  maxHeight = 400,
  className,
  copyable = true,
}: CodeBlockProps) {
  const [copied, setCopied] = React.useState(false);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(children);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      // Silently fail — no toast system yet
    }
  }

  const lines = children.split("\n");

  return (
    <div
      className={cn(
        "group relative rounded-[var(--radius-inner)] border border-[var(--color-border-base)] bg-[var(--color-bg-surface)]",
        className,
      )}
    >
      {/* Header bar */}
      {(language || copyable) && (
        <div className="flex items-center justify-between border-b border-[var(--color-border-base)] px-3 py-1.5">
          {language && (
            <span className="text-[10px] font-medium uppercase tracking-wider text-[var(--color-text-tertiary)]">
              {language}
            </span>
          )}
          {copyable && (
            <button
              type="button"
              onClick={handleCopy}
              aria-label={copied ? "Copied" : "Copy code"}
              className="rounded-[var(--radius-control)] px-2 py-0.5 text-[10px] font-medium text-[var(--color-text-tertiary)] transition-colors duration-hover ease-lux-out hover:bg-[var(--color-bg-hover)] hover:text-[var(--color-text-secondary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent-border)]"
            >
              {copied ? "Copied" : "Copy"}
            </button>
          )}
        </div>
      )}

      {/* Code body */}
      <div
        className="overflow-auto"
        style={{ maxHeight }}
        role="region"
        aria-label={language ? `${language} code block` : "Code block"}
        tabIndex={0}
      >
        <pre className="p-3 text-xs leading-relaxed">
          <code>
            {lines.map((line, i) => (
              <div key={i} className="flex">
                {lineNumbers && (
                  <span
                    className="mr-4 inline-block w-8 select-none text-right tabular-nums text-[var(--color-text-tertiary)]/60"
                    aria-hidden="true"
                  >
                    {i + 1}
                  </span>
                )}
                <span className="flex-1 text-[var(--color-text-secondary)]">{line || "\u00A0"}</span>
              </div>
            ))}
          </code>
        </pre>
      </div>
    </div>
  );
});
