"use client";

import * as React from "react";

interface ScreenErrorBoundaryProps {
  /** Fallback screen name for error display */
  screenName: string;
  children: React.ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * ScreenErrorBoundary — wraps individual screen components in CenterCanvas.
 * If a single screen throws, it shows an inline error card instead of
 * taking down the entire proof workspace via the route-level error boundary.
 */
export class ScreenErrorBoundary extends React.Component<ScreenErrorBoundaryProps, State> {
  constructor(props: ScreenErrorBoundaryProps) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  override componentDidCatch(error: Error, info: React.ErrorInfo) {
    // Log to console in development; production would use reportError
    console.error(`[ScreenErrorBoundary] ${this.props.screenName}:`, error, info.componentStack);
  }

  override render() {
    if (this.state.error) {
      return (
        <div
          role="alert"
          className="rounded-[var(--radius-outer)] border border-[var(--color-status-fail)]/30 bg-[var(--color-status-fail)]/5 p-4"
        >
          <div className="text-xs font-medium uppercase tracking-wider text-[var(--color-status-fail)]">
            Screen Error
          </div>
          <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
            <span className="font-medium text-[var(--color-text-primary)]">{this.props.screenName}</span> failed to
            render.
          </p>
          <pre className="mt-2 max-h-24 overflow-auto whitespace-pre-wrap break-words font-mono text-xs text-[var(--color-status-fail)]">
            {this.state.error.message}
          </pre>
          <button
            type="button"
            onClick={() => this.setState({ error: null })}
            className="mt-3 rounded-md border border-[var(--color-border-base)] bg-[var(--color-bg-surface)] px-3 py-1.5 text-xs font-medium text-[var(--color-text-primary)] transition-colors hover:bg-[var(--color-bg-hover)]"
          >
            Retry
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
