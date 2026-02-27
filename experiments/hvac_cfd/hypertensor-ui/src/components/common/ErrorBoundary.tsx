/**
 * Error Boundary - Graceful Error Handling
 * 
 * Provides graceful failure handling for visualization components
 * per Article VIII of the Constitution.
 */

'use client';

import React, { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Bug, ChevronDown, ChevronUp } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

// ============================================
// TYPES
// ============================================

interface ErrorBoundaryProps {
  /** Child components to render */
  children: ReactNode;
  /** Custom fallback UI - overrides default error display */
  fallback?: ReactNode;
  /** Callback when error occurs */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  /** Custom reset handler */
  onReset?: () => void;
  /** Component name for error reporting */
  componentName?: string;
  /** Whether to show technical details */
  showDetails?: boolean;
  /** Compact error display for embedded components */
  compact?: boolean;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  showStack: boolean;
}

// ============================================
// ERROR BOUNDARY COMPONENT
// ============================================

/**
 * Error Boundary for catching and displaying React errors
 * 
 * @example
 * ```tsx
 * <ErrorBoundary componentName="MeshViewer" onError={logToSentry}>
 *   <MeshViewer mesh={mesh} />
 * </ErrorBoundary>
 * ```
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      showStack: false,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.setState({ errorInfo });
    
    // Call error callback if provided
    this.props.onError?.(error, errorInfo);
    
    // Log error in development
    if (process.env.NODE_ENV === 'development') {
      console.group(`🚨 Error in ${this.props.componentName ?? 'component'}`);
      console.error('Error:', error);
      console.error('Component Stack:', errorInfo.componentStack);
      console.groupEnd();
    }
  }

  handleReset = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      showStack: false,
    });
    this.props.onReset?.();
  };

  toggleStack = (): void => {
    this.setState((prev) => ({ showStack: !prev.showStack }));
  };

  render(): ReactNode {
    const { hasError, error, errorInfo, showStack } = this.state;
    const { children, fallback, componentName, showDetails = true, compact = false } = this.props;

    if (!hasError) {
      return children;
    }

    // Custom fallback UI
    if (fallback) {
      return fallback;
    }

    // Compact error display for embedded components
    if (compact) {
      return (
        <div className="flex items-center justify-center h-full min-h-[100px] bg-destructive/5 rounded-lg border border-destructive/20 p-4">
          <div className="text-center space-y-2">
            <AlertTriangle className="h-8 w-8 mx-auto text-destructive" />
            <p className="text-sm text-muted-foreground">
              {componentName ? `${componentName} failed to load` : 'Component error'}
            </p>
            <Button variant="outline" size="sm" onClick={this.handleReset}>
              <RefreshCw className="h-3 w-3 mr-1" />
              Retry
            </Button>
          </div>
        </div>
      );
    }

    // Full error display
    return (
      <Card className="border-destructive/50 bg-destructive/5">
        <CardHeader>
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-destructive" />
            <CardTitle className="text-lg">
              {componentName ? `${componentName} Error` : 'Something went wrong'}
            </CardTitle>
          </div>
          <CardDescription>
            An error occurred while rendering this component.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Error message */}
          <div className="bg-background/50 rounded-md p-3 border">
            <p className="text-sm font-mono text-destructive">
              {error?.message ?? 'Unknown error'}
            </p>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2">
            <Button onClick={this.handleReset}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Try Again
            </Button>
            {showDetails && errorInfo && (
              <Button variant="outline" onClick={this.toggleStack}>
                <Bug className="h-4 w-4 mr-2" />
                {showStack ? 'Hide' : 'Show'} Details
                {showStack ? (
                  <ChevronUp className="h-4 w-4 ml-2" />
                ) : (
                  <ChevronDown className="h-4 w-4 ml-2" />
                )}
              </Button>
            )}
          </div>

          {/* Stack trace (collapsible) */}
          {showStack && errorInfo && (
            <div className="bg-muted rounded-md p-3 overflow-auto max-h-[300px]">
              <pre className="text-xs font-mono whitespace-pre-wrap text-muted-foreground">
                {error?.stack}
                {'\n\nComponent Stack:'}
                {errorInfo.componentStack}
              </pre>
            </div>
          )}
        </CardContent>
      </Card>
    );
  }
}

// ============================================
// FUNCTIONAL WRAPPER HOOK
// ============================================

interface UseErrorBoundaryReturn {
  /** Reset the error boundary state */
  resetBoundary: () => void;
  /** Show an error programmatically */
  showError: (error: Error) => void;
}

/**
 * Hook to control error boundary from child components
 * 
 * @example
 * ```tsx
 * function ChildComponent() {
 *   const { showError } = useErrorBoundary();
 *   
 *   const handleAsyncError = async () => {
 *     try {
 *       await riskyOperation();
 *     } catch (error) {
 *       showError(error as Error);
 *     }
 *   };
 * }
 * ```
 */
// Note: This would require React context - simplified implementation below

// ============================================
// SPECIALIZED ERROR BOUNDARIES
// ============================================

interface CFDErrorBoundaryProps {
  children: ReactNode;
  componentName?: string;
}

/**
 * Error boundary specifically for CFD visualization components
 */
export function CFDErrorBoundary({ children, componentName }: CFDErrorBoundaryProps) {
  return (
    <ErrorBoundary
      componentName={componentName ?? 'CFD Component'}
      compact
      onError={(error, info) => {
        // In production, send to error tracking service
        if (process.env.NODE_ENV === 'production') {
          // TODO: Integrate with error tracking (Sentry, etc.)
          console.error('CFD Component Error:', error, info);
        }
      }}
    >
      {children}
    </ErrorBoundary>
  );
}

/**
 * Error boundary for 3D viewer components with WebGL fallback handling
 */
export function ViewerErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary
      componentName="3D Viewer"
      fallback={
        <div className="flex flex-col items-center justify-center h-full min-h-[400px] bg-muted/30 rounded-lg border-2 border-dashed">
          <AlertTriangle className="h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-lg font-semibold mb-2">3D Viewer Unavailable</h3>
          <p className="text-sm text-muted-foreground text-center max-w-md mb-4">
            The 3D visualization could not be loaded. This may be due to WebGL
            not being supported or a graphics driver issue.
          </p>
          <Button variant="outline" onClick={() => window.location.reload()}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Reload Page
          </Button>
        </div>
      }
    >
      {children}
    </ErrorBoundary>
  );
}

/**
 * Error boundary for chart components
 */
export function ChartErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary
      componentName="Chart"
      compact
      onError={(error) => {
        console.error('Chart rendering error:', error);
      }}
    >
      {children}
    </ErrorBoundary>
  );
}

export default ErrorBoundary;
