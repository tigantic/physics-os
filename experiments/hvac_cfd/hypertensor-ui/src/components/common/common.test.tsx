/**
 * Common Components Tests
 * 
 * Tests for ErrorBoundary, SkipLink, NetworkStatusBanner
 * Constitutional Compliance: Article III Testing Protocols
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ErrorBoundary } from './ErrorBoundary';
import { SkipLink } from './SkipLink';

// ============================================
// ERROR BOUNDARY TESTS
// ============================================

describe('ErrorBoundary', () => {
  const ErrorThrowingComponent = () => {
    throw new Error('Test error');
  };

  const GoodComponent = () => <div>Good content</div>;

  beforeEach(() => {
    // Suppress console errors from error boundary
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(console, 'group').mockImplementation(() => {});
    vi.spyOn(console, 'groupEnd').mockImplementation(() => {});
  });

  it('should render children when no error', () => {
    render(
      <ErrorBoundary>
        <GoodComponent />
      </ErrorBoundary>
    );

    expect(screen.getByText('Good content')).toBeInTheDocument();
  });

  it('should catch and display errors', () => {
    render(
      <ErrorBoundary>
        <ErrorThrowingComponent />
      </ErrorBoundary>
    );

    expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
  });

  it('should use custom fallback when provided', () => {
    render(
      <ErrorBoundary fallback={<div>Custom error UI</div>}>
        <ErrorThrowingComponent />
      </ErrorBoundary>
    );

    expect(screen.getByText('Custom error UI')).toBeInTheDocument();
  });

  it('should call onError callback when error occurs', () => {
    const onError = vi.fn();

    render(
      <ErrorBoundary onError={onError}>
        <ErrorThrowingComponent />
      </ErrorBoundary>
    );

    expect(onError).toHaveBeenCalled();
  });

  it('should display component name in error', () => {
    render(
      <ErrorBoundary componentName="TestComponent">
        <ErrorThrowingComponent />
      </ErrorBoundary>
    );

    expect(screen.getByText(/TestComponent/)).toBeInTheDocument();
  });

  it('should reset on retry button click', () => {
    const onReset = vi.fn();

    const { container } = render(
      <ErrorBoundary onReset={onReset}>
        <ErrorThrowingComponent />
      </ErrorBoundary>
    );

    // Find and click retry button
    const retryButton = container.querySelector('button');
    if (retryButton) {
      fireEvent.click(retryButton);
      expect(onReset).toHaveBeenCalled();
    }
  });

  it('should handle compact mode', () => {
    const { container } = render(
      <ErrorBoundary compact>
        <ErrorThrowingComponent />
      </ErrorBoundary>
    );

    // Compact mode should still render error state
    expect(container.textContent).toContain('error');
  });
});

// ============================================
// SKIP LINK TESTS
// ============================================

describe('SkipLink', () => {
  it('should render with default target', () => {
    render(<SkipLink />);

    const link = screen.getByText('Skip to main content');
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute('href', '#main-content');
  });

  it('should use custom target id', () => {
    render(<SkipLink targetId="content-area" />);

    const link = screen.getByText('Skip to main content');
    expect(link).toHaveAttribute('href', '#content-area');
  });

  it('should apply custom className', () => {
    render(<SkipLink className="custom-skip" />);

    const link = screen.getByText('Skip to main content');
    expect(link).toHaveClass('custom-skip');
  });

  it('should have skip-to-main class', () => {
    render(<SkipLink />);

    const link = screen.getByText('Skip to main content');
    expect(link).toHaveClass('skip-to-main');
  });

  it('should be a link element', () => {
    render(<SkipLink />);

    const link = screen.getByText('Skip to main content');
    expect(link.tagName).toBe('A');
  });
});

// ============================================
// NETWORK STATUS BANNER TESTS
// ============================================

// Mock the useNetworkStatus hook
vi.mock('@/hooks/useNetworkStatus', () => ({
  useNetworkStatus: vi.fn(() => ({
    isOnline: true,
    isApiReachable: true,
    isSlowConnection: false,
    checkApiConnectivity: vi.fn(),
  })),
}));

import { NetworkStatusBanner } from './NetworkStatusBanner';
import { useNetworkStatus } from '@/hooks/useNetworkStatus';

describe('NetworkStatusBanner', () => {
  beforeEach(() => {
    vi.mocked(useNetworkStatus).mockReturnValue({
      isOnline: true,
      isApiReachable: true,
      isSlowConnection: false,
      checkApiConnectivity: vi.fn().mockResolvedValue(true),
    });
  });

  it('should not render when online and API reachable', () => {
    const { container } = render(<NetworkStatusBanner />);

    expect(container.firstChild).toBeNull();
  });

  it('should render offline message when offline', () => {
    vi.mocked(useNetworkStatus).mockReturnValue({
      isOnline: false,
      isApiReachable: false,
      isSlowConnection: false,
      checkApiConnectivity: vi.fn(),
    });

    render(<NetworkStatusBanner />);

    expect(screen.getByText(/you are offline/i)).toBeInTheDocument();
  });

  it('should render API down message when API unreachable', () => {
    vi.mocked(useNetworkStatus).mockReturnValue({
      isOnline: true,
      isApiReachable: false,
      isSlowConnection: false,
      checkApiConnectivity: vi.fn(),
    });

    render(<NetworkStatusBanner />);

    expect(screen.getByText(/cannot connect to server/i)).toBeInTheDocument();
  });

  it('should render slow connection message', () => {
    vi.mocked(useNetworkStatus).mockReturnValue({
      isOnline: true,
      isApiReachable: true,
      isSlowConnection: true,
      checkApiConnectivity: vi.fn(),
    });

    render(<NetworkStatusBanner />);

    expect(screen.getByText(/slow connection/i)).toBeInTheDocument();
  });

  it('should have role alert', () => {
    vi.mocked(useNetworkStatus).mockReturnValue({
      isOnline: false,
      isApiReachable: false,
      isSlowConnection: false,
      checkApiConnectivity: vi.fn(),
    });

    render(<NetworkStatusBanner />);

    expect(screen.getByRole('alert')).toBeInTheDocument();
  });

  it('should apply custom className', () => {
    vi.mocked(useNetworkStatus).mockReturnValue({
      isOnline: false,
      isApiReachable: false,
      isSlowConnection: false,
      checkApiConnectivity: vi.fn(),
    });

    render(<NetworkStatusBanner className="custom-banner" />);

    const banner = screen.getByRole('alert');
    expect(banner).toHaveClass('custom-banner');
  });
});
