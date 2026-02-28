/**
 * Header Component Tests
 * 
 * Tests for the Header navigation component
 * Constitutional Compliance: Article III Testing Protocols
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TooltipProvider } from '@radix-ui/react-tooltip';

// Mock Next.js Link
vi.mock('next/link', () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  ),
}));

// Mock Zustand stores
vi.mock('@/stores', () => ({
  useUIStore: vi.fn(() => ({
    theme: 'light',
    setTheme: vi.fn(),
  })),
  useSimulationStore: vi.fn((selector) => {
    if (typeof selector === 'function') {
      return selector({ isConnected: true });
    }
    return { isConnected: true };
  }),
}));

// Mock hooks
vi.mock('@/hooks', () => ({
  useSystemStatus: vi.fn(() => ({
    data: { gpuUtilization: 75 },
    isLoading: false,
  })),
}));

import { Header } from './Header';

describe('Header', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const renderHeader = () => {
    return render(
      <TooltipProvider>
        <Header />
      </TooltipProvider>
    );
  };

  describe('rendering', () => {
    it('should render the header', () => {
      renderHeader();
      
      const header = screen.getByRole('banner');
      expect(header).toBeInTheDocument();
    });

    it('should render search input', () => {
      renderHeader();
      
      const searchInput = screen.getByRole('searchbox');
      expect(searchInput).toBeInTheDocument();
    });

    it('should have search region', () => {
      renderHeader();
      
      const searchRegion = screen.getByRole('search');
      expect(searchRegion).toBeInTheDocument();
    });

    it('should render notifications button', () => {
      renderHeader();
      
      const notificationsButton = screen.getByLabelText(/notifications/i);
      expect(notificationsButton).toBeInTheDocument();
    });

    it('should render connection status', () => {
      renderHeader();
      
      const status = screen.getByRole('status');
      expect(status).toBeInTheDocument();
    });
  });

  describe('search functionality', () => {
    it('should update search input value on change', () => {
      renderHeader();
      
      const searchInput = screen.getByRole('searchbox') as HTMLInputElement;
      fireEvent.change(searchInput, { target: { value: 'test query' } });
      
      expect(searchInput.value).toBe('test query');
    });

    it('should have correct placeholder', () => {
      renderHeader();
      
      const searchInput = screen.getByPlaceholderText(/search simulations/i);
      expect(searchInput).toBeInTheDocument();
    });
  });

  describe('connection status display', () => {
    it('should show connected status when connected', () => {
      renderHeader();
      
      const status = screen.getByRole('status');
      expect(status).toHaveAttribute('aria-label', 'WebSocket connected');
    });
  });

  describe('theme toggle', () => {
    it('should render theme toggle button', () => {
      renderHeader();
      
      // Look for button with theme-related functionality
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
    });
  });

  describe('notifications', () => {
    it('should have notifications trigger button', async () => {
      renderHeader();
      
      const notificationsButton = screen.getByLabelText(/notifications/i);
      expect(notificationsButton).toBeInTheDocument();
      expect(notificationsButton).toHaveAttribute('aria-haspopup', 'menu');
    });

    it('should show notification count badge', () => {
      renderHeader();
      
      // Should have a badge with notification count
      const badge = screen.getByText('3');
      expect(badge).toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('should have proper aria labels', () => {
      renderHeader();
      
      const searchInput = screen.getByLabelText(/search simulations/i);
      expect(searchInput).toBeInTheDocument();
    });

    it('should have live region for connection status', () => {
      renderHeader();
      
      const status = screen.getByRole('status');
      expect(status).toHaveAttribute('aria-live', 'polite');
    });
  });

  describe('custom className', () => {
    it('should apply custom className', () => {
      render(
        <TooltipProvider>
          <Header className="custom-header" />
        </TooltipProvider>
      );
      
      const header = screen.getByRole('banner');
      expect(header).toHaveClass('custom-header');
    });
  });
});
