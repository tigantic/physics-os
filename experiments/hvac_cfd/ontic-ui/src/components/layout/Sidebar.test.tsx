/**
 * Sidebar Component Tests
 * 
 * Tests for the Sidebar navigation component
 * Constitutional Compliance: Article III Testing Protocols
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TooltipProvider } from '@radix-ui/react-tooltip';

// Mock Next.js
vi.mock('next/link', () => ({
  default: ({ children, href, className }: { children: React.ReactNode; href: string; className?: string }) => (
    <a href={href} className={className}>{children}</a>
  ),
}));

vi.mock('next/navigation', () => ({
  usePathname: vi.fn(() => '/dashboard'),
}));

// Mock Zustand stores
vi.mock('@/stores', () => ({
  useUIStore: vi.fn(() => ({
    sidebarCollapsed: false,
    setSidebarCollapsed: vi.fn(),
  })),
}));

import { Sidebar } from './Sidebar';

describe('Sidebar', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const renderSidebar = () => {
    return render(
      <TooltipProvider>
        <Sidebar />
      </TooltipProvider>
    );
  };

  describe('rendering', () => {
    it('should render the sidebar', () => {
      renderSidebar();
      
      const sidebar = screen.getByRole('complementary', { name: /main navigation/i });
      expect(sidebar).toBeInTheDocument();
    });

    it('should render navigation sections', () => {
      renderSidebar();
      
      // Check for section headings
      expect(screen.getByText('Overview')).toBeInTheDocument();
      expect(screen.getByText('Simulation')).toBeInTheDocument();
      expect(screen.getByText('Advanced')).toBeInTheDocument();
    });

    it('should render Dashboard link', () => {
      renderSidebar();
      
      const dashboardLink = screen.getByRole('link', { name: /dashboard/i });
      expect(dashboardLink).toBeInTheDocument();
      expect(dashboardLink).toHaveAttribute('href', '/dashboard');
    });

    it('should render Meshes link', () => {
      renderSidebar();
      
      const meshesLink = screen.getByRole('link', { name: /meshes/i });
      expect(meshesLink).toBeInTheDocument();
      expect(meshesLink).toHaveAttribute('href', '/meshes');
    });

    it('should render Simulations link', () => {
      renderSidebar();
      
      const simulationsLink = screen.getByRole('link', { name: /simulations/i });
      expect(simulationsLink).toBeInTheDocument();
      expect(simulationsLink).toHaveAttribute('href', '/simulations');
    });

    it('should render Results link', () => {
      renderSidebar();
      
      const resultsLink = screen.getByRole('link', { name: /results/i });
      expect(resultsLink).toBeInTheDocument();
      expect(resultsLink).toHaveAttribute('href', '/results');
    });
  });

  describe('navigation items', () => {
    it('should have System Status link', () => {
      renderSidebar();
      
      const statusLink = screen.getByRole('link', { name: /system status/i });
      expect(statusLink).toBeInTheDocument();
      expect(statusLink).toHaveAttribute('href', '/status');
    });
  });

  describe('collapse toggle', () => {
    it('should have collapse button', () => {
      renderSidebar();
      
      const collapseButton = screen.getByRole('button', { name: /collapse/i });
      expect(collapseButton).toBeInTheDocument();
    });

    it('should toggle sidebar width on collapse click', () => {
      renderSidebar();
      
      const sidebar = screen.getByRole('complementary', { name: /main navigation/i });
      const collapseButton = screen.getByRole('button', { name: /collapse/i });
      
      // Initial state should be expanded
      expect(sidebar).toHaveClass('w-64');
      
      // Click to collapse
      fireEvent.click(collapseButton);
      
      // Should toggle
      expect(sidebar).toHaveClass('w-16');
    });
  });

  describe('branding', () => {
    it('should display HyperTensor branding', () => {
      renderSidebar();
      
      expect(screen.getByText(/hypertensor/i)).toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('should have proper aria-label', () => {
      renderSidebar();
      
      const sidebar = screen.getByRole('complementary', { name: /main navigation/i });
      expect(sidebar).toBeInTheDocument();
    });

    it('should render dashboard link properly', () => {
      renderSidebar();
      
      // Dashboard is the current pathname - verify it's highlighted
      const dashboardLink = screen.getByRole('link', { name: /dashboard/i });
      expect(dashboardLink).toBeInTheDocument();
      expect(dashboardLink).toHaveAttribute('href', '/dashboard');
    });
  });

  describe('custom className', () => {
    it('should apply custom className', () => {
      render(
        <TooltipProvider>
          <Sidebar className="custom-sidebar" />
        </TooltipProvider>
      );
      
      const sidebar = screen.getByRole('complementary', { name: /main navigation/i });
      expect(sidebar).toHaveClass('custom-sidebar');
    });
  });
});
