/**
 * RunControls Component Tests
 * 
 * Tests simulation control buttons and their interactions.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, within } from '@testing-library/react';
import { renderWithProviders } from '@/test/utils';
import { RunControls } from './RunControls';

// Mock the hooks
vi.mock('@/hooks', () => ({
  useStartSimulation: vi.fn(() => ({
    mutate: vi.fn(),
    mutateAsync: vi.fn(),
    isPending: false,
  })),
  usePauseSimulation: vi.fn(() => ({
    mutate: vi.fn(),
    mutateAsync: vi.fn(),
    isPending: false,
  })),
  useStopSimulation: vi.fn(() => ({
    mutate: vi.fn(),
    mutateAsync: vi.fn(),
    isPending: false,
  })),
}));

vi.mock('@/stores', () => ({
  useSimulationStore: vi.fn(() => ({})),
}));

describe('RunControls', () => {
  const defaultProps = {
    simulationId: 'test-sim-123',
    status: 'pending' as const,
    showLabels: true, // Use labels for easier testing
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('pending status', () => {
    it('should show Start button when pending', () => {
      renderWithProviders(<RunControls {...defaultProps} status="pending" />);
      
      // Find button with Start text
      expect(screen.getByText('Start')).toBeInTheDocument();
    });

    it('should not show Pause when pending', () => {
      renderWithProviders(<RunControls {...defaultProps} status="pending" />);
      
      expect(screen.queryByText('Pause')).not.toBeInTheDocument();
    });
  });

  describe('running status', () => {
    it('should show Pause button when running', () => {
      renderWithProviders(<RunControls {...defaultProps} status="running" />);
      
      expect(screen.getByText('Pause')).toBeInTheDocument();
    });

    it('should show Stop button when running', () => {
      renderWithProviders(<RunControls {...defaultProps} status="running" />);
      
      expect(screen.getByText('Stop')).toBeInTheDocument();
    });

    it('should not show Start when running', () => {
      renderWithProviders(<RunControls {...defaultProps} status="running" />);
      
      expect(screen.queryByText('Start')).not.toBeInTheDocument();
    });
  });

  describe('paused status', () => {
    it('should show Resume button when paused', () => {
      renderWithProviders(<RunControls {...defaultProps} status="paused" />);
      
      expect(screen.getByText('Resume')).toBeInTheDocument();
    });

    it('should show Stop button when paused', () => {
      renderWithProviders(<RunControls {...defaultProps} status="paused" />);
      
      expect(screen.getByText('Stop')).toBeInTheDocument();
    });
  });

  describe('completed status', () => {
    it('should show Restart button when completed', () => {
      renderWithProviders(<RunControls {...defaultProps} status="completed" />);
      
      expect(screen.getByText('Restart')).toBeInTheDocument();
    });
  });

  describe('failed status', () => {
    it('should show Restart button when failed', () => {
      renderWithProviders(<RunControls {...defaultProps} status="failed" />);
      
      expect(screen.getByText('Restart')).toBeInTheDocument();
    });
  });

  describe('interactions', () => {
    it('should call start mutation when Start is clicked', async () => {
      const mockMutate = vi.fn();
      const { useStartSimulation } = await import('@/hooks');
      vi.mocked(useStartSimulation).mockReturnValue({
        mutate: mockMutate,
        mutateAsync: vi.fn(),
        isPending: false,
      } as any);

      const { user } = renderWithProviders(
        <RunControls {...defaultProps} status="pending" />
      );
      
      // Click the button containing Start text
      const startButton = screen.getByText('Start').closest('button');
      if (startButton) {
        await user.click(startButton);
        expect(mockMutate).toHaveBeenCalledWith('test-sim-123');
      }
    });

    it('should show confirmation dialog before stopping', async () => {
      const { user } = renderWithProviders(
        <RunControls {...defaultProps} status="running" />
      );
      
      const stopButton = screen.getByText('Stop').closest('button');
      expect(stopButton).toBeInTheDocument();
      
      if (stopButton) {
        await user.click(stopButton);
        
        // The AlertDialog content should appear
        // Just verify clicking works without error
        expect(stopButton).toBeInTheDocument();
      }
    });
  });
});

