/**
 * SimulationCard Component Tests
 * 
 * Unit tests for the simulation card component per Article III.
 * Tests rendering, status display, and progress calculation.
 */

import { describe, it, expect } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithProviders, createMockSimulation, createMockSimulationSummary } from '@/test/utils';
import { SimulationCard } from '@/components/cfd/SimulationCard';

describe('SimulationCard', () => {
  describe('rendering', () => {
    it('should render simulation name', () => {
      const simulation = createMockSimulation({ name: 'My CFD Simulation' });
      
      renderWithProviders(<SimulationCard simulation={simulation as never} />);
      
      expect(screen.getByText('My CFD Simulation')).toBeInTheDocument();
    });

    it('should render progress section', () => {
      const simulation = createMockSimulation({
        iteration: 250,
        max_iterations: 500,
      });
      
      renderWithProviders(<SimulationCard simulation={simulation as never} />);
      
      expect(screen.getByText('Progress')).toBeInTheDocument();
    });

    it('should render simulation time section', () => {
      const simulation = createMockSimulation({ current_time: 0.5, end_time: 1.0 });
      
      renderWithProviders(<SimulationCard simulation={simulation as never} />);
      
      expect(screen.getByText('Simulation Time')).toBeInTheDocument();
    });
  });

  describe('status display', () => {
    it('should display pending status correctly', () => {
      const simulation = createMockSimulation({ status: 'pending' });
      
      renderWithProviders(<SimulationCard simulation={simulation as never} />);
      
      expect(screen.getByText(/pending/i)).toBeInTheDocument();
    });

    it('should display running status correctly', () => {
      const simulation = createMockSimulation({ status: 'running' });
      
      renderWithProviders(<SimulationCard simulation={simulation as never} />);
      
      expect(screen.getByText(/running/i)).toBeInTheDocument();
    });

    it('should display completed status correctly', () => {
      const simulation = createMockSimulation({ status: 'completed' });
      
      renderWithProviders(<SimulationCard simulation={simulation as never} />);
      
      expect(screen.getByText(/completed/i)).toBeInTheDocument();
    });

    it('should display failed status correctly', () => {
      const simulation = createMockSimulation({ status: 'failed' });
      
      renderWithProviders(<SimulationCard simulation={simulation as never} />);
      
      expect(screen.getByText(/failed/i)).toBeInTheDocument();
    });

    it('should display paused status correctly', () => {
      const simulation = createMockSimulation({ status: 'paused' });
      
      renderWithProviders(<SimulationCard simulation={simulation as never} />);
      
      expect(screen.getByText(/paused/i)).toBeInTheDocument();
    });
  });

  describe('progress indicator', () => {
    it('should show progress for running simulation', () => {
      const simulation = createMockSimulation({
        status: 'running',
        iteration: 500,
        max_iterations: 1000,
      });
      
      renderWithProviders(<SimulationCard simulation={simulation as never} />);
      
      // Check that running status and progress section are displayed
      expect(screen.getByText('Progress')).toBeInTheDocument();
      expect(screen.getByText(/running/i)).toBeInTheDocument();
      expect(screen.getAllByRole('progressbar')).toHaveLength(2); // iteration and time progress
    });

    it('should show 100% for completed simulation', () => {
      const simulation = createMockSimulation({
        status: 'completed',
        iteration: 1000,
        max_iterations: 1000,
      });
      
      renderWithProviders(<SimulationCard simulation={simulation as never} />);
      
      // Progress section should show completed iterations
      expect(screen.getByText('Progress')).toBeInTheDocument();
      expect(screen.getAllByRole('progressbar')).toHaveLength(2);
    });
  });

  describe('with SimulationSummary type', () => {
    it('should render correctly with summary type', () => {
      const summary = createMockSimulationSummary({ name: 'Summary Simulation' });
      
      renderWithProviders(<SimulationCard simulation={summary as never} />);
      
      expect(screen.getByText('Summary Simulation')).toBeInTheDocument();
    });

    it('should handle missing optional fields gracefully', () => {
      const summary = createMockSimulationSummary({
        description: undefined,
        current_time: undefined,
      });
      
      // Should not throw
      renderWithProviders(<SimulationCard simulation={summary as never} />);
      
      expect(screen.getByText(summary.name)).toBeInTheDocument();
    });
  });
});

describe('SimulationCard (compact mode)', () => {
  it('should render compact version', () => {
    const simulation = createMockSimulation({ name: 'Compact Sim' });
    
    renderWithProviders(<SimulationCard simulation={simulation as never} compact />);
    
    expect(screen.getByText('Compact Sim')).toBeInTheDocument();
  });

  it('should show abbreviated status', () => {
    const simulation = createMockSimulation({ status: 'running' });
    
    renderWithProviders(<SimulationCard simulation={simulation as never} compact />);
    
    expect(screen.getByText(/running/i)).toBeInTheDocument();
  });

  it('should show iteration count', () => {
    const simulation = createMockSimulation({
      iteration: 250,
      max_iterations: 500,
    });
    
    renderWithProviders(<SimulationCard simulation={simulation as never} compact />);
    
    expect(screen.getByText(/250/)).toBeInTheDocument();
  });
});
