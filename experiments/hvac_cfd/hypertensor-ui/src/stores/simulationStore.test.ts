/**
 * Simulation Store Tests
 * 
 * Unit tests for Zustand simulation store per Article III.
 * Tests all state mutations and selectors.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act } from '@testing-library/react';
import { useSimulationStore, selectActiveSimulation, selectResiduals, selectIsRunning, selectProgress } from '@/stores/simulationStore';
import { createMockSimulation, createMockResidual } from '@/test/utils';

describe('simulationStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    act(() => {
      useSimulationStore.getState().reset();
    });
  });

  describe('setActiveSimulation', () => {
    it('should set active simulation and clear residuals', () => {
      const mockSim = createMockSimulation();
      
      act(() => {
        useSimulationStore.getState().setActiveSimulation(mockSim as never);
      });

      const state = useSimulationStore.getState();
      expect(state.activeSimulation).toEqual(mockSim);
      expect(state.residuals).toHaveLength(0);
      expect(state.performance).toBeNull();
    });

    it('should clear active simulation when set to null', () => {
      const mockSim = createMockSimulation();
      
      act(() => {
        useSimulationStore.getState().setActiveSimulation(mockSim as never);
        useSimulationStore.getState().setActiveSimulation(null);
      });

      expect(useSimulationStore.getState().activeSimulation).toBeNull();
    });
  });

  describe('updateStatus', () => {
    it('should update status of active simulation', () => {
      const mockSim = createMockSimulation({ status: 'created' });
      
      act(() => {
        useSimulationStore.getState().setActiveSimulation(mockSim as never);
        useSimulationStore.getState().updateStatus('running', 50, 0.5);
      });

      const state = useSimulationStore.getState();
      expect(state.activeSimulation?.status).toBe('running');
      expect(state.activeSimulation?.iteration).toBe(50);
      expect(state.activeSimulation?.current_time).toBe(0.5);
      expect(state.lastUpdate).toBeInstanceOf(Date);
    });

    it('should not update if no active simulation', () => {
      act(() => {
        useSimulationStore.getState().updateStatus('running');
      });

      expect(useSimulationStore.getState().activeSimulation).toBeNull();
    });
  });

  describe('addResidual', () => {
    it('should add residual point to array', () => {
      const residual = createMockResidual(1);
      
      act(() => {
        useSimulationStore.getState().addResidual(residual as never);
      });

      expect(useSimulationStore.getState().residuals).toHaveLength(1);
      expect(useSimulationStore.getState().residuals[0]).toEqual(residual);
    });

    it('should keep only last 1000 residuals (rolling buffer)', () => {
      act(() => {
        // Add 1100 residuals
        for (let i = 0; i < 1100; i++) {
          useSimulationStore.getState().addResidual(createMockResidual(i) as never);
        }
      });

      const residuals = useSimulationStore.getState().residuals;
      expect(residuals).toHaveLength(1000);
      expect(residuals[0]!.iteration).toBe(100); // First 100 should be dropped
      expect(residuals[999]!.iteration).toBe(1099);
    });
  });

  describe('setResiduals', () => {
    it('should replace entire residuals array', () => {
      const residuals = [createMockResidual(1), createMockResidual(2)];
      
      act(() => {
        useSimulationStore.getState().setResiduals(residuals as never);
      });

      expect(useSimulationStore.getState().residuals).toHaveLength(2);
    });
  });

  describe('updatePerformance', () => {
    it('should update performance metrics', () => {
      const metrics = {
        gpu_utilization: 95,
        memory_used: 8 * 1024 * 1024 * 1024,
        iterations_per_second: 150,
        cells_per_second: 7.5e9,
      };
      
      act(() => {
        useSimulationStore.getState().updatePerformance(metrics as never);
      });

      expect(useSimulationStore.getState().performance).toEqual(metrics);
      expect(useSimulationStore.getState().lastUpdate).toBeInstanceOf(Date);
    });
  });

  describe('setConnected', () => {
    it('should update connection status', () => {
      expect(useSimulationStore.getState().isConnected).toBe(false);
      
      act(() => {
        useSimulationStore.getState().setConnected(true);
      });

      expect(useSimulationStore.getState().isConnected).toBe(true);
    });
  });

  describe('clearResiduals', () => {
    it('should clear residuals array', () => {
      act(() => {
        useSimulationStore.getState().addResidual(createMockResidual(1) as never);
        useSimulationStore.getState().addResidual(createMockResidual(2) as never);
        useSimulationStore.getState().clearResiduals();
      });

      expect(useSimulationStore.getState().residuals).toHaveLength(0);
    });
  });

  describe('reset', () => {
    it('should reset to initial state', () => {
      const mockSim = createMockSimulation();
      
      act(() => {
        useSimulationStore.getState().setActiveSimulation(mockSim as never);
        useSimulationStore.getState().addResidual(createMockResidual(1) as never);
        useSimulationStore.getState().setConnected(true);
        useSimulationStore.getState().reset();
      });

      const state = useSimulationStore.getState();
      expect(state.activeSimulation).toBeNull();
      expect(state.residuals).toHaveLength(0);
      expect(state.performance).toBeNull();
      expect(state.isConnected).toBe(false);
      expect(state.lastUpdate).toBeNull();
    });
  });

  describe('selectors', () => {
    it('selectActiveSimulation should return active simulation', () => {
      const mockSim = createMockSimulation();
      
      act(() => {
        useSimulationStore.getState().setActiveSimulation(mockSim as never);
      });

      expect(selectActiveSimulation(useSimulationStore.getState())).toEqual(mockSim);
    });

    it('selectResiduals should return residuals array', () => {
      const residuals = [createMockResidual(1), createMockResidual(2)];
      
      act(() => {
        useSimulationStore.getState().setResiduals(residuals as never);
      });

      expect(selectResiduals(useSimulationStore.getState())).toHaveLength(2);
    });

    it('selectIsRunning should return true when status is running', () => {
      const mockSim = createMockSimulation({ status: 'running' });
      
      act(() => {
        useSimulationStore.getState().setActiveSimulation(mockSim as never);
      });

      expect(selectIsRunning(useSimulationStore.getState())).toBe(true);
    });

    it('selectIsRunning should return false when not running', () => {
      const mockSim = createMockSimulation({ status: 'created' });
      
      act(() => {
        useSimulationStore.getState().setActiveSimulation(mockSim as never);
      });

      expect(selectIsRunning(useSimulationStore.getState())).toBe(false);
    });

    it('selectProgress should calculate correct percentage', () => {
      const mockSim = createMockSimulation({ iteration: 500, max_iterations: 1000 });
      
      act(() => {
        useSimulationStore.getState().setActiveSimulation(mockSim as never);
      });

      expect(selectProgress(useSimulationStore.getState())).toBe(50);
    });

    it('selectProgress should cap at 100%', () => {
      const mockSim = createMockSimulation({ iteration: 1500, max_iterations: 1000 });
      
      act(() => {
        useSimulationStore.getState().setActiveSimulation(mockSim as never);
      });

      expect(selectProgress(useSimulationStore.getState())).toBe(100);
    });

    it('selectProgress should return 0 when no active simulation', () => {
      expect(selectProgress(useSimulationStore.getState())).toBe(0);
    });
  });
});
