/**
 * Simulation Store - Zustand State Management
 * 
 * Manages active simulation state, real-time updates, and solver control.
 * Integrates with WebSocket for live residual and performance streaming.
 */

import { create } from 'zustand';
import { devtools, subscribeWithSelector } from 'zustand/middleware';
import type { 
  Simulation, 
  SimulationStatus, 
  ResidualPoint, 
  PerformanceMetrics 
} from '@/types';
import { RESIDUAL_BUFFER_SLICE_OFFSET } from '@/lib/constants';

// ============================================
// STATE INTERFACE
// ============================================

interface SimulationState {
  // Active simulation
  activeSimulation: Simulation | null;
  
  // Real-time data
  residuals: ResidualPoint[];
  performance: PerformanceMetrics | null;
  
  // Connection state
  isConnected: boolean;
  lastUpdate: Date | null;
  
  // Actions
  setActiveSimulation: (sim: Simulation | null) => void;
  updateStatus: (status: SimulationStatus, iteration?: number, time?: number) => void;
  addResidual: (point: ResidualPoint) => void;
  setResiduals: (residuals: ResidualPoint[]) => void;
  updatePerformance: (perf: PerformanceMetrics) => void;
  setConnected: (connected: boolean) => void;
  clearResiduals: () => void;
  reset: () => void;
}

// ============================================
// INITIAL STATE
// ============================================

const initialState = {
  activeSimulation: null,
  residuals: [],
  performance: null,
  isConnected: false,
  lastUpdate: null,
};

// ============================================
// STORE IMPLEMENTATION
// ============================================

export const useSimulationStore = create<SimulationState>()(
  devtools(
    subscribeWithSelector((set, get) => ({
      ...initialState,

      setActiveSimulation: (sim) => 
        set(
          { activeSimulation: sim, residuals: [], performance: null },
          false,
          'setActiveSimulation'
        ),

      updateStatus: (status, iteration, time) =>
        set(
          (state) => {
            if (!state.activeSimulation) return state;
            return {
              activeSimulation: {
                ...state.activeSimulation,
                status,
                iteration: iteration ?? state.activeSimulation.iteration,
                current_time: time ?? state.activeSimulation.current_time,
              },
              lastUpdate: new Date(),
            };
          },
          false,
          'updateStatus'
        ),

      addResidual: (point) =>
        set(
          (state) => ({
            residuals: [...state.residuals.slice(RESIDUAL_BUFFER_SLICE_OFFSET), point],
            lastUpdate: new Date(),
          }),
          false,
          'addResidual'
        ),

      setResiduals: (residuals) =>
        set({ residuals }, false, 'setResiduals'),

      updatePerformance: (performance) =>
        set({ performance, lastUpdate: new Date() }, false, 'updatePerformance'),

      setConnected: (isConnected) =>
        set({ isConnected }, false, 'setConnected'),

      clearResiduals: () =>
        set({ residuals: [] }, false, 'clearResiduals'),

      reset: () =>
        set(initialState, false, 'reset'),
    })),
    { name: 'SimulationStore' }
  )
);

// ============================================
// SELECTORS
// ============================================

export const selectActiveSimulation = (state: SimulationState) => state.activeSimulation;
export const selectResiduals = (state: SimulationState) => state.residuals;
export const selectPerformance = (state: SimulationState) => state.performance;
export const selectIsRunning = (state: SimulationState) => 
  state.activeSimulation?.status === 'running';
export const selectProgress = (state: SimulationState) => {
  const sim = state.activeSimulation;
  if (!sim) return 0;
  return Math.min(100, (sim.iteration / sim.max_iterations) * 100);
};
