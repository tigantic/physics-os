/**
 * Viewer Store - 3D Visualization State
 * 
 * Manages camera, field selection, colormap, and visualization options
 * for the Three.js mesh viewer and post-processing views.
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { 
  FieldName, 
  ColormapName, 
  SlicePlane,
  VisualizationOptions,
  DEFAULT_VISUALIZATION_OPTIONS 
} from '@/types';

// ============================================
// STATE INTERFACE
// ============================================

interface ViewerState {
  // Field selection
  activeField: FieldName;
  
  // Colormap settings
  colormap: ColormapName;
  colorRangeMode: 'auto' | 'manual';
  colorRangeMin: number;
  colorRangeMax: number;
  
  // Slice settings
  slicePlane: SlicePlane | null;
  slicePosition: number;
  
  // Display options
  opacity: number;
  showMesh: boolean;
  showBoundaries: boolean;
  showStreamlines: boolean;
  streamlineCount: number;
  showAxes: boolean;
  showGrid: boolean;
  
  // Camera
  cameraPosition: [number, number, number];
  cameraTarget: [number, number, number];
  
  // Actions
  setActiveField: (field: FieldName) => void;
  setColormap: (colormap: ColormapName) => void;
  setColorRange: (mode: 'auto' | 'manual', min?: number, max?: number) => void;
  setSlice: (plane: SlicePlane | null, position?: number) => void;
  setOpacity: (opacity: number) => void;
  toggleMesh: () => void;
  toggleBoundaries: () => void;
  toggleStreamlines: () => void;
  setStreamlineCount: (count: number) => void;
  toggleAxes: () => void;
  toggleGrid: () => void;
  setCameraPosition: (pos: [number, number, number]) => void;
  setCameraTarget: (target: [number, number, number]) => void;
  resetCamera: () => void;
  resetAll: () => void;
}

// ============================================
// INITIAL STATE
// ============================================

const initialState = {
  activeField: 'temperature' as FieldName,
  colormap: 'viridis' as ColormapName,
  colorRangeMode: 'auto' as const,
  colorRangeMin: 0,
  colorRangeMax: 1,
  slicePlane: null,
  slicePosition: 0.5,
  opacity: 1.0,
  showMesh: false,
  showBoundaries: true,
  showStreamlines: false,
  streamlineCount: 100,
  showAxes: true,
  showGrid: true,
  cameraPosition: [2, 2, 2] as [number, number, number],
  cameraTarget: [0, 0, 0] as [number, number, number],
};

// ============================================
// STORE IMPLEMENTATION
// ============================================

export const useViewerStore = create<ViewerState>()(
  devtools(
    persist(
      (set) => ({
        ...initialState,

        setActiveField: (activeField) =>
          set({ activeField }, false, 'setActiveField'),

        setColormap: (colormap) =>
          set({ colormap }, false, 'setColormap'),

        setColorRange: (mode, min, max) =>
          set(
            {
              colorRangeMode: mode,
              colorRangeMin: min ?? 0,
              colorRangeMax: max ?? 1,
            },
            false,
            'setColorRange'
          ),

        setSlice: (plane, position) =>
          set(
            {
              slicePlane: plane,
              slicePosition: position ?? 0.5,
            },
            false,
            'setSlice'
          ),

        setOpacity: (opacity) =>
          set({ opacity: Math.max(0, Math.min(1, opacity)) }, false, 'setOpacity'),

        toggleMesh: () =>
          set((state) => ({ showMesh: !state.showMesh }), false, 'toggleMesh'),

        toggleBoundaries: () =>
          set((state) => ({ showBoundaries: !state.showBoundaries }), false, 'toggleBoundaries'),

        toggleStreamlines: () =>
          set((state) => ({ showStreamlines: !state.showStreamlines }), false, 'toggleStreamlines'),

        setStreamlineCount: (streamlineCount) =>
          set({ streamlineCount }, false, 'setStreamlineCount'),

        toggleAxes: () =>
          set((state) => ({ showAxes: !state.showAxes }), false, 'toggleAxes'),

        toggleGrid: () =>
          set((state) => ({ showGrid: !state.showGrid }), false, 'toggleGrid'),

        setCameraPosition: (cameraPosition) =>
          set({ cameraPosition }, false, 'setCameraPosition'),

        setCameraTarget: (cameraTarget) =>
          set({ cameraTarget }, false, 'setCameraTarget'),

        resetCamera: () =>
          set(
            {
              cameraPosition: initialState.cameraPosition,
              cameraTarget: initialState.cameraTarget,
            },
            false,
            'resetCamera'
          ),

        resetAll: () =>
          set(initialState, false, 'resetAll'),
      }),
      {
        name: 'viewer-settings',
        partialize: (state) => ({
          colormap: state.colormap,
          showAxes: state.showAxes,
          showGrid: state.showGrid,
          showBoundaries: state.showBoundaries,
        }),
      }
    ),
    { name: 'ViewerStore' }
  )
);

// ============================================
// SELECTORS
// ============================================

export const selectVisualizationOptions = (state: ViewerState): VisualizationOptions => ({
  colormap: state.colormap,
  colorRange: {
    mode: state.colorRangeMode,
    min: state.colorRangeMin,
    max: state.colorRangeMax,
  },
  opacity: state.opacity,
  showMesh: state.showMesh,
  showBoundaries: state.showBoundaries,
  showStreamlines: state.showStreamlines,
  streamlineCount: state.streamlineCount,
});
