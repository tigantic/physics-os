/**
 * Viewer Store Tests
 * 
 * Tests for viewerStore (Zustand)
 * Constitutional Compliance: Article III Testing Protocols
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useViewerStore } from './viewerStore';

describe('viewerStore', () => {
  beforeEach(() => {
    // Reset store between tests
    useViewerStore.getState().resetAll();
  });

  describe('initial state', () => {
    it('should have camera position as tuple', () => {
      const { result } = renderHook(() => useViewerStore());
      expect(result.current.cameraPosition).toBeDefined();
      expect(Array.isArray(result.current.cameraPosition)).toBe(true);
      expect(result.current.cameraPosition).toHaveLength(3);
    });

    it('should have camera target as tuple', () => {
      const { result } = renderHook(() => useViewerStore());
      expect(result.current.cameraTarget).toBeDefined();
      expect(Array.isArray(result.current.cameraTarget)).toBe(true);
    });

    it('should have axes setting', () => {
      const { result } = renderHook(() => useViewerStore());
      expect(typeof result.current.showAxes).toBe('boolean');
    });

    it('should have grid setting', () => {
      const { result } = renderHook(() => useViewerStore());
      expect(typeof result.current.showGrid).toBe('boolean');
    });

    it('should have colormap', () => {
      const { result } = renderHook(() => useViewerStore());
      expect(result.current.colormap).toBeDefined();
    });
  });

  describe('camera controls', () => {
    it('should update camera position', () => {
      const { result } = renderHook(() => useViewerStore());
      
      act(() => {
        result.current.setCameraPosition([10, 20, 30]);
      });
      
      expect(result.current.cameraPosition).toEqual([10, 20, 30]);
    });

    it('should update camera target', () => {
      const { result } = renderHook(() => useViewerStore());
      
      act(() => {
        result.current.setCameraTarget([1, 2, 3]);
      });
      
      expect(result.current.cameraTarget).toEqual([1, 2, 3]);
    });

    it('should reset camera', () => {
      const { result } = renderHook(() => useViewerStore());
      
      // Change camera position
      act(() => {
        result.current.setCameraPosition([100, 100, 100]);
      });
      
      // Reset
      act(() => {
        result.current.resetCamera();
      });
      
      // Should be back to default [2, 2, 2]
      expect(result.current.cameraPosition).toEqual([2, 2, 2]);
    });
  });

  describe('visualization options', () => {
    it('should toggle axes visibility', () => {
      const { result } = renderHook(() => useViewerStore());
      
      const initial = result.current.showAxes;
      
      act(() => {
        result.current.toggleAxes();
      });
      
      expect(result.current.showAxes).toBe(!initial);
    });

    it('should toggle grid visibility', () => {
      const { result } = renderHook(() => useViewerStore());
      
      const initial = result.current.showGrid;
      
      act(() => {
        result.current.toggleGrid();
      });
      
      expect(result.current.showGrid).toBe(!initial);
    });

    it('should toggle mesh visibility', () => {
      const { result } = renderHook(() => useViewerStore());
      
      const initial = result.current.showMesh;
      
      act(() => {
        result.current.toggleMesh();
      });
      
      expect(result.current.showMesh).toBe(!initial);
    });

    it('should toggle boundaries visibility', () => {
      const { result } = renderHook(() => useViewerStore());
      
      const initial = result.current.showBoundaries;
      
      act(() => {
        result.current.toggleBoundaries();
      });
      
      expect(result.current.showBoundaries).toBe(!initial);
    });

    it('should set colormap', () => {
      const { result } = renderHook(() => useViewerStore());
      
      act(() => {
        result.current.setColormap('plasma');
      });
      
      expect(result.current.colormap).toBe('plasma');
    });

    it('should set opacity', () => {
      const { result } = renderHook(() => useViewerStore());
      
      act(() => {
        result.current.setOpacity(0.5);
      });
      
      expect(result.current.opacity).toBe(0.5);
    });
  });

  describe('field selection', () => {
    it('should set active field', () => {
      const { result } = renderHook(() => useViewerStore());
      
      act(() => {
        result.current.setActiveField('pressure');
      });
      
      expect(result.current.activeField).toBe('pressure');
    });
  });

  describe('slice settings', () => {
    it('should set slice plane', () => {
      const { result } = renderHook(() => useViewerStore());
      
      act(() => {
        result.current.setSlice('x', 0.5);
      });
      
      expect(result.current.slicePlane).toBe('x');
      expect(result.current.slicePosition).toBe(0.5);
    });

    it('should clear slice plane', () => {
      const { result } = renderHook(() => useViewerStore());
      
      act(() => {
        result.current.setSlice('y', 0.3);
      });
      
      act(() => {
        result.current.setSlice(null);
      });
      
      expect(result.current.slicePlane).toBeNull();
    });
  });

  describe('color range', () => {
    it('should set color range to auto', () => {
      const { result } = renderHook(() => useViewerStore());
      
      act(() => {
        result.current.setColorRange('auto');
      });
      
      expect(result.current.colorRangeMode).toBe('auto');
    });

    it('should set color range to manual with values', () => {
      const { result } = renderHook(() => useViewerStore());
      
      act(() => {
        result.current.setColorRange('manual', 0, 100);
      });
      
      expect(result.current.colorRangeMode).toBe('manual');
      expect(result.current.colorRangeMin).toBe(0);
      expect(result.current.colorRangeMax).toBe(100);
    });
  });

  describe('reset', () => {
    it('should reset all settings', () => {
      const { result } = renderHook(() => useViewerStore());
      
      // Make changes
      act(() => {
        result.current.setCameraPosition([100, 100, 100]);
        result.current.setColormap('plasma');
        result.current.setOpacity(0.3);
      });
      
      // Reset
      act(() => {
        result.current.resetAll();
      });
      
      // Check defaults restored
      expect(result.current.cameraPosition).toEqual([2, 2, 2]);
      expect(result.current.colormap).toBe('viridis');
      expect(result.current.opacity).toBe(1.0);
    });
  });
});
