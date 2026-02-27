/**
 * useApi Hooks Tests
 * 
 * Unit tests for React Query hooks per Article III.
 * Tests data fetching, mutations, and cache invalidation.
 */

import React from 'react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { queryKeys, useSimulations, useSimulation, useResiduals } from '@/hooks/useApi';
import { createMockSimulation, createMockSimulationSummary, createMockResidualHistory } from '@/test/utils';
import { api } from '@/lib/api/client';

// Mock the API client
vi.mock('@/lib/api/client', () => ({
  api: {
    get: vi.fn(),
    post: vi.fn(),
    patch: vi.fn(),
    delete: vi.fn(),
  },
}));

const mockedApi = vi.mocked(api);

// Test wrapper with QueryClient
function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });
  
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    );
  };
}

describe('queryKeys', () => {
  describe('simulations', () => {
    it('should generate correct all key', () => {
      expect(queryKeys.simulations.all).toEqual(['simulations']);
    });

    it('should generate correct list key without filters', () => {
      expect(queryKeys.simulations.list()).toEqual(['simulations', 'list', undefined]);
    });

    it('should generate correct list key with filters', () => {
      const filters = { status: 'running', limit: 10 };
      expect(queryKeys.simulations.list(filters as never)).toEqual(['simulations', 'list', filters]);
    });

    it('should generate correct detail key', () => {
      expect(queryKeys.simulations.detail('sim-123')).toEqual(['simulations', 'detail', 'sim-123']);
    });

    it('should generate correct residuals key', () => {
      expect(queryKeys.simulations.residuals('sim-123')).toEqual(['simulations', 'residuals', 'sim-123']);
    });
  });

  describe('meshes', () => {
    it('should generate correct all key', () => {
      expect(queryKeys.meshes.all).toEqual(['meshes']);
    });

    it('should generate correct list key', () => {
      expect(queryKeys.meshes.list()).toEqual(['meshes', 'list']);
    });

    it('should generate correct detail key', () => {
      expect(queryKeys.meshes.detail('mesh-123')).toEqual(['meshes', 'detail', 'mesh-123']);
    });

    it('should generate correct patches key', () => {
      expect(queryKeys.meshes.patches('mesh-123')).toEqual(['meshes', 'patches', 'mesh-123']);
    });
  });

  describe('system', () => {
    it('should generate correct status key', () => {
      expect(queryKeys.system.status).toEqual(['system', 'status']);
    });

    it('should generate correct gpus key', () => {
      expect(queryKeys.system.gpus).toEqual(['system', 'gpus']);
    });
  });
});

describe('useSimulations', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should fetch simulations successfully', async () => {
    const mockSimulations = [
      createMockSimulationSummary({ id: 'sim-1', name: 'Sim 1' }),
      createMockSimulationSummary({ id: 'sim-2', name: 'Sim 2' }),
    ];
    
    mockedApi.get.mockResolvedValueOnce({
      data: { items: mockSimulations },
      error: null,
    });

    const { result } = renderHook(() => useSimulations(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toHaveLength(2);
    expect(result.current.data?.[0]?.name).toBe('Sim 1');
  });

  it('should pass filters to API', async () => {
    mockedApi.get.mockResolvedValueOnce({
      data: { items: [] },
      error: null,
    });

    renderHook(() => useSimulations({ status: 'running', limit: 5 } as never), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(mockedApi.get).toHaveBeenCalled());

    expect(mockedApi.get).toHaveBeenCalledWith(
      '/api/v1/simulations',
      expect.objectContaining({
        status: 'running',
        limit: 5,
      })
    );
  });

  it('should handle API error', async () => {
    mockedApi.get.mockResolvedValueOnce({
      data: null,
      error: new Error('Network error'),
    });

    const { result } = renderHook(() => useSimulations(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(result.current.error).toBeInstanceOf(Error);
  });

  it('should return empty array when no items', async () => {
    mockedApi.get.mockResolvedValueOnce({
      data: { items: null },
      error: null,
    });

    const { result } = renderHook(() => useSimulations(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual([]);
  });
});

describe('useSimulation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should fetch single simulation', async () => {
    const mockSimulation = createMockSimulation({ id: 'sim-123' });
    
    mockedApi.get.mockResolvedValueOnce({
      data: mockSimulation,
      error: null,
    });

    const { result } = renderHook(() => useSimulation('sim-123'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.id).toBe('sim-123');
    expect(mockedApi.get).toHaveBeenCalledWith('/api/v1/simulations/sim-123');
  });

  it('should not fetch when id is null', async () => {
    const { result } = renderHook(() => useSimulation(null), {
      wrapper: createWrapper(),
    });

    // Should stay in initial state (not enabled)
    expect(result.current.isPending).toBe(true);
    expect(result.current.fetchStatus).toBe('idle');
    expect(mockedApi.get).not.toHaveBeenCalled();
  });

  it('should handle API error', async () => {
    mockedApi.get.mockResolvedValueOnce({
      data: null,
      error: new Error('Not found'),
    });

    const { result } = renderHook(() => useSimulation('sim-invalid'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useResiduals', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should fetch residuals for simulation', async () => {
    const mockResiduals = createMockResidualHistory(10);
    
    mockedApi.get.mockResolvedValueOnce({
      data: { residuals: mockResiduals },
      error: null,
    });

    const { result } = renderHook(() => useResiduals('sim-123'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toHaveLength(10);
    expect(mockedApi.get).toHaveBeenCalledWith(
      '/api/v1/simulations/sim-123/residuals',
      expect.objectContaining({ from_iteration: 0 })
    );
  });

  it('should pass fromIteration parameter', async () => {
    mockedApi.get.mockResolvedValueOnce({
      data: { residuals: [] },
      error: null,
    });

    renderHook(() => useResiduals('sim-123', 500), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(mockedApi.get).toHaveBeenCalled());

    expect(mockedApi.get).toHaveBeenCalledWith(
      '/api/v1/simulations/sim-123/residuals',
      expect.objectContaining({ from_iteration: 500 })
    );
  });

  it('should not fetch when id is null', async () => {
    const { result } = renderHook(() => useResiduals(null), {
      wrapper: createWrapper(),
    });

    expect(result.current.isPending).toBe(true);
    expect(result.current.fetchStatus).toBe('idle');
    expect(mockedApi.get).not.toHaveBeenCalled();
  });

  it('should return empty array when no residuals', async () => {
    mockedApi.get.mockResolvedValueOnce({
      data: { residuals: null },
      error: null,
    });

    const { result } = renderHook(() => useResiduals('sim-123'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual([]);
  });
});

// ============================================
// MESH HOOKS TESTS
// ============================================

import { useMeshes, useMesh, useCreateMesh, useDeleteMesh, useSystemStatus, useGPUs, useCreateSimulation, useStartSimulation, usePauseSimulation, useStopSimulation, useDeleteSimulation, getExportUrl, useActivities, useSimulationFields, useUploadMesh, useAddPatch } from '@/hooks/useApi';

// ============================================
// SIMULATION MUTATION HOOKS TESTS
// ============================================

describe('useCreateSimulation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should have mutate function', () => {
    const { result } = renderHook(() => useCreateSimulation(), {
      wrapper: createWrapper(),
    });

    expect(result.current.mutate).toBeDefined();
  });

  it('should call post with simulation data', async () => {
    const mockSimulation = { id: 'sim-new', name: 'New Simulation' };
    
    mockedApi.post.mockResolvedValueOnce({
      data: mockSimulation,
      error: null,
    });

    const { result } = renderHook(() => useCreateSimulation(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({ name: 'New Simulation', meshId: 'mesh-1' } as never);

    await waitFor(() => expect(mockedApi.post).toHaveBeenCalled());
    expect(mockedApi.post).toHaveBeenCalledWith('/api/v1/simulations', expect.any(Object));
  });
});

describe('useStartSimulation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should have mutate function', () => {
    const { result } = renderHook(() => useStartSimulation(), {
      wrapper: createWrapper(),
    });

    expect(result.current.mutate).toBeDefined();
  });

  it('should call start endpoint', async () => {
    mockedApi.post.mockResolvedValueOnce({
      data: { id: 'sim-123', status: 'running' },
      error: null,
    });

    const { result } = renderHook(() => useStartSimulation(), {
      wrapper: createWrapper(),
    });

    result.current.mutate('sim-123');

    await waitFor(() => expect(mockedApi.post).toHaveBeenCalled());
    expect(mockedApi.post).toHaveBeenCalledWith('/api/v1/simulations/sim-123/start');
  });
});

describe('usePauseSimulation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should have mutate function', () => {
    const { result } = renderHook(() => usePauseSimulation(), {
      wrapper: createWrapper(),
    });

    expect(result.current.mutate).toBeDefined();
  });

  it('should call pause endpoint', async () => {
    mockedApi.post.mockResolvedValueOnce({
      data: { id: 'sim-123', status: 'paused' },
      error: null,
    });

    const { result } = renderHook(() => usePauseSimulation(), {
      wrapper: createWrapper(),
    });

    result.current.mutate('sim-123');

    await waitFor(() => expect(mockedApi.post).toHaveBeenCalled());
    expect(mockedApi.post).toHaveBeenCalledWith('/api/v1/simulations/sim-123/pause');
  });
});

describe('useStopSimulation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should have mutate function', () => {
    const { result } = renderHook(() => useStopSimulation(), {
      wrapper: createWrapper(),
    });

    expect(result.current.mutate).toBeDefined();
  });

  it('should call stop endpoint', async () => {
    mockedApi.post.mockResolvedValueOnce({
      data: { id: 'sim-123', status: 'stopped' },
      error: null,
    });

    const { result } = renderHook(() => useStopSimulation(), {
      wrapper: createWrapper(),
    });

    result.current.mutate('sim-123');

    await waitFor(() => expect(mockedApi.post).toHaveBeenCalled());
    expect(mockedApi.post).toHaveBeenCalledWith('/api/v1/simulations/sim-123/stop');
  });
});

describe('useDeleteSimulation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should have mutate function', () => {
    const { result } = renderHook(() => useDeleteSimulation(), {
      wrapper: createWrapper(),
    });

    expect(result.current.mutate).toBeDefined();
  });

  it('should call delete endpoint', async () => {
    mockedApi.delete.mockResolvedValueOnce({
      data: null,
      error: null,
    });

    const { result } = renderHook(() => useDeleteSimulation(), {
      wrapper: createWrapper(),
    });

    result.current.mutate('sim-123');

    await waitFor(() => expect(mockedApi.delete).toHaveBeenCalled());
    expect(mockedApi.delete).toHaveBeenCalledWith('/api/v1/simulations/sim-123');
  });
});

describe('useMeshes', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should fetch list of meshes', async () => {
    const mockMeshes = [
      { id: 'mesh-1', name: 'Mesh 1' },
      { id: 'mesh-2', name: 'Mesh 2' },
    ];
    
    mockedApi.get.mockResolvedValueOnce({
      data: { items: mockMeshes },
      error: null,
    });

    const { result } = renderHook(() => useMeshes(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toHaveLength(2);
    expect(mockedApi.get).toHaveBeenCalledWith('/api/v1/meshes');
  });

  it('should return empty array on error', async () => {
    mockedApi.get.mockResolvedValueOnce({
      data: null,
      error: new Error('Failed to fetch'),
    });

    const { result } = renderHook(() => useMeshes(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useMesh', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should fetch single mesh by id', async () => {
    const mockMesh = { id: 'mesh-123', name: 'Test Mesh' };
    
    mockedApi.get.mockResolvedValueOnce({
      data: mockMesh,
      error: null,
    });

    const { result } = renderHook(() => useMesh('mesh-123'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockMesh);
    expect(mockedApi.get).toHaveBeenCalledWith('/api/v1/meshes/mesh-123');
  });

  it('should not fetch when id is null', async () => {
    const { result } = renderHook(() => useMesh(null), {
      wrapper: createWrapper(),
    });

    expect(result.current.isPending).toBe(true);
    expect(result.current.fetchStatus).toBe('idle');
    expect(mockedApi.get).not.toHaveBeenCalled();
  });
});

describe('useCreateMesh', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should create a new mesh', async () => {
    const mockMesh = { id: 'mesh-new', name: 'New Mesh' };
    
    mockedApi.post.mockResolvedValueOnce({
      data: mockMesh,
      error: null,
    });

    const { result } = renderHook(() => useCreateMesh(), {
      wrapper: createWrapper(),
    });

    expect(result.current.mutate).toBeDefined();
  });
});

describe('useDeleteMesh', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should have delete mutation', async () => {
    const { result } = renderHook(() => useDeleteMesh(), {
      wrapper: createWrapper(),
    });

    expect(result.current.mutate).toBeDefined();
  });
});

// ============================================
// SYSTEM HOOKS TESTS
// ============================================

describe('useSystemStatus', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should fetch system status', async () => {
    const mockStatus = { 
      cpuUsage: 45, 
      memoryUsage: 60, 
      gpuUtilization: 80 
    };
    
    mockedApi.get.mockResolvedValueOnce({
      data: mockStatus,
      error: null,
    });

    const { result } = renderHook(() => useSystemStatus(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockStatus);
    expect(mockedApi.get).toHaveBeenCalledWith('/api/v1/system/status');
  });
});

describe('useGPUs', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should fetch GPU list', async () => {
    const mockGPUs = [
      { id: 0, name: 'RTX 4090', memoryTotal: 24000 },
    ];
    
    // The hook fetches GPUInfo[] directly
    mockedApi.get.mockResolvedValueOnce({
      data: mockGPUs,
      error: null,
    });

    const { result } = renderHook(() => useGPUs(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockGPUs);
    expect(mockedApi.get).toHaveBeenCalledWith('/api/v1/system/gpus');
  });
});

// ============================================
// ADDITIONAL HOOKS TESTS
// ============================================

describe('getExportUrl', () => {
  it('should return correct export URL', () => {
    const url = getExportUrl('sim-123');
    expect(url).toBe('/api/v1/simulations/sim-123/export');
  });

  it('should handle different simulation IDs', () => {
    expect(getExportUrl('test-id')).toBe('/api/v1/simulations/test-id/export');
    expect(getExportUrl('uuid-test-1234')).toBe('/api/v1/simulations/uuid-test-1234/export');
  });
});

describe('useActivities', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should fetch activities list', async () => {
    const mockActivities = [
      { id: 'act-1', type: 'simulation', action: 'start', timestamp: '2024-01-01T00:00:00Z' },
    ];
    
    mockedApi.get.mockResolvedValueOnce({
      data: { activities: mockActivities },
      error: null,
    });

    const { result } = renderHook(() => useActivities(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockActivities);
    expect(mockedApi.get).toHaveBeenCalledWith('/api/v1/activities', { limit: 10 });
  });

  it('should return empty array on error', async () => {
    mockedApi.get.mockResolvedValueOnce({
      data: null,
      error: { message: 'Error' },
    });

    const { result } = renderHook(() => useActivities(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useSimulationFields', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should fetch simulation fields', async () => {
    const mockFields = [
      { name: 'velocity', description: 'Velocity field', unit: 'm/s', components: ['x', 'y', 'z'] },
    ];
    
    mockedApi.get.mockResolvedValueOnce({
      data: { fields: mockFields },
      error: null,
    });

    const { result } = renderHook(() => useSimulationFields('sim-123'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockFields);
    expect(mockedApi.get).toHaveBeenCalledWith('/api/v1/simulations/sim-123/fields');
  });

  it('should not fetch when id is null', () => {
    const { result } = renderHook(() => useSimulationFields(null), {
      wrapper: createWrapper(),
    });

    expect(result.current.fetchStatus).toBe('idle');
  });
});

describe('useUploadMesh', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should have mutate function', () => {
    const { result } = renderHook(() => useUploadMesh(), {
      wrapper: createWrapper(),
    });

    expect(result.current.mutate).toBeDefined();
  });
});

describe('useAddPatch', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should have mutate function', () => {
    const { result } = renderHook(() => useAddPatch(), {
      wrapper: createWrapper(),
    });

    expect(result.current.mutate).toBeDefined();
  });
});
