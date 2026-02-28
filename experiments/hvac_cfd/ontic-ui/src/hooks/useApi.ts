/**
 * HyperTensor Custom Hooks - CFD Operations
 * 
 * React Query hooks for data fetching and mutation.
 */

'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api/client';
import type {
  Simulation,
  SimulationCreate,
  SimulationSummary,
  SimulationFilters,
  SimulationListResponse,
  ResidualPoint,
  Mesh,
  MeshCreate,
  MeshSummary,
  BoundaryPatch,
  SystemStatus,
  GPUInfo,
} from '@/types';

// ============================================
// QUERY KEYS
// ============================================

export const queryKeys = {
  simulations: {
    all: ['simulations'] as const,
    list: (filters?: SimulationFilters) => [...queryKeys.simulations.all, 'list', filters] as const,
    detail: (id: string) => [...queryKeys.simulations.all, 'detail', id] as const,
    residuals: (id: string) => [...queryKeys.simulations.all, 'residuals', id] as const,
  },
  meshes: {
    all: ['meshes'] as const,
    list: () => [...queryKeys.meshes.all, 'list'] as const,
    detail: (id: string) => [...queryKeys.meshes.all, 'detail', id] as const,
    patches: (id: string) => [...queryKeys.meshes.all, 'patches', id] as const,
  },
  system: {
    status: ['system', 'status'] as const,
    gpus: ['system', 'gpus'] as const,
  },
} as const;

// ============================================
// SIMULATION HOOKS
// ============================================

/**
 * Fetch list of simulations with optional filters
 */
export function useSimulations(filters?: SimulationFilters) {
  return useQuery({
    queryKey: queryKeys.simulations.list(filters),
    queryFn: async () => {
      const { data, error } = await api.get<SimulationListResponse>('/api/v1/simulations', {
        status: filters?.status,
        limit: filters?.limit,
        offset: filters?.offset,
      });
      if (error) throw error;
      return data?.items ?? [];
    },
  });
}

/**
 * Fetch single simulation details
 */
export function useSimulation(id: string | null) {
  return useQuery({
    queryKey: queryKeys.simulations.detail(id ?? ''),
    queryFn: async () => {
      if (!id) throw new Error('No simulation ID');
      const { data, error } = await api.get<Simulation>(`/api/v1/simulations/${id}`);
      if (error) throw error;
      return data;
    },
    enabled: !!id,
  });
}

/**
 * Fetch residual history for a simulation
 */
export function useResiduals(id: string | null, fromIteration = 0) {
  return useQuery({
    queryKey: queryKeys.simulations.residuals(id ?? ''),
    queryFn: async () => {
      if (!id) throw new Error('No simulation ID');
      const { data, error } = await api.get<{ residuals: ResidualPoint[] }>(`/api/v1/simulations/${id}/residuals`, {
        from_iteration: fromIteration,
      });
      if (error) throw error;
      return data?.residuals ?? [];
    },
    enabled: !!id,
    refetchInterval: 2000, // Poll every 2 seconds for live updates
  });
}

/**
 * Create a new simulation
 */
export function useCreateSimulation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (newSimulation: SimulationCreate) => {
      const { data, error } = await api.post<Simulation>('/api/v1/simulations', newSimulation);
      if (error) throw error;
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.simulations.all });
    },
  });
}

/**
 * Start or resume a simulation
 */
export function useStartSimulation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (id: string) => {
      const { data, error } = await api.post<Simulation>(`/api/v1/simulations/${id}/start`);
      if (error) throw error;
      return data;
    },
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.simulations.detail(id) });
      queryClient.invalidateQueries({ queryKey: queryKeys.simulations.all });
    },
  });
}

/**
 * Pause a running simulation
 */
export function usePauseSimulation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (id: string) => {
      const { data, error } = await api.post<Simulation>(`/api/v1/simulations/${id}/pause`);
      if (error) throw error;
      return data;
    },
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.simulations.detail(id) });
      queryClient.invalidateQueries({ queryKey: queryKeys.simulations.all });
    },
  });
}

/**
 * Stop a simulation
 */
export function useStopSimulation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (id: string) => {
      const { data, error } = await api.post<Simulation>(`/api/v1/simulations/${id}/stop`);
      if (error) throw error;
      return data;
    },
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.simulations.detail(id) });
      queryClient.invalidateQueries({ queryKey: queryKeys.simulations.all });
    },
  });
}

/**
 * Delete a simulation
 */
export function useDeleteSimulation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (id: string) => {
      const { error } = await api.delete<void>(`/api/v1/simulations/${id}`);
      if (error) throw error;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.simulations.all });
    },
  });
}

// ============================================
// MESH HOOKS
// ============================================

interface MeshListResponse {
  items: MeshSummary[];
  total: number;
}

/**
 * Fetch list of meshes
 */
export function useMeshes() {
  return useQuery({
    queryKey: queryKeys.meshes.list(),
    queryFn: async () => {
      const { data, error } = await api.get<MeshListResponse>('/api/v1/meshes');
      if (error) throw error;
      return data?.items ?? [];
    },
  });
}

/**
 * Fetch single mesh details
 */
export function useMesh(id: string | null) {
  return useQuery({
    queryKey: queryKeys.meshes.detail(id ?? ''),
    queryFn: async () => {
      if (!id) throw new Error('No mesh ID');
      const { data, error } = await api.get<Mesh>(`/api/v1/meshes/${id}`);
      if (error) throw error;
      return data;
    },
    enabled: !!id,
  });
}

/**
 * Create a new mesh
 */
export function useCreateMesh() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (newMesh: MeshCreate) => {
      const { data, error } = await api.post<Mesh>('/api/v1/meshes', newMesh);
      if (error) throw error;
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.meshes.all });
    },
  });
}

/**
 * Upload a mesh file
 */
export function useUploadMesh() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('/api/v1/meshes/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Upload failed');
      }
      
      return response.json() as Promise<Mesh>;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.meshes.all });
    },
  });
}

/**
 * Delete a mesh
 */
export function useDeleteMesh() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (id: string) => {
      const { error } = await api.delete<void>(`/api/v1/meshes/${id}`);
      if (error) throw error;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.meshes.all });
    },
  });
}

/**
 * Add a boundary patch to a mesh
 */
export function useAddPatch() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ meshId, patch }: { meshId: string; patch: Omit<BoundaryPatch, 'id'> }) => {
      const { data, error } = await api.post<BoundaryPatch>(`/api/v1/meshes/${meshId}/patches`, patch);
      if (error) throw error;
      return data;
    },
    onSuccess: (_, { meshId }) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.meshes.detail(meshId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.meshes.patches(meshId) });
    },
  });
}

// ============================================
// SYSTEM HOOKS
// ============================================

/**
 * Fetch system status
 */
export function useSystemStatus() {
  return useQuery({
    queryKey: queryKeys.system.status,
    queryFn: async () => {
      const { data, error } = await api.get<SystemStatus>('/api/v1/system/status');
      if (error) throw error;
      return data;
    },
    refetchInterval: 5000, // Poll every 5 seconds
  });
}

/**
 * Fetch GPU information
 */
export function useGPUs() {
  return useQuery({
    queryKey: queryKeys.system.gpus,
    queryFn: async () => {
      const { data, error } = await api.get<GPUInfo[]>('/api/v1/system/gpus');
      if (error) throw error;
      return data ?? [];
    },
    refetchInterval: 10000, // Poll every 10 seconds
  });
}

// ============================================
// ACTIVITY HOOKS
// ============================================

export interface Activity {
  id: string;
  type: 'simulation_completed' | 'simulation_started' | 'simulation_failed' | 'mesh_imported';
  title: string;
  description: string;
  timestamp: string;
  icon: string;
}

/**
 * Fetch recent activities
 */
export function useActivities(limit = 10) {
  return useQuery({
    queryKey: ['activities', limit],
    queryFn: async () => {
      const { data, error } = await api.get<{ activities: Activity[] }>('/api/v1/activities', { limit });
      if (error) throw error;
      return data?.activities ?? [];
    },
    refetchInterval: 10000, // Poll every 10 seconds
  });
}

// ============================================
// EXPORT HOOKS
// ============================================

/**
 * Get export URL for a simulation
 */
export function getExportUrl(simulationId: string): string {
  return `/api/v1/simulations/${simulationId}/export`;
}

/**
 * Fetch available fields for a simulation
 */
export function useSimulationFields(id: string | null) {
  return useQuery({
    queryKey: ['simulation-fields', id],
    queryFn: async () => {
      if (!id) throw new Error('No simulation ID');
      const { data, error } = await api.get<{ fields: Array<{
        name: string;
        description: string;
        unit: string;
        components: string[] | null;
      }> }>(`/api/v1/simulations/${id}/fields`);
      if (error) throw error;
      return data?.fields ?? [];
    },
    enabled: !!id,
  });
}
