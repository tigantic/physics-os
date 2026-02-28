/**
 * Test Utilities - Render Helpers and Mock Providers
 * 
 * Provides wrapped render function with all required providers
 * for component testing per Article III of the Constitution.
 */

import React, { type ReactElement, type ReactNode } from 'react';
import { render, type RenderOptions, type RenderResult } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from 'next-themes';
import userEvent from '@testing-library/user-event';

// ============================================
// MOCK QUERY CLIENT
// ============================================

/**
 * Create a fresh QueryClient for testing
 * Disables retries and caching for predictable tests
 */
export function createTestQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
        staleTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });
}

// ============================================
// TEST PROVIDERS WRAPPER
// ============================================

interface AllTheProvidersProps {
  children: ReactNode;
  queryClient?: QueryClient;
}

function AllTheProviders({ children, queryClient }: AllTheProvidersProps) {
  const client = queryClient ?? createTestQueryClient();

  return (
    <QueryClientProvider client={client}>
      <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false}>
        {children}
      </ThemeProvider>
    </QueryClientProvider>
  );
}

// ============================================
// CUSTOM RENDER
// ============================================

interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  queryClient?: QueryClient;
}

/**
 * Custom render function that wraps component with all providers
 * 
 * @param ui - The React element to render
 * @param options - Custom render options including queryClient
 * @returns Render result with user event helper
 * 
 * @example
 * ```tsx
 * const { getByText, user } = renderWithProviders(<MyComponent />);
 * await user.click(getByText('Submit'));
 * ```
 */
export function renderWithProviders(
  ui: ReactElement,
  options: CustomRenderOptions = {}
): RenderResult & { user: ReturnType<typeof userEvent.setup> } {
  const { queryClient, ...renderOptions } = options;
  
  const user = userEvent.setup();
  
  const Wrapper = ({ children }: { children: React.ReactNode }) => (
    <AllTheProviders queryClient={queryClient}>{children}</AllTheProviders>
  );

  return {
    user,
    ...render(ui, { wrapper: Wrapper, ...renderOptions }),
  };
}

// Legacy export for backwards compatibility
const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) => render(ui, { wrapper: AllTheProviders, ...options });

// ============================================
// MOCK DATA FACTORIES
// ============================================

/**
 * Create mock simulation data for testing
 */
export function createMockSimulation(overrides: Record<string, unknown> = {}) {
  return {
    id: 'sim-test-001',
    name: 'Test Simulation',
    description: 'A test simulation for unit tests',
    meshId: 'mesh-test-001',
    status: 'pending',
    solverType: 'steady',
    turbulenceModel: 'k-epsilon',
    iteration: 0,
    max_iterations: 1000,
    current_time: 0,
    end_time: 1.0,
    cfl_number: 0.9,
    convergence_tolerance: 1e-6,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    ...overrides,
  };
}

/**
 * Create mock simulation summary for list views
 */
export function createMockSimulationSummary(overrides: Record<string, unknown> = {}) {
  return {
    id: 'sim-test-001',
    name: 'Test Simulation',
    status: 'pending',
    solverType: 'steady',
    meshId: 'mesh-test-001',
    iteration: 0,
    max_iterations: 1000,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    ...overrides,
  };
}

/**
 * Create mock residual data point
 */
export function createMockResidual(iteration: number, overrides: Record<string, unknown> = {}) {
  return {
    iteration,
    time: iteration * 0.001,
    continuity: Math.pow(10, -3 - iteration * 0.005),
    momentum_x: Math.pow(10, -3 - iteration * 0.004),
    momentum_y: Math.pow(10, -3 - iteration * 0.0045),
    momentum_z: Math.pow(10, -3 - iteration * 0.0042),
    energy: Math.pow(10, -4 - iteration * 0.005),
    ...overrides,
  };
}

/**
 * Create array of mock residuals
 */
export function createMockResidualHistory(count: number) {
  return Array.from({ length: count }, (_, i) => createMockResidual(i));
}

/**
 * Create mock mesh data
 */
export function createMockMesh(overrides: Record<string, unknown> = {}) {
  return {
    id: 'mesh-test-001',
    name: 'Test Mesh',
    description: 'A test mesh for unit tests',
    cell_count: 50000,
    node_count: 51000,
    face_count: 100000,
    domain_min: [0, 0, 0],
    domain_max: [1, 1, 1],
    boundary_patches: [
      {
        id: 'patch-inlet',
        name: 'inlet',
        patch_type: 'velocity_inlet',
        face_count: 100,
        face_direction: 'x_min',
        color: '#00ff00',
      },
      {
        id: 'patch-outlet',
        name: 'outlet',
        patch_type: 'pressure_outlet',
        face_count: 100,
        face_direction: 'x_max',
        color: '#ff0000',
      },
    ],
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    ...overrides,
  };
}

/**
 * Create mock GPU info
 */
export function createMockGPU(overrides: Record<string, unknown> = {}) {
  return {
    id: 0,
    name: 'NVIDIA RTX 4090',
    memory_total: 24 * 1024 * 1024 * 1024,
    memory_used: 8 * 1024 * 1024 * 1024,
    memory_free: 16 * 1024 * 1024 * 1024,
    utilization: 45,
    temperature: 65,
    power_draw: 350,
    compute_capability: '8.9',
    ...overrides,
  };
}

/**
 * Create mock system status
 */
export function createMockSystemStatus(overrides: Record<string, unknown> = {}) {
  return {
    status: 'healthy',
    version: '0.1.0',
    gpu_count: 1,
    gpus: [createMockGPU()],
    active_simulations: 0,
    queued_simulations: 0,
    ...overrides,
  };
}

// ============================================
// WAIT UTILITIES
// ============================================

/**
 * Wait for a condition to be true
 */
export async function waitForCondition(
  condition: () => boolean,
  timeout = 1000,
  interval = 50
): Promise<void> {
  const start = Date.now();
  while (!condition()) {
    if (Date.now() - start > timeout) {
      throw new Error('Timeout waiting for condition');
    }
    await new Promise((resolve) => setTimeout(resolve, interval));
  }
}

// ============================================
// EXPORTS
// ============================================

// Re-export testing utilities (excluding render which we override)
export { 
  screen, 
  waitFor, 
  within, 
  fireEvent,
  act,
  cleanup,
  prettyDOM,
} from '@testing-library/react';
export type { RenderResult, RenderOptions } from '@testing-library/react';

// Export our custom render functions (renderWithProviders already exported at definition)
export { customRender as render, userEvent };
