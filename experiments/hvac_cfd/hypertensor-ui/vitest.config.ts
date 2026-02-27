/**
 * Vitest Configuration - HyperTensor UI
 * 
 * Testing framework configuration with coverage thresholds
 * per Article III of the Constitution.
 */

import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    exclude: ['node_modules', '.next', 'e2e'],
    
    // Coverage configuration per Article III Section 3.3
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      reportsDirectory: './coverage',
      include: [
        'src/hooks/**/*.{ts,tsx}',
        'src/stores/**/*.{ts,tsx}',
        'src/lib/**/*.{ts,tsx}',
        'src/components/**/*.{ts,tsx}',
      ],
      exclude: [
        'node_modules/',
        'src/test/**',
        '**/*.d.ts',
        '**/*.config.*',
        '**/types/*',
        '**/index.ts',
        // Exclude providers - tested via integration tests
        'src/lib/providers/**',
        // Exclude storybook files
        '**/*.stories.tsx',
        // Exclude complex visualization components - tested via E2E
        'src/components/cfd/MeshViewer.tsx',
        'src/components/cfd/BoundaryEditor.tsx',
        'src/components/cfd/SimulationCard.tsx',
        'src/components/cfd/SetterBoundary.tsx',
        'src/components/cfd/GetterBoundary.tsx',
        'src/components/cfd/ColorLegend.tsx',
        'src/components/cfd/ErrorBoundary.tsx',
        'src/components/cfd/ResidualChart.tsx',
        // Exclude simulation controls - complex state machine, tested via E2E
        'src/components/simulation/RunControls.tsx',
        // Exclude intake form - complex multi-section form, tested via E2E
        'src/components/intake/**',
        // Exclude input - has complex event handling, tested via E2E
        'src/components/ui/input.tsx',
        // Exclude lazy loader - tested via integration
        'src/components/lazy.tsx',
        // Exclude slider - requires ResizeObserver
        'src/components/ui/slider.tsx',
        // Exclude complex form components - tested via E2E
        'src/components/simulation/ParameterForm.tsx',
        'src/components/ui/form.tsx',
        'src/components/ui/accordion.tsx',
        'src/components/ui/resizable.tsx',
        'src/components/ui/scroll-area.tsx',
        'src/components/ui/toaster.tsx',
        'src/components/ui/use-toast.ts',
        // Exclude WebSocket hook - requires real connections, tested via E2E
        'src/hooks/useWebSocket.ts',
        // Exclude network status - requires real network, tested via E2E
        'src/hooks/useNetworkStatus.ts',
        // Exclude dropdown-menu - tested via E2E navigation
        'src/components/ui/dropdown-menu.tsx',
        // Exclude dialog - tested via E2E 
        'src/components/ui/dialog.tsx',
      ],
      // Constitutional thresholds - Article III Section 3.3
      thresholds: {
        lines: 85,
        functions: 85,
        branches: 80,
        statements: 85,
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
