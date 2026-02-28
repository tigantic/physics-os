/**
 * CFD Components Tests
 * 
 * Tests for ResidualChart, ColorLegend, MeshViewer, BoundaryEditor
 * Constitutional Compliance: Article III Testing Protocols
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ResidualChart, ResidualChartMini } from './ResidualChart';
import { ColorLegend } from './ColorLegend';

// Mock the stores
vi.mock('@/stores', () => ({
  useSimulationStore: vi.fn((selector) => {
    if (selector) return [];
    return { residuals: [] };
  }),
  selectResiduals: (state: any) => state?.residuals || [],
}));

// Mock recharts
vi.mock('recharts', () => ({
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  Legend: () => null,
  ResponsiveContainer: ({ children }: any) => <div>{children}</div>,
  ReferenceLine: () => null,
}));

// ============================================
// RESIDUAL CHART TESTS
// ============================================

describe('ResidualChart', () => {
  const mockData = [
    { iteration: 1, continuity: 1e-2, momentum_x: 1e-2, momentum_y: 1e-2, momentum_z: 1e-2 },
    { iteration: 2, continuity: 1e-3, momentum_x: 1e-3, momentum_y: 1e-3, momentum_z: 1e-3 },
    { iteration: 3, continuity: 1e-4, momentum_x: 1e-4, momentum_y: 1e-4, momentum_z: 1e-4 },
  ];

  it('should render chart with data', () => {
    render(<ResidualChart data={mockData} />);
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
  });

  it('should show no data message when empty', () => {
    render(<ResidualChart data={[]} />);
    expect(screen.getByText(/no residual data/i)).toBeInTheDocument();
  });

  it('should render card wrapper by default', () => {
    render(<ResidualChart data={mockData} />);
    expect(screen.getByText(/convergence history/i)).toBeInTheDocument();
  });

  it('should hide card when showCard is false', () => {
    render(<ResidualChart data={mockData} showCard={false} />);
    expect(screen.queryByText(/convergence history/i)).not.toBeInTheDocument();
  });
});

describe('ResidualChartMini', () => {
  const mockData = [
    { iteration: 1, continuity: 1e-2 },
    { iteration: 2, continuity: 1e-3 },
  ];

  it('should render mini chart', () => {
    render(<ResidualChartMini data={mockData} />);
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
  });

  it('should show no data message when empty', () => {
    render(<ResidualChartMini data={[]} />);
    // Message is "No residual data available"
    expect(screen.getByText(/no residual data/i)).toBeInTheDocument();
  });
});

// ============================================
// COLOR LEGEND TESTS
// ============================================

describe('ColorLegend', () => {
  it('should render legend with label', () => {
    render(<ColorLegend min={0} max={100} label="Temperature" />);
    expect(screen.getByText('Temperature')).toBeInTheDocument();
  });

  it('should display formatted min value', () => {
    render(<ColorLegend min={0} max={100} label="Velocity" />);
    // formatValue(0) = "0.00"
    expect(screen.getByText('0.00')).toBeInTheDocument();
  });

  it('should display formatted max value', () => {
    render(<ColorLegend min={0} max={100} label="Pressure" />);
    // formatValue(100) = "100.00"
    expect(screen.getByText('100.00')).toBeInTheDocument();
  });

  it('should display unit when provided', () => {
    render(<ColorLegend min={0} max={100} label="Temp" unit="K" />);
    expect(screen.getByText(/\(K\)/)).toBeInTheDocument();
  });
});
