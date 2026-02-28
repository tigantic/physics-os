/**
 * ResidualChart - Convergence History Visualization
 * 
 * Real-time log-scale plot of solver residuals using Recharts.
 * Displays continuity, momentum, energy, and turbulence residuals.
 */

'use client';

import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { useSimulationStore, selectResiduals } from '@/stores';
import type { ResidualPoint } from '@/types';

// ============================================
// PROPS INTERFACE
// ============================================

interface ResidualChartProps {
  data?: ResidualPoint[];
  convergenceTolerance?: number;
  height?: number;
  showCard?: boolean;
  className?: string;
}

// ============================================
// CHART CONFIGURATION
// ============================================

const RESIDUAL_COLORS = {
  continuity: '#ef4444',    // Red
  momentum_x: '#22c55e',    // Green
  momentum_y: '#3b82f6',    // Blue
  momentum_z: '#8b5cf6',    // Purple
  energy: '#f59e0b',        // Amber
  turbulent_ke: '#06b6d4',  // Cyan
  turbulent_omega: '#ec4899', // Pink
} as const;

const RESIDUAL_LABELS = {
  continuity: 'Continuity',
  momentum_x: 'Ux',
  momentum_y: 'Uy',
  momentum_z: 'Uz',
  energy: 'Energy',
  turbulent_ke: 'k',
  turbulent_omega: 'ω',
} as const;

// ============================================
// MAIN COMPONENT
// ============================================

export function ResidualChart({
  data: externalData,
  convergenceTolerance = 1e-6,
  height = 256,
  showCard = true,
  className = '',
}: ResidualChartProps) {
  // Use store data if not provided externally
  const storeResiduals = useSimulationStore(selectResiduals);
  const residuals = externalData ?? storeResiduals;

  // Determine which residuals to show based on data
  const activeResiduals = useMemo(() => {
    if (!residuals || !residuals.length) return [];
    const sample = residuals[0];
    if (!sample) return [];
    
    const active: (keyof typeof RESIDUAL_COLORS)[] = [];
    
    if (sample.continuity !== undefined) active.push('continuity');
    if (sample.momentum_x !== undefined) active.push('momentum_x');
    if (sample.momentum_y !== undefined) active.push('momentum_y');
    if (sample.momentum_z !== undefined) active.push('momentum_z');
    if (sample.energy !== undefined) active.push('energy');
    if (sample.turbulent_ke !== undefined) active.push('turbulent_ke');
    if (sample.turbulent_omega !== undefined) active.push('turbulent_omega');
    
    return active;
  }, [residuals]);

  // Calculate Y-axis domain
  const yDomain = useMemo(() => {
    if (!residuals.length) return [1e-8, 1];
    
    let min = Infinity;
    let max = -Infinity;
    
    residuals.forEach((point) => {
      activeResiduals.forEach((key) => {
        const val = point[key as keyof ResidualPoint] as number;
        if (val && val > 0) {
          min = Math.min(min, val);
          max = Math.max(max, val);
        }
      });
    });
    
    // Add padding
    return [Math.max(1e-10, min / 10), Math.min(10, max * 10)];
  }, [residuals, activeResiduals]);

  const chartContent = (
    <div style={{ height }} className={!showCard ? className : ''}>
      {residuals.length === 0 ? (
        <div className="h-full flex items-center justify-center text-muted-foreground">
          No residual data available
        </div>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={residuals}
            margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
          >
            <XAxis
              dataKey="iteration"
              tick={{ fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: '#374151' }}
            />
            <YAxis
              scale="log"
              domain={yDomain}
              tick={{ fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: '#374151' }}
              tickFormatter={(val) => val.toExponential(0)}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2937',
                border: '1px solid #374151',
                borderRadius: '6px',
                fontSize: '12px',
              }}
              labelStyle={{ color: '#9ca3af' }}
              formatter={(value: number) => value.toExponential(3)}
            />
            <Legend
              wrapperStyle={{ fontSize: '11px' }}
              iconType="line"
            />
            
            {/* Convergence threshold line */}
            <ReferenceLine
              y={convergenceTolerance}
              stroke="#4ade80"
              strokeDasharray="3 3"
              label={{
                value: 'Tolerance',
                position: 'right',
                fill: '#4ade80',
                fontSize: 10,
              }}
            />

            {/* Residual lines */}
            {activeResiduals.map((key) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={RESIDUAL_COLORS[key]}
                name={RESIDUAL_LABELS[key]}
                dot={false}
                strokeWidth={1.5}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );

  if (!showCard) {
    return chartContent;
  }

  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center justify-between">
          <span>Convergence History</span>
          {residuals.length > 0 && (
            <span className="text-xs font-normal text-muted-foreground">
              {residuals.length} iterations
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        {chartContent}
      </CardContent>
    </Card>
  );
}

// ============================================
// MINI VERSION FOR DASHBOARD
// ============================================

interface ResidualChartMiniProps {
  data?: ResidualPoint[];
  height?: number;
  className?: string;
}

export function ResidualChartMini({ data, height = 120, className = '' }: ResidualChartMiniProps) {
  return (
    <ResidualChart
      data={data}
      height={height}
      showCard={false}
      className={className}
    />
  );
}
