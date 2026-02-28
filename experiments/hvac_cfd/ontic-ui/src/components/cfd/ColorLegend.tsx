/**
 * ColorLegend - Scientific Colormap Legend
 * 
 * Displays a vertical or horizontal colorbar with value labels.
 * Supports multiple scientific colormaps.
 */

'use client';

import { useMemo } from 'react';
import type { ColormapName } from '@/types';

// ============================================
// PROPS INTERFACE
// ============================================

interface ColorLegendProps {
  min: number;
  max: number;
  label?: string;
  unit?: string;
  colormap?: ColormapName;
  orientation?: 'vertical' | 'horizontal';
  height?: number;
  width?: number;
  className?: string;
}

// ============================================
// COLORMAP DEFINITIONS
// ============================================

const COLORMAPS: Record<ColormapName, string[]> = {
  viridis: [
    '#440154', '#482878', '#3e4a89', '#31688e', '#26838f',
    '#1f9d8a', '#6cce5a', '#b6de2b', '#fee825',
  ],
  plasma: [
    '#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786',
    '#d8576b', '#ed7953', '#fb9f3a', '#fdca26',
  ],
  inferno: [
    '#000004', '#1b0c41', '#4a0c6b', '#781c6d', '#a52c60',
    '#cf4446', '#ed6925', '#fb9b06', '#fcffa4',
  ],
  magma: [
    '#000004', '#180f3d', '#440f76', '#721f81', '#9e2f7f',
    '#cd4071', '#f1605d', '#fd9668', '#fcfdbf',
  ],
  coolwarm: [
    '#3b4cc0', '#5977e3', '#7b9ff9', '#9ebeff', '#c0d4f5',
    '#f2cbb7', '#f7a889', '#ee7b51', '#d6412a', '#b40426',
  ],
  jet: [
    '#00007f', '#0000ff', '#007fff', '#00ffff', '#7fff7f',
    '#ffff00', '#ff7f00', '#ff0000', '#7f0000',
  ],
  turbo: [
    '#30123b', '#4145ab', '#4675ed', '#39a2fc', '#1bd0d5',
    '#4af47e', '#a4fb39', '#e9d938', '#f99c3c', '#dd503a',
  ],
  rainbow: [
    '#ff0000', '#ff7f00', '#ffff00', '#7fff00', '#00ff00',
    '#00ff7f', '#00ffff', '#007fff', '#0000ff', '#7f00ff',
  ],
};

// ============================================
// MAIN COMPONENT
// ============================================

export function ColorLegend({
  min,
  max,
  label,
  unit,
  colormap = 'viridis',
  orientation = 'vertical',
  height = 200,
  width = 20,
  className = '',
}: ColorLegendProps) {
  // Generate gradient CSS
  const gradientStyle = useMemo(() => {
    const colors = COLORMAPS[colormap];
    const stops = colors.map((color, i) => {
      const percent = (i / (colors.length - 1)) * 100;
      return `${color} ${percent}%`;
    });

    const direction = orientation === 'vertical' ? 'to top' : 'to right';
    return {
      background: `linear-gradient(${direction}, ${stops.join(', ')})`,
    };
  }, [colormap, orientation]);

  // Generate tick values
  const ticks = useMemo(() => {
    const count = 5;
    const step = (max - min) / (count - 1);
    return Array.from({ length: count }, (_, i) => min + step * i);
  }, [min, max]);

  // Format number for display
  const formatValue = (val: number): string => {
    if (Math.abs(val) >= 1000 || (Math.abs(val) < 0.01 && val !== 0)) {
      return val.toExponential(2);
    }
    return val.toFixed(2);
  };

  if (orientation === 'horizontal') {
    return (
      <div className={`flex flex-col gap-1 ${className}`}>
        {label && (
          <div className="text-xs text-muted-foreground text-center">
            {label} {unit && `(${unit})`}
          </div>
        )}
        <div
          className="rounded"
          style={{ ...gradientStyle, height: width, width: '100%' }}
        />
        <div className="flex justify-between text-xs text-muted-foreground font-mono">
          <span>{formatValue(min)}</span>
          <span>{formatValue((min + max) / 2)}</span>
          <span>{formatValue(max)}</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex gap-2 ${className}`}>
      {/* Colorbar */}
      <div
        className="rounded"
        style={{ ...gradientStyle, width, height }}
      />

      {/* Ticks and labels */}
      <div
        className="flex flex-col justify-between text-xs text-muted-foreground font-mono"
        style={{ height }}
      >
        {ticks.reverse().map((val, i) => (
          <span key={i}>{formatValue(val)}</span>
        ))}
      </div>

      {/* Label */}
      {label && (
        <div
          className="text-xs text-muted-foreground writing-mode-vertical"
          style={{ writingMode: 'vertical-rl', textOrientation: 'mixed' }}
        >
          {label} {unit && `(${unit})`}
        </div>
      )}
    </div>
  );
}

// ============================================
// COLORMAP SELECTOR
// ============================================

interface ColormapSelectorProps {
  value: ColormapName;
  onChange: (colormap: ColormapName) => void;
  className?: string;
}

export function ColormapSelector({ value, onChange, className = '' }: ColormapSelectorProps) {
  return (
    <div className={`flex flex-wrap gap-2 ${className}`}>
      {(Object.keys(COLORMAPS) as ColormapName[]).map((name) => {
        const colors = COLORMAPS[name];
        const gradient = `linear-gradient(to right, ${colors.join(', ')})`;
        const isSelected = name === value;

        return (
          <button
            key={name}
            onClick={() => onChange(name)}
            className={`
              w-16 h-6 rounded cursor-pointer border-2 transition-all
              ${isSelected ? 'border-primary ring-2 ring-primary/30' : 'border-transparent hover:border-muted'}
            `}
            style={{ background: gradient }}
            title={name}
          />
        );
      })}
    </div>
  );
}
