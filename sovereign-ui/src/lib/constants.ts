/**
 * Shared constants for the Sovereign UI.
 * Single source of truth for values used across multiple components.
 */

/** 20-color region palette shared by MeshViewer and RegionLegend. */
export const REGION_PALETTE: readonly string[] = [
  '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
  '#EC4899', '#06B6D4', '#84CC16', '#F97316', '#6366F1',
  '#14B8A6', '#E11D48', '#A855F7', '#0EA5E9', '#D946EF',
  '#22C55E', '#FB923C', '#4F46E5', '#2DD4BF', '#F43F5E',
] as const;

/** Get the palette color for a region by its ID. */
export function getRegionColor(regionId: string, overrides?: Record<string, string> | null): string {
  if (overrides && overrides[regionId]) return overrides[regionId];
  const idx = parseInt(regionId, 10) || 0;
  return REGION_PALETTE[idx % REGION_PALETTE.length];
}
