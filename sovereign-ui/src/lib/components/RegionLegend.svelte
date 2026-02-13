<script>
  import { getRegionColor } from '$lib/constants';

  /** @type {import('$lib/api-client').TwinSummary | null} */
  export let twin = null;

  /** @type {string | null} */
  export let highlightRegion = null;

  /** @type {Set<string>} */
  export let hiddenRegions = new Set();

  /** @type {(val: Set<string>) => void} */
  export let onHiddenRegionsChange = () => {};

  /** @type {(id: string) => void} */
  export let onRegionClick = () => {};

  /** @type {(id: string | null) => void} */
  export let onRegionHover = () => {};

  function getColor(id) {
    return getRegionColor(id);
  }

  function showAll() {
    onHiddenRegionsChange(new Set());
  }

  function hideAll() {
    if (!twin?.mesh?.regions) return;
    onHiddenRegionsChange(new Set(Object.keys(twin.mesh.regions)));
  }

  function formatStructure(s) {
    if (!s) return '—';
    return s.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, c => c.toUpperCase());
  }

  function formatMaterial(m) {
    if (!m) return '—';
    return m.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, c => c.toUpperCase());
  }

  $: regions = twin?.mesh?.regions ? Object.entries(twin.mesh.regions) : [];
</script>

{#if regions.length > 0}
  <div class="legend">
    <div class="legend-header">
      <span class="legend-title">Regions</span>
      <div style="display: flex; align-items: center; gap: 6px;">
        <button class="legend-toggle-btn" on:click={showAll} title="Show all">All</button>
        <button class="legend-toggle-btn" on:click={hideAll} title="Hide all">None</button>
        <span class="legend-count">{regions.length}</span>
      </div>
    </div>
    <div class="legend-items">
      {#each regions as [id, region]}
        <button
          class="legend-item"
          class:highlighted={highlightRegion === id}
          on:click={() => onRegionClick(id)}
          on:mouseenter={() => onRegionHover(id)}
          on:mouseleave={() => onRegionHover(null)}
        >
          <span class="legend-swatch" style="background: {getColor(id)};"></span>
          <div class="legend-info">
            <span class="legend-structure">{formatStructure(region.structure)}</span>
            <span class="legend-material">{formatMaterial(region.material)}</span>
          </div>
          <span class="legend-id">{id}</span>
        </button>
      {/each}
    </div>
  </div>
{/if}

<style>
  .legend {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .legend-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 8px 0;
  }

  .legend-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--sov-text-tertiary, #6B7280);
  }

  .legend-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--sov-text-muted, #4B5563);
    background: var(--sov-bg-elevated, #161920);
    padding: 1px 6px;
    border-radius: 3px;
  }

  .legend-items {
    display: flex;
    flex-direction: column;
    gap: 1px;
    max-height: 360px;
    overflow-y: auto;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 8px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    cursor: pointer;
    text-align: left;
    font-family: inherit;
    transition: all 120ms ease-out;
    width: 100%;
  }

  .legend-item:hover {
    background: var(--sov-bg-hover, #1C1F28);
    border-color: var(--sov-border, #1F2937);
  }

  .legend-item.highlighted {
    background: var(--sov-accent-glow, rgba(59, 130, 246, 0.12));
    border-color: var(--sov-accent, #3B82F6);
  }

  .legend-swatch {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
  }

  .legend-info {
    display: flex;
    flex-direction: column;
    gap: 0;
    flex: 1;
    min-width: 0;
  }

  .legend-structure {
    font-size: 12px;
    font-weight: 500;
    color: var(--sov-text-primary, #E8E8EC);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .legend-material {
    font-size: 10px;
    color: var(--sov-text-tertiary, #6B7280);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .legend-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--sov-text-muted, #4B5563);
    flex-shrink: 0;
  }

  .legend-toggle-btn {
    font-size: 9px;
    padding: 1px 6px;
    border: 1px solid var(--sov-border, #1F2937);
    border-radius: 3px;
    background: transparent;
    color: var(--sov-text-tertiary, #6B7280);
    cursor: pointer;
    font-family: inherit;
  }
  .legend-toggle-btn:hover {
    background: var(--sov-bg-hover, #1C1F28);
    color: var(--sov-text-secondary, #9CA3AF);
  }
</style>
