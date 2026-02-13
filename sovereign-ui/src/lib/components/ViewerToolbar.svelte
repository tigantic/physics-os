<script>
  export let showWireframe = false;
  export let showLandmarks = true;
  export let showLabels = true;
  export let tissueOpacity = 1.0;
  export let clipEnabled = false;
  export let clipAxis = 'y';
  export let clipPosition = 0;
  export let interactionMode = 'orbit';

  /** @type {{ mesh?: { n_regions: number, regions: Record<string, any> } } | null} */
  export let twin = null;

  /** @type {Set<string>} */
  export let hiddenRegions = new Set();

  /** @type {(val: boolean) => void} */
  export let onWireframeChange = () => {};
  /** @type {(val: boolean) => void} */
  export let onLandmarksChange = () => {};
  /** @type {(val: boolean) => void} */
  export let onLabelsChange = () => {};
  /** @type {(val: number) => void} */
  export let onOpacityChange = () => {};
  /** @type {(val: { enabled: boolean, axis: string, position: number }) => void} */
  export let onClipChange = () => {};
  /** @type {(val: string) => void} */
  export let onModeChange = () => {};
  /** @type {(val: Set<string>) => void} */
  export let onHiddenRegionsChange = () => {};
  /** @type {() => void} */
  export let onResetView = () => {};

  let showClipPanel = false;
  let showRegionPanel = false;

  function toggleRegion(id) {
    const next = new Set(hiddenRegions);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    hiddenRegions = next;
    onHiddenRegionsChange(hiddenRegions);
  }

  function showAllRegions() {
    hiddenRegions = new Set();
    onHiddenRegionsChange(hiddenRegions);
  }

  function hideAllRegions() {
    if (!twin?.mesh?.regions) return;
    hiddenRegions = new Set(Object.keys(twin.mesh.regions));
    onHiddenRegionsChange(hiddenRegions);
  }

  function soloRegion(id) {
    if (!twin?.mesh?.regions) return;
    hiddenRegions = new Set(Object.keys(twin.mesh.regions).filter(k => k !== id));
    onHiddenRegionsChange(hiddenRegions);
  }

  function formatName(n) {
    return n ? n.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) : '';
  }

  $: regions = twin?.mesh?.regions ? Object.entries(twin.mesh.regions) : [];
</script>

<div class="vt">
  <!-- Row 1: Primary toggles -->
  <div class="vt-row">
    <button class="vt-btn" class:active={showWireframe}
      on:click={() => onWireframeChange(!showWireframe)}
      title="Wireframe overlay">
      <span class="vt-icon">△</span>
      <span class="vt-label">Wire</span>
    </button>

    <button class="vt-btn" class:active={showLandmarks}
      on:click={() => onLandmarksChange(!showLandmarks)}
      title="Toggle landmarks">
      <span class="vt-icon">◉</span>
      <span class="vt-label">LM</span>
    </button>

    <button class="vt-btn" class:active={showLabels}
      on:click={() => onLabelsChange(!showLabels)}
      title="Toggle labels">
      <span class="vt-icon">Aa</span>
      <span class="vt-label">Labels</span>
    </button>

    <div class="vt-sep"></div>

    <button class="vt-btn" class:active={interactionMode === 'measure'}
      on:click={() => onModeChange(interactionMode === 'measure' ? 'orbit' : 'measure')}
      title="Measurement tool">
      <span class="vt-icon">📏</span>
      <span class="vt-label">Measure</span>
    </button>

    <button class="vt-btn" class:active={clipEnabled}
      on:click={() => { showClipPanel = !showClipPanel; }}
      title="Cross-section plane">
      <span class="vt-icon">✂</span>
      <span class="vt-label">Clip</span>
    </button>

    <button class="vt-btn" class:active={showRegionPanel}
      on:click={() => showRegionPanel = !showRegionPanel}
      title="Region visibility">
      <span class="vt-icon">◧</span>
      <span class="vt-label">Regions</span>
    </button>

    <div class="vt-sep"></div>

    <!-- Opacity slider -->
    <div class="vt-opacity">
      <span class="vt-opacity-label">α</span>
      <input type="range" class="vt-slider" min="0.1" max="1" step="0.05"
        value={tissueOpacity}
        on:input={(e) => onOpacityChange(parseFloat(e.target.value))} />
      <span class="vt-opacity-val">{(tissueOpacity * 100).toFixed(0)}%</span>
    </div>

    <div class="vt-sep"></div>

    <button class="vt-btn" on:click={() => onResetView()} title="Reset camera">
      <span class="vt-icon">⟲</span>
      <span class="vt-label">Reset</span>
    </button>
  </div>

  <!-- Clip panel -->
  {#if showClipPanel}
    <div class="vt-panel">
      <div class="vt-panel-header">
        <span>Cross-Section Plane</span>
        <button class="vt-panel-close" on:click={() => showClipPanel = false}>✕</button>
      </div>
      <div class="vt-panel-body">
        <label class="vt-panel-toggle">
          <input type="checkbox" checked={clipEnabled}
            on:change={() => onClipChange({ enabled: !clipEnabled, axis: clipAxis, position: clipPosition })} />
          <span>Enable clipping</span>
        </label>

        {#if clipEnabled}
          <div class="vt-panel-field">
            <span class="vt-panel-label">Axis</span>
            <div class="vt-axis-btns">
              {#each ['x', 'y', 'z'] as axis}
                <button class="vt-axis-btn" class:active={clipAxis === axis}
                  on:click={() => onClipChange({ enabled: clipEnabled, axis, position: clipPosition })}>
                  {axis.toUpperCase()}
                </button>
              {/each}
            </div>
          </div>
          <div class="vt-panel-field">
            <span class="vt-panel-label">Position</span>
            <input type="range" class="vt-slider wide" min="-100" max="100" step="0.5"
              value={clipPosition}
              on:input={(e) => onClipChange({ enabled: clipEnabled, axis: clipAxis, position: parseFloat(e.target.value) })} />
            <span class="font-data" style="font-size: 10px; color: var(--sov-text-muted);">
              {clipPosition.toFixed(1)}
            </span>
          </div>
        {/if}
      </div>
    </div>
  {/if}

  <!-- Region visibility panel -->
  {#if showRegionPanel}
    <div class="vt-panel" style="max-height: 300px;">
      <div class="vt-panel-header">
        <span>Region Visibility</span>
        <button class="vt-panel-close" on:click={() => showRegionPanel = false}>✕</button>
      </div>
      <div class="vt-panel-actions">
        <button class="vt-mini-btn" on:click={showAllRegions}>Show All</button>
        <button class="vt-mini-btn" on:click={hideAllRegions}>Hide All</button>
      </div>
      <div class="vt-panel-body" style="max-height: 200px; overflow-y: auto;">
        {#each regions as [id, region]}
          <div class="vt-region-row">
            <label class="vt-region-toggle">
              <input type="checkbox" checked={!hiddenRegions.has(id)}
                on:change={() => toggleRegion(id)} />
              <span class="vt-region-name">{formatName(region.structure)}</span>
            </label>
            <button class="vt-solo-btn" on:click={() => soloRegion(id)} title="Solo this region">
              S
            </button>
          </div>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .vt { display: flex; flex-direction: column; gap: 0; }

  .vt-row {
    display: flex; align-items: center; gap: 2px;
    padding: 4px 8px; flex-wrap: wrap;
  }

  .vt-btn {
    display: flex; align-items: center; gap: 4px;
    padding: 4px 8px; background: transparent;
    border: 1px solid transparent; border-radius: 3px;
    cursor: pointer; font-family: inherit; font-size: 11px;
    color: var(--sov-text-muted, #4B5563);
    transition: all 100ms ease-out; white-space: nowrap;
  }
  .vt-btn:hover { background: var(--sov-bg-hover, #1C1F28); color: var(--sov-text-secondary, #9CA3AF); }
  .vt-btn.active {
    background: var(--sov-accent-glow, rgba(59,130,246,0.12));
    color: var(--sov-accent, #3B82F6);
    border-color: var(--sov-accent, #3B82F6)40;
  }
  .vt-icon { font-size: 12px; }
  .vt-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em; }

  .vt-sep { width: 1px; height: 18px; background: var(--sov-border-subtle, #151820); margin: 0 4px; }

  .vt-opacity {
    display: flex; align-items: center; gap: 4px;
  }
  .vt-opacity-label { font-size: 11px; color: var(--sov-text-muted, #4B5563); font-family: 'JetBrains Mono', monospace; }
  .vt-opacity-val { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--sov-text-tertiary, #6B7280); min-width: 28px; }

  .vt-slider {
    width: 80px; height: 3px;
    -webkit-appearance: none; appearance: none;
    background: var(--sov-bg-elevated, #161920);
    border-radius: 2px; outline: none; cursor: pointer;
  }
  .vt-slider.wide { width: 120px; }
  .vt-slider::-webkit-slider-thumb {
    -webkit-appearance: none; width: 12px; height: 12px;
    background: var(--sov-accent, #3B82F6); border-radius: 50%;
    cursor: pointer; border: 2px solid var(--sov-bg-card, #111318);
  }

  /* Panels */
  .vt-panel {
    background: var(--sov-bg-card, #111318);
    border: 1px solid var(--sov-border, #1F2937);
    border-radius: 0 0 6px 6px; margin: 0 4px;
  }
  .vt-panel-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 10px; font-size: 11px; font-weight: 500;
    color: var(--sov-text-secondary, #9CA3AF);
    border-bottom: 1px solid var(--sov-border-subtle, #151820);
  }
  .vt-panel-close {
    background: none; border: none; color: var(--sov-text-muted, #4B5563);
    cursor: pointer; font-size: 12px;
  }
  .vt-panel-body { padding: 8px 10px; display: flex; flex-direction: column; gap: 8px; }
  .vt-panel-field { display: flex; align-items: center; gap: 8px; }
  .vt-panel-label { font-size: 10px; color: var(--sov-text-muted, #4B5563); min-width: 40px; }
  .vt-panel-toggle {
    display: flex; align-items: center; gap: 6px; font-size: 11px;
    color: var(--sov-text-secondary, #9CA3AF); cursor: pointer;
  }
  .vt-panel-toggle input { accent-color: var(--sov-accent, #3B82F6); }

  .vt-panel-actions {
    display: flex; gap: 4px; padding: 4px 10px;
  }
  .vt-mini-btn {
    padding: 2px 8px; font-size: 10px; background: var(--sov-bg-elevated, #161920);
    border: 1px solid var(--sov-border, #1F2937); border-radius: 3px;
    color: var(--sov-text-tertiary, #6B7280); cursor: pointer; font-family: inherit;
  }
  .vt-mini-btn:hover { color: var(--sov-text-secondary, #9CA3AF); border-color: var(--sov-text-muted, #4B5563); }

  .vt-axis-btns { display: flex; gap: 2px; }
  .vt-axis-btn {
    width: 28px; height: 22px; display: flex; align-items: center; justify-content: center;
    background: var(--sov-bg-elevated, #161920); border: 1px solid var(--sov-border, #1F2937);
    border-radius: 3px; color: var(--sov-text-tertiary, #6B7280);
    cursor: pointer; font-family: 'JetBrains Mono', monospace; font-size: 10px;
  }
  .vt-axis-btn.active {
    background: var(--sov-accent-dim, #1E3A5F);
    color: var(--sov-accent, #3B82F6);
    border-color: var(--sov-accent, #3B82F6)40;
  }

  .vt-region-row {
    display: flex; align-items: center; gap: 4px;
  }
  .vt-region-toggle {
    flex: 1; display: flex; align-items: center; gap: 6px;
    font-size: 11px; color: var(--sov-text-secondary, #9CA3AF); cursor: pointer;
  }
  .vt-region-toggle input { accent-color: var(--sov-accent, #3B82F6); }
  .vt-region-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .vt-solo-btn {
    width: 18px; height: 18px; font-size: 9px; font-weight: 700;
    background: transparent; border: 1px solid var(--sov-border, #1F2937);
    border-radius: 2px; color: var(--sov-text-muted, #4B5563);
    cursor: pointer; display: flex; align-items: center; justify-content: center;
  }
  .vt-solo-btn:hover { color: var(--sov-accent, #3B82F6); border-color: var(--sov-accent, #3B82F6); }
</style>
