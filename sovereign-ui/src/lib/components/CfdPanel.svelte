<script>
  import ColorBar from './ColorBar.svelte';

  /** @type {CfdResults | null} */
  export let cfdData = null;

  /** @type {boolean} */
  export let visible = true;

  /** @type {number} */
  export let streamlineOpacity = 0.7;

  /** @type {number} */
  export let streamlineDensity = 1.0;

  /** @type {string} */
  export let colormap = 'jet';

  // ── Callback props (replaces createEventDispatcher) ────────
  /** @type {(val: boolean) => void} */
  export let onVisibilityChange = () => {};
  /** @type {() => void} */
  export let onRunCfd = () => {};
  /** @type {(val: number) => void} */
  export let onOpacityChange = () => {};
  /** @type {(val: number) => void} */
  export let onDensityChange = () => {};
  /** @type {(val: string) => void} */
  export let onColormapChange = () => {};

  $: hasData = cfdData != null;
  $: stats = cfdData?.summary ?? null;
  $: resistance = cfdData?.resistance ?? null;
  $: velocityRange = cfdData ? { min: cfdData.velocity_min ?? 0, max: cfdData.velocity_max ?? 5 } : { min: 0, max: 5 };
</script>

<div class="cfd-panel">
  <div class="cfd-header">
    <span class="cfd-title">Computational Fluid Dynamics</span>
    <label class="cfd-toggle">
      <input type="checkbox" bind:checked={visible} on:change={() => onVisibilityChange(visible)} />
      <span>Visible</span>
    </label>
  </div>

  {#if !hasData}
    <div class="cfd-empty">
      <span class="cfd-empty-icon">🌊</span>
      <span>No CFD results available</span>
      <span class="cfd-empty-sub">Run airflow simulation after plan compilation</span>
      <button class="sov-btn sov-btn-primary sov-btn-sm" on:click={onRunCfd}>
        Run Airflow Simulation
      </button>
    </div>
  {:else}
    <!-- Velocity colorbar -->
    <div class="cfd-colorbar">
      <ColorBar
        label="Velocity"
        unit="m/s"
        min={velocityRange.min}
        max={velocityRange.max}
        {colormap}
        orientation="horizontal"
        ticks={5}
        precision={1}
        compact
      />
    </div>

    <!-- Flow stats -->
    {#if stats}
      <div class="cfd-stats">
        <div class="cfd-stat">
          <span class="cfd-stat-val">{stats.peak_velocity?.toFixed(2) ?? '—'}</span>
          <span class="cfd-stat-label">Peak Vel (m/s)</span>
        </div>
        <div class="cfd-stat">
          <span class="cfd-stat-val">{stats.mean_velocity?.toFixed(2) ?? '—'}</span>
          <span class="cfd-stat-label">Mean Vel (m/s)</span>
        </div>
        <div class="cfd-stat">
          <span class="cfd-stat-val">{stats.flow_rate?.toFixed(1) ?? '—'}</span>
          <span class="cfd-stat-label">Flow Rate (L/min)</span>
        </div>
        <div class="cfd-stat">
          <span class="cfd-stat-val">{stats.reynolds?.toFixed(0) ?? '—'}</span>
          <span class="cfd-stat-label">Reynolds #</span>
        </div>
      </div>
    {/if}

    <!-- Nasal resistance -->
    {#if resistance}
      <div class="cfd-section">
        <span class="cfd-section-label">Nasal Resistance</span>
        <div class="cfd-resistance">
          <div class="cfd-res-row">
            <span>Left</span>
            <div class="cfd-res-bar-wrap">
              <div class="cfd-res-bar left" style="width: {Math.min(100, (resistance.left / (resistance.left + resistance.right)) * 100)}%;"></div>
            </div>
            <span class="cfd-res-val">{resistance.left?.toFixed(2)} Pa·s/mL</span>
          </div>
          <div class="cfd-res-row">
            <span>Right</span>
            <div class="cfd-res-bar-wrap">
              <div class="cfd-res-bar right" style="width: {Math.min(100, (resistance.right / (resistance.left + resistance.right)) * 100)}%;"></div>
            </div>
            <span class="cfd-res-val">{resistance.right?.toFixed(2)} Pa·s/mL</span>
          </div>
          <div class="cfd-res-row total">
            <span>Total</span>
            <span class="cfd-res-val">{resistance.total?.toFixed(2)} Pa·s/mL</span>
          </div>
        </div>
      </div>
    {/if}

    <!-- Controls -->
    <div class="cfd-section">
      <span class="cfd-section-label">Display</span>
      <div class="cfd-control">
        <span class="cfd-ctrl-label">Opacity</span>
        <input type="range" class="cfd-slider" min="0.1" max="1" step="0.05"
          bind:value={streamlineOpacity}
          on:input={() => onOpacityChange(streamlineOpacity)} />
        <span class="cfd-ctrl-val">{(streamlineOpacity * 100).toFixed(0)}%</span>
      </div>
      <div class="cfd-control">
        <span class="cfd-ctrl-label">Density</span>
        <input type="range" class="cfd-slider" min="0.1" max="2" step="0.1"
          bind:value={streamlineDensity}
          on:input={() => onDensityChange(streamlineDensity)} />
        <span class="cfd-ctrl-val">{streamlineDensity.toFixed(1)}x</span>
      </div>
      <div class="cfd-control">
        <span class="cfd-ctrl-label">Colormap</span>
        <select class="cfd-select" bind:value={colormap}
          on:change={() => onColormapChange(colormap)}>
          <option value="jet">Jet</option>
          <option value="viridis">Viridis</option>
          <option value="plasma">Plasma</option>
          <option value="coolwarm">Cool-Warm</option>
        </select>
      </div>
    </div>

    <!-- Simulation info -->
    {#if cfdData.solver}
      <div class="cfd-section">
        <span class="cfd-section-label">Solver</span>
        <div class="cfd-meta">
          {#if cfdData.solver.method}<div class="cfd-meta-row"><span>Method</span><span>{cfdData.solver.method}</span></div>{/if}
          {#if cfdData.solver.mesh_elements}<div class="cfd-meta-row"><span>Elements</span><span>{cfdData.solver.mesh_elements.toLocaleString()}</span></div>{/if}
          {#if cfdData.solver.iterations}<div class="cfd-meta-row"><span>Iterations</span><span>{cfdData.solver.iterations}</span></div>{/if}
          {#if cfdData.solver.convergence}<div class="cfd-meta-row"><span>Convergence</span><span>{cfdData.solver.convergence}</span></div>{/if}
          {#if cfdData.solver.compute_time}<div class="cfd-meta-row"><span>Time</span><span>{cfdData.solver.compute_time}</span></div>{/if}
        </div>
      </div>
    {/if}
  {/if}
</div>

<style>
  .cfd-panel { display: flex; flex-direction: column; gap: 8px; }
  .cfd-header { display: flex; justify-content: space-between; align-items: center; }
  .cfd-title { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--sov-text-tertiary); }
  .cfd-toggle { display: flex; align-items: center; gap: 4px; font-size: 10px; color: var(--sov-text-muted); cursor: pointer; }
  .cfd-toggle input { accent-color: var(--sov-accent); }

  .cfd-empty { display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 20px 0; color: var(--sov-text-muted); font-size: 12px; text-align: center; }
  .cfd-empty-icon { font-size: 24px; opacity: 0.4; }
  .cfd-empty-sub { font-size: 10px; color: var(--sov-text-muted); }

  .cfd-colorbar { padding: 4px 0; }

  .cfd-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .cfd-stat { padding: 6px 8px; background: var(--sov-bg-root); border: 1px solid var(--sov-border-subtle); border-radius: 4px; }
  .cfd-stat-val { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; color: var(--sov-accent); display: block; }
  .cfd-stat-label { font-size: 9px; color: var(--sov-text-muted); text-transform: uppercase; letter-spacing: 0.03em; }

  .cfd-section { padding-top: 6px; border-top: 1px solid var(--sov-border-subtle); }
  .cfd-section-label { font-size: 10px; font-weight: 600; color: var(--sov-text-muted); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; display: block; }

  .cfd-resistance { display: flex; flex-direction: column; gap: 6px; }
  .cfd-res-row { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--sov-text-secondary); }
  .cfd-res-row span:first-child { width: 36px; font-size: 10px; }
  .cfd-res-bar-wrap { flex: 1; height: 6px; background: var(--sov-bg-elevated); border-radius: 3px; overflow: hidden; }
  .cfd-res-bar { height: 100%; border-radius: 3px; transition: width 300ms ease-out; }
  .cfd-res-bar.left { background: #3B82F6; }
  .cfd-res-bar.right { background: #10B981; }
  .cfd-res-val { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--sov-text-tertiary); min-width: 80px; text-align: right; }
  .cfd-res-row.total { padding-top: 4px; border-top: 1px solid var(--sov-border-subtle); font-weight: 500; }

  .cfd-control { display: flex; align-items: center; gap: 6px; padding: 2px 0; }
  .cfd-ctrl-label { font-size: 10px; color: var(--sov-text-muted); min-width: 50px; }
  .cfd-slider { flex: 1; height: 3px; -webkit-appearance: none; background: var(--sov-bg-elevated); border-radius: 2px; }
  .cfd-slider::-webkit-slider-thumb { -webkit-appearance: none; width: 10px; height: 10px; background: var(--sov-accent); border-radius: 50%; cursor: pointer; }
  .cfd-ctrl-val { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--sov-text-muted); min-width: 28px; text-align: right; }
  .cfd-select { font-family: 'JetBrains Mono', monospace; font-size: 10px; background: var(--sov-bg-root); color: var(--sov-text-secondary); border: 1px solid var(--sov-border); border-radius: 3px; padding: 2px 4px; }

  .cfd-meta { display: flex; flex-direction: column; gap: 2px; }
  .cfd-meta-row { display: flex; justify-content: space-between; font-size: 10px; color: var(--sov-text-tertiary); }
  .cfd-meta-row span:last-child { font-family: 'JetBrains Mono', monospace; color: var(--sov-text-secondary); }
</style>
