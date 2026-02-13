<script>
  import ColorBar from './ColorBar.svelte';

  /** @type {FemResults | null} */
  export let femData = null;

  /** @type {boolean} */
  export let visible = false;

  /** @type {string} */
  export let activeField = 'stress';

  /** @type {string} */
  export let colormap = 'stress';

  // ── Callback props (replaces createEventDispatcher) ────────
  /** @type {(val: boolean) => void} */
  export let onVisibilityChange = () => {};
  /** @type {(detail: { field: string, colormap: string }) => void} */
  export let onFieldChange = () => {};
  /** @type {() => void} */
  export let onRunFem = () => {};

  $: hasData = femData != null;
  $: fields = femData?.fields ?? {};
  $: activeFieldData = fields[activeField] ?? null;
  $: solver = femData?.solver ?? null;
  $: boundaries = femData?.boundary_conditions ?? [];
  $: materialProps = femData?.material_properties ?? [];

  const FIELD_META = {
    stress:       { label: 'Von Mises Stress', unit: 'MPa',  cmap: 'stress',   precision: 3 },
    displacement: { label: 'Displacement',     unit: 'mm',   cmap: 'viridis',  precision: 3 },
    strain:       { label: 'Strain',           unit: 'ε',    cmap: 'plasma',   precision: 4 },
    thickness:    { label: 'Tissue Thickness', unit: 'mm',   cmap: 'coolwarm', precision: 2 },
    pressure:     { label: 'Contact Pressure', unit: 'kPa',  cmap: 'inferno',  precision: 2 },
  };

  $: meta = FIELD_META[activeField] || FIELD_META.stress;

  function selectField(f) {
    activeField = f;
    colormap = FIELD_META[f]?.cmap || 'stress';
    onFieldChange({ field: f, colormap });
  }
</script>

<div class="fem-panel">
  <div class="fem-header">
    <span class="fem-title">Finite Element Analysis</span>
    <label class="fem-toggle">
      <input type="checkbox" bind:checked={visible} on:change={() => onVisibilityChange(visible)} />
      <span>Overlay</span>
    </label>
  </div>

  {#if !hasData}
    <div class="fem-empty">
      <span class="fem-empty-icon">🔥</span>
      <span>No FEM results available</span>
      <span class="fem-empty-sub">Run biomechanical simulation after plan compilation</span>
      <button class="sov-btn sov-btn-primary sov-btn-sm" on:click={onRunFem}>
        Run FEM Simulation
      </button>
    </div>
  {:else}
    <!-- Field selector tabs -->
    <div class="fem-field-tabs">
      {#each Object.entries(fields) as [key]}
        {@const fm = FIELD_META[key] || { label: key }}
        <button class="fem-field-tab" class:active={activeField === key}
          on:click={() => selectField(key)}>
          {fm.label}
        </button>
      {/each}
    </div>

    <!-- Active field colorbar + stats -->
    {#if activeFieldData}
      <div class="fem-colorbar">
        <ColorBar
          label={meta.label}
          unit={meta.unit}
          min={activeFieldData.min ?? 0}
          max={activeFieldData.max ?? 1}
          colormap={meta.cmap}
          orientation="horizontal"
          ticks={6}
          precision={meta.precision}
          compact
        />
      </div>

      <div class="fem-stats">
        <div class="fem-stat">
          <span class="fem-stat-val">{activeFieldData.min?.toFixed(meta.precision) ?? '—'}</span>
          <span class="fem-stat-label">Min {meta.unit}</span>
        </div>
        <div class="fem-stat">
          <span class="fem-stat-val">{activeFieldData.max?.toFixed(meta.precision) ?? '—'}</span>
          <span class="fem-stat-label">Max {meta.unit}</span>
        </div>
        <div class="fem-stat">
          <span class="fem-stat-val">{activeFieldData.mean?.toFixed(meta.precision) ?? '—'}</span>
          <span class="fem-stat-label">Mean {meta.unit}</span>
        </div>
        <div class="fem-stat">
          <span class="fem-stat-val">{activeFieldData.std?.toFixed(meta.precision) ?? '—'}</span>
          <span class="fem-stat-label">Std Dev</span>
        </div>
      </div>

      <!-- Per-region breakdown if available -->
      {#if activeFieldData.by_region}
        <div class="fem-section">
          <span class="fem-section-label">By Region</span>
          <div class="fem-regions">
            {#each Object.entries(activeFieldData.by_region) as [region, vals]}
              <div class="fem-region-row">
                <span class="fem-region-name">{region.replace(/_/g, ' ')}</span>
                <span class="fem-region-val">{vals.max?.toFixed(meta.precision)}</span>
                <div class="fem-region-bar-wrap">
                  <div class="fem-region-bar"
                    style="width: {((vals.max - activeFieldData.min) / (activeFieldData.max - activeFieldData.min || 1)) * 100}%;
                           background: {vals.max > activeFieldData.mean * 1.5 ? '#EF4444' : vals.max > activeFieldData.mean ? '#F59E0B' : '#3B82F6'};"></div>
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    {/if}

    <!-- Material properties -->
    {#if materialProps.length > 0}
      <div class="fem-section">
        <span class="fem-section-label">Material Properties</span>
        <div class="fem-mat-table">
          {#each materialProps as mat}
            <div class="fem-mat-row">
              <span class="fem-mat-name">{mat.tissue ?? mat.name}</span>
              <div class="fem-mat-props">
                {#if mat.youngs_modulus != null}<span>E={mat.youngs_modulus}{mat.unit ?? 'kPa'}</span>{/if}
                {#if mat.poisson_ratio != null}<span>ν={mat.poisson_ratio}</span>{/if}
                {#if mat.density != null}<span>ρ={mat.density}</span>{/if}
              </div>
            </div>
          {/each}
        </div>
      </div>
    {/if}

    <!-- Solver info -->
    {#if solver}
      <div class="fem-section">
        <span class="fem-section-label">Solver</span>
        <div class="fem-meta">
          {#if solver.type}<div class="fem-meta-row"><span>Type</span><span>{solver.type}</span></div>{/if}
          {#if solver.elements}<div class="fem-meta-row"><span>Elements</span><span>{solver.elements.toLocaleString()}</span></div>{/if}
          {#if solver.nodes}<div class="fem-meta-row"><span>Nodes</span><span>{solver.nodes.toLocaleString()}</span></div>{/if}
          {#if solver.dof}<div class="fem-meta-row"><span>DOF</span><span>{solver.dof.toLocaleString()}</span></div>{/if}
          {#if solver.iterations}<div class="fem-meta-row"><span>Iterations</span><span>{solver.iterations}</span></div>{/if}
          {#if solver.residual}<div class="fem-meta-row"><span>Residual</span><span>{solver.residual}</span></div>{/if}
          {#if solver.compute_time}<div class="fem-meta-row"><span>Time</span><span>{solver.compute_time}</span></div>{/if}
        </div>
      </div>
    {/if}
  {/if}
</div>

<style>
  .fem-panel { display: flex; flex-direction: column; gap: 8px; }
  .fem-header { display: flex; justify-content: space-between; align-items: center; }
  .fem-title { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--sov-text-tertiary); }
  .fem-toggle { display: flex; align-items: center; gap: 4px; font-size: 10px; color: var(--sov-text-muted); cursor: pointer; }
  .fem-toggle input { accent-color: var(--sov-accent); }

  .fem-empty { display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 20px 0; color: var(--sov-text-muted); font-size: 12px; text-align: center; }
  .fem-empty-icon { font-size: 24px; opacity: 0.4; }
  .fem-empty-sub { font-size: 10px; }

  .fem-field-tabs { display: flex; gap: 2px; flex-wrap: wrap; }
  .fem-field-tab {
    padding: 3px 8px; font-size: 10px; background: var(--sov-bg-elevated); border: 1px solid var(--sov-border);
    border-radius: 3px; color: var(--sov-text-muted); cursor: pointer; font-family: inherit;
  }
  .fem-field-tab.active { background: var(--sov-accent-dim); color: var(--sov-accent); border-color: var(--sov-accent)40; }

  .fem-colorbar { padding: 4px 0; }

  .fem-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .fem-stat { padding: 5px 8px; background: var(--sov-bg-root); border: 1px solid var(--sov-border-subtle); border-radius: 4px; }
  .fem-stat-val { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; color: var(--sov-text-primary); display: block; }
  .fem-stat-label { font-size: 9px; color: var(--sov-text-muted); }

  .fem-section { padding-top: 6px; border-top: 1px solid var(--sov-border-subtle); }
  .fem-section-label { font-size: 10px; font-weight: 600; color: var(--sov-text-muted); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; display: block; }

  .fem-regions { display: flex; flex-direction: column; gap: 4px; }
  .fem-region-row { display: flex; align-items: center; gap: 6px; }
  .fem-region-name { font-size: 10px; color: var(--sov-text-secondary); width: 70px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .fem-region-val { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--sov-text-tertiary); width: 50px; text-align: right; }
  .fem-region-bar-wrap { flex: 1; height: 4px; background: var(--sov-bg-elevated); border-radius: 2px; overflow: hidden; }
  .fem-region-bar { height: 100%; border-radius: 2px; transition: width 200ms; }

  .fem-mat-table { display: flex; flex-direction: column; gap: 3px; }
  .fem-mat-row { display: flex; justify-content: space-between; align-items: center; padding: 3px 0; }
  .fem-mat-name { font-size: 10px; color: var(--sov-text-secondary); }
  .fem-mat-props { display: flex; gap: 8px; font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--sov-text-muted); }

  .fem-meta { display: flex; flex-direction: column; gap: 2px; }
  .fem-meta-row { display: flex; justify-content: space-between; font-size: 10px; color: var(--sov-text-tertiary); }
  .fem-meta-row span:last-child { font-family: 'JetBrains Mono', monospace; color: var(--sov-text-secondary); }
</style>
