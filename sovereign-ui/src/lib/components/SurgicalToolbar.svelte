<script>
  /** @type {'orbit' | 'measure' | 'incision' | 'osteotomy' | 'graft'} */
  export let mode = 'orbit';

  /** @type {boolean} */
  export let canUndo = false;

  /** @type {boolean} */
  export let canCommit = false;

  /** @type {boolean} */
  export let showStreamlines = true;

  /** @type {boolean} */
  export let showScalarField = false;

  /** @type {boolean} */
  export let showLandmarks = true;

  /** @type {boolean} */
  export let showLabels = true;

  /** @type {boolean} */
  export let showIncisions = true;

  /** @type {boolean} */
  export let clipEnabled = false;

  /** @type {string} */
  export let scalarFieldType = 'stress';

  /** @type {string} */
  export let viewPreset = 'default';

  // ── Callback props (replaces createEventDispatcher) ────────
  /** @type {(mode: string) => void} */
  export let onModeChange = () => {};
  /** @type {() => void} */
  export let onCommit = () => {};
  /** @type {() => void} */
  export let onUndo = () => {};
  /** @type {() => void} */
  export let onCancel = () => {};
  /** @type {(val: boolean) => void} */
  export let onStreamlinesToggle = () => {};
  /** @type {(val: boolean) => void} */
  export let onScalarToggle = () => {};
  /** @type {(val: boolean) => void} */
  export let onLandmarksToggle = () => {};
  /** @type {(val: boolean) => void} */
  export let onLabelsToggle = () => {};
  /** @type {(val: boolean) => void} */
  export let onIncisionsToggle = () => {};
  /** @type {(val: boolean) => void} */
  export let onClipToggle = () => {};
  /** @type {(type: string) => void} */
  export let onScalarTypeChange = () => {};
  /** @type {() => void} */
  export let onResetView = () => {};
  /** @type {(preset: string) => void} */
  export let onViewPreset = () => {};

  const modes = [
    { key: 'orbit',     icon: '⟲', label: 'Navigate', hotkey: 'V' },
    { key: 'measure',   icon: '📏', label: 'Measure',  hotkey: 'M' },
    { key: 'incision',  icon: '🔪', label: 'Incision',  hotkey: 'I' },
    { key: 'osteotomy', icon: '⚒',  label: 'Osteotomy', hotkey: 'O' },
    { key: 'graft',     icon: '◆',  label: 'Graft',     hotkey: 'G' },
  ];

  const viewPresets = [
    { key: 'default',  label: 'Default' },
    { key: 'anterior', label: 'Anterior' },
    { key: 'lateral',  label: 'Lateral' },
    { key: 'inferior', label: 'Inferior' },
    { key: 'oblique',  label: '3/4 View' },
  ];

  const scalarOptions = [
    { key: 'stress',       label: 'Von Mises Stress' },
    { key: 'displacement', label: 'Displacement' },
    { key: 'strain',       label: 'Strain' },
    { key: 'thickness',    label: 'Tissue Thickness' },
    { key: 'pressure',     label: 'Pressure' },
  ];

  function setMode(m) {
    mode = m;
    onModeChange(m);
  }
</script>

<div class="st">
  <!-- Mode selector -->
  <div class="st-group">
    <span class="st-group-label">Tools</span>
    <div class="st-modes">
      {#each modes as m}
        <button class="st-btn" class:active={mode === m.key}
          on:click={() => setMode(m.key)}
          title="{m.label} ({m.hotkey})">
          <span class="st-icon">{m.icon}</span>
          <span class="st-lbl">{m.label}</span>
          <span class="st-key">{m.hotkey}</span>
        </button>
      {/each}
    </div>
  </div>

  <!-- Commit / undo for active tools -->
  {#if mode !== 'orbit'}
    <div class="st-group">
      <span class="st-group-label">Action</span>
      <div class="st-actions">
        <button class="st-act-btn commit" disabled={!canCommit}
          on:click={onCommit}>
          ✓ Commit
        </button>
        <button class="st-act-btn undo" disabled={!canUndo}
          on:click={onUndo}>
          ↩ Undo
        </button>
        <button class="st-act-btn cancel"
          on:click={() => { setMode('orbit'); onCancel(); }}>
          ✕ Cancel
        </button>
      </div>
    </div>
  {/if}

  <div class="st-sep"></div>

  <!-- Visualization toggles -->
  <div class="st-group">
    <span class="st-group-label">Display</span>
    <div class="st-toggles">
      <button class="st-btn" class:active={showStreamlines}
        on:click={() => { showStreamlines = !showStreamlines; onStreamlinesToggle(showStreamlines); }}
        title="CFD Airflow">
        <span class="st-icon">🌊</span><span class="st-lbl">Flow</span>
      </button>
      <button class="st-btn" class:active={showScalarField}
        on:click={() => { showScalarField = !showScalarField; onScalarToggle(showScalarField); }}
        title="FEM Scalar Field">
        <span class="st-icon">🔥</span><span class="st-lbl">FEM</span>
      </button>
      <button class="st-btn" class:active={showLandmarks}
        on:click={() => { showLandmarks = !showLandmarks; onLandmarksToggle(showLandmarks); }}
        title="Landmarks">
        <span class="st-icon">◉</span><span class="st-lbl">LM</span>
      </button>
      <button class="st-btn" class:active={showLabels}
        on:click={() => { showLabels = !showLabels; onLabelsToggle(showLabels); }}>
        <span class="st-icon">Aa</span><span class="st-lbl">Labels</span>
      </button>
      <button class="st-btn" class:active={showIncisions}
        on:click={() => { showIncisions = !showIncisions; onIncisionsToggle(showIncisions); }}>
        <span class="st-icon">╱</span><span class="st-lbl">Cuts</span>
      </button>
      <button class="st-btn" class:active={clipEnabled}
        on:click={() => { clipEnabled = !clipEnabled; onClipToggle(clipEnabled); }}>
        <span class="st-icon">✂</span><span class="st-lbl">Clip</span>
      </button>
    </div>
  </div>

  <div class="st-sep"></div>

  <!-- Scalar field selector -->
  {#if showScalarField}
    <div class="st-group">
      <span class="st-group-label">Field</span>
      <select class="st-select" bind:value={scalarFieldType}
        on:change={() => onScalarTypeChange(scalarFieldType)}>
        {#each scalarOptions as opt}
          <option value={opt.key}>{opt.label}</option>
        {/each}
      </select>
    </div>
    <div class="st-sep"></div>
  {/if}

  <!-- View presets -->
  <div class="st-group">
    <span class="st-group-label">View</span>
    <select class="st-select" bind:value={viewPreset}
      on:change={() => onViewPreset(viewPreset)}>
      {#each viewPresets as vp}
        <option value={vp.key}>{vp.label}</option>
      {/each}
    </select>
    <button class="st-btn" on:click={onResetView} title="Reset (R)">
      <span class="st-icon">⟲</span>
    </button>
  </div>
</div>

<style>
  .st { display: flex; align-items: center; gap: 4px; padding: 4px 8px; flex-wrap: wrap; }

  .st-group { display: flex; align-items: center; gap: 4px; }
  .st-group-label {
    font-size: 9px; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--sov-text-muted, #4B5563); font-weight: 600; padding-right: 2px;
    white-space: nowrap;
  }

  .st-modes, .st-toggles, .st-actions { display: flex; gap: 1px; }

  .st-btn {
    display: flex; align-items: center; gap: 3px;
    padding: 4px 6px; background: transparent; border: 1px solid transparent; border-radius: 3px;
    cursor: pointer; font-family: inherit; color: var(--sov-text-muted, #4B5563);
    transition: all 80ms ease-out; white-space: nowrap;
  }
  .st-btn:hover { background: var(--sov-bg-hover); color: var(--sov-text-secondary); }
  .st-btn.active { background: var(--sov-accent-glow); color: var(--sov-accent); border-color: var(--sov-accent)40; }

  .st-icon { font-size: 12px; }
  .st-lbl { font-size: 10px; }
  .st-key { font-family: 'JetBrains Mono', monospace; font-size: 8px; color: var(--sov-text-muted); opacity: 0; transition: opacity 80ms; }
  .st-btn:hover .st-key { opacity: 1; }

  .st-act-btn {
    padding: 3px 8px; font-size: 10px; font-family: inherit; border-radius: 3px;
    cursor: pointer; border: 1px solid var(--sov-border);
    background: var(--sov-bg-elevated); color: var(--sov-text-secondary);
  }
  .st-act-btn:disabled { opacity: 0.3; cursor: default; }
  .st-act-btn.commit { color: #10B981; border-color: #10B98140; }
  .st-act-btn.commit:not(:disabled):hover { background: rgba(16,185,129,0.12); }
  .st-act-btn.undo { color: #F59E0B; border-color: #F59E0B40; }
  .st-act-btn.cancel { color: #EF4444; border-color: #EF444440; }

  .st-sep { width: 1px; height: 22px; background: var(--sov-border-subtle); margin: 0 4px; }

  .st-select {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    background: var(--sov-bg-elevated); color: var(--sov-text-secondary);
    border: 1px solid var(--sov-border); border-radius: 3px; padding: 3px 6px;
    outline: none; cursor: pointer;
  }
  .st-select:focus { border-color: var(--sov-accent); }
</style>
