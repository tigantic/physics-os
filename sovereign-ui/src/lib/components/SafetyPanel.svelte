<script>
  /**
   * SafetyPanel — displays surgical safety evaluation metrics.
   * Skin tension, vascular risk, structural integrity, and overall safety index.
   */

  /** @type {import('$lib/api-client-ext').SafetyReport | null} */
  export let safetyData = null;

  /** @type {boolean} */
  export let loading = false;

  /** @type {() => void} */
  export let onRunSafety = () => {};

  $: hasData = safetyData != null && !safetyData.error;
  $: safetyIndex = safetyData?.overall_safety_index ?? 0;
  $: isSafe = safetyData?.is_safe ?? false;
  $: skinTension = safetyData?.skin_tension ?? null;
  $: vascularRisk = safetyData?.vascular_risk ?? null;
  $: structural = safetyData?.structural_integrity ?? null;
  $: violations = safetyData?.stress_violations ?? [];

  function safetyColor(index) {
    if (index >= 90) return '#10B981';
    if (index >= 70) return '#F59E0B';
    return '#EF4444';
  }
</script>

<div class="safety-panel">
  <div class="safety-header">
    <span class="safety-title">Safety Evaluation</span>
    {#if hasData}
      <span class="safety-badge" style="background: {safetyColor(safetyIndex)}20; color: {safetyColor(safetyIndex)}; border: 1px solid {safetyColor(safetyIndex)}40;">
        {isSafe ? '✓ SAFE' : '⚠ UNSAFE'}
      </span>
    {/if}
  </div>

  {#if !hasData}
    <div class="safety-empty">
      <span class="safety-empty-icon">🛡️</span>
      <span>No safety evaluation available</span>
      <span class="safety-empty-sub">Run safety check to evaluate surgical plan viability</span>
      <button class="sov-btn sov-btn-primary sov-btn-sm" on:click={onRunSafety} disabled={loading}>
        {loading ? 'Evaluating…' : 'Run Safety Check'}
      </button>
    </div>
  {:else}
    <!-- Safety index gauge -->
    <div class="safety-gauge">
      <div class="safety-gauge-track">
        <div class="safety-gauge-fill" style="width: {safetyIndex}%; background: {safetyColor(safetyIndex)};"></div>
      </div>
      <div class="safety-gauge-label">
        <span class="safety-gauge-val" style="color: {safetyColor(safetyIndex)};">{safetyIndex.toFixed(1)}</span>
        <span class="safety-gauge-unit">/ 100</span>
      </div>
    </div>

    <!-- Skin tension -->
    {#if skinTension}
      <div class="safety-section">
        <div class="safety-section-header">
          <span class="safety-section-label">Skin Tension</span>
          <span class="safety-indicator" class:safe={skinTension.safe} class:unsafe={!skinTension.safe}>
            {skinTension.safe ? '✓' : '⚠'}
          </span>
        </div>
        <div class="safety-metrics">
          <div class="safety-metric">
            <span class="safety-metric-label">Max</span>
            <span class="safety-metric-val">{skinTension.max_tension_pa?.toFixed(0) ?? '—'} Pa</span>
          </div>
          <div class="safety-metric">
            <span class="safety-metric-label">Mean</span>
            <span class="safety-metric-val">{skinTension.mean_tension_pa?.toFixed(0) ?? '—'} Pa</span>
          </div>
          <div class="safety-metric">
            <span class="safety-metric-label">Threshold</span>
            <span class="safety-metric-val">{skinTension.threshold_pa?.toFixed(0) ?? '—'} Pa</span>
          </div>
        </div>
      </div>
    {/if}

    <!-- Vascular risk -->
    {#if vascularRisk}
      <div class="safety-section">
        <div class="safety-section-header">
          <span class="safety-section-label">Vascular Risk</span>
          <span class="safety-indicator" class:safe={vascularRisk.safe} class:unsafe={!vascularRisk.safe}>
            {vascularRisk.safe ? '✓' : '⚠'}
          </span>
        </div>
        <div class="safety-metrics">
          <div class="safety-metric">
            <span class="safety-metric-label">Risk Level</span>
            <span class="safety-metric-val" style="text-transform: capitalize;">{vascularRisk.risk_level ?? '—'}</span>
          </div>
          <div class="safety-metric">
            <span class="safety-metric-label">Max Depth</span>
            <span class="safety-metric-val">{vascularRisk.max_depth_mm?.toFixed(1) ?? '—'} mm</span>
          </div>
        </div>
      </div>
    {/if}

    <!-- Structural integrity -->
    {#if structural}
      <div class="safety-section">
        <div class="safety-section-header">
          <span class="safety-section-label">Structural Integrity</span>
          <span class="safety-indicator" class:safe={structural.safe} class:unsafe={!structural.safe}>
            {structural.safe ? '✓' : '⚠'}
          </span>
        </div>
        <div class="safety-metrics">
          <div class="safety-metric">
            <span class="safety-metric-label">Min Thickness</span>
            <span class="safety-metric-val">{structural.min_thickness_mm?.toFixed(1) ?? '—'} mm</span>
          </div>
        </div>
      </div>
    {/if}

    <!-- Stress violations -->
    {#if violations.length > 0}
      <div class="safety-section">
        <span class="safety-section-label" style="color: #EF4444;">Stress Violations ({violations.length})</span>
        {#each violations as v}
          <div class="safety-violation">
            <span class="safety-violation-region">{v.region?.replace(/_/g, ' ') ?? '—'}</span>
            <span class="safety-violation-val">{v.value?.toFixed(1)} / {v.threshold?.toFixed(1)}</span>
          </div>
        {/each}
      </div>
    {/if}
  {/if}
</div>

<style>
  .safety-panel { display: flex; flex-direction: column; gap: 8px; }
  .safety-header { display: flex; justify-content: space-between; align-items: center; }
  .safety-title { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--sov-text-tertiary); }
  .safety-badge { padding: 2px 8px; border-radius: 3px; font-size: 10px; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

  .safety-empty { display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 20px 0; color: var(--sov-text-muted); font-size: 12px; text-align: center; }
  .safety-empty-icon { font-size: 24px; opacity: 0.4; }
  .safety-empty-sub { font-size: 10px; }

  .safety-gauge { padding: 8px 0; }
  .safety-gauge-track { height: 8px; background: var(--sov-bg-elevated); border-radius: 4px; overflow: hidden; }
  .safety-gauge-fill { height: 100%; border-radius: 4px; transition: width 600ms ease-out; }
  .safety-gauge-label { display: flex; align-items: baseline; gap: 4px; margin-top: 4px; }
  .safety-gauge-val { font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 700; }
  .safety-gauge-unit { font-size: 11px; color: var(--sov-text-muted); }

  .safety-section { padding-top: 6px; border-top: 1px solid var(--sov-border-subtle); }
  .safety-section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
  .safety-section-label { font-size: 10px; font-weight: 600; color: var(--sov-text-muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .safety-indicator { font-size: 11px; }
  .safety-indicator.safe { color: #10B981; }
  .safety-indicator.unsafe { color: #EF4444; }

  .safety-metrics { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; }
  .safety-metric { display: flex; flex-direction: column; gap: 1px; }
  .safety-metric-label { font-size: 9px; color: var(--sov-text-muted); text-transform: uppercase; }
  .safety-metric-val { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--sov-text-primary); }

  .safety-violation { display: flex; justify-content: space-between; padding: 3px 0; font-size: 10px; }
  .safety-violation-region { color: var(--sov-text-secondary); text-transform: capitalize; }
  .safety-violation-val { font-family: 'JetBrains Mono', monospace; color: #EF4444; }
</style>
