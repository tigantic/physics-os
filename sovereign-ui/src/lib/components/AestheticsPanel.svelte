<script>
  /**
   * AestheticsPanel — displays aesthetic angle measurements and proportional analysis.
   * Nasofrontal angle, nasolabial angle, Goode ratio, symmetry, dorsal line deviation.
   */

  /** @type {import('$lib/api-client-ext').AestheticsReport | null} */
  export let aestheticsData = null;

  /** @type {boolean} */
  export let loading = false;

  /** @type {() => void} */
  export let onRunAesthetics = () => {};

  $: hasData = aestheticsData != null && !aestheticsData.error;

  // Ideal ranges for rhinoplasty aesthetics
  const IDEALS = {
    nasofrontal_angle: { min: 115, max: 130, unit: '°', label: 'Nasofrontal Angle' },
    nasolabial_angle:  { min: 90,  max: 110, unit: '°', label: 'Nasolabial Angle' },
    goode_ratio:       { min: 0.55, max: 0.60, unit: '', label: 'Goode Ratio' },
    dorsal_line_deviation: { min: 0, max: 2, unit: 'mm', label: 'Dorsal Deviation' },
    symmetry_score:    { min: 0.85, max: 1.0, unit: '', label: 'Symmetry' },
  };

  function inRange(key, val) {
    const ideal = IDEALS[key];
    if (!ideal || val == null) return 'unknown';
    return val >= ideal.min && val <= ideal.max ? 'optimal' : 'suboptimal';
  }

  function formatVal(key, val) {
    if (val == null) return '—';
    const ideal = IDEALS[key];
    const precision = key === 'goode_ratio' || key === 'symmetry_score' ? 2 : 1;
    return val.toFixed(precision) + (ideal?.unit ?? '');
  }
</script>

<div class="aes-panel">
  <div class="aes-header">
    <span class="aes-title">Aesthetic Analysis</span>
  </div>

  {#if !hasData}
    <div class="aes-empty">
      <span class="aes-empty-icon">📐</span>
      <span>No aesthetic analysis available</span>
      <span class="aes-empty-sub">Evaluate facial angles and proportions</span>
      <button class="sov-btn sov-btn-primary sov-btn-sm" on:click={onRunAesthetics} disabled={loading}>
        {loading ? 'Analyzing…' : 'Run Aesthetic Analysis'}
      </button>
    </div>
  {:else}
    <!-- Overall aesthetic score -->
    {#if aestheticsData.overall_aesthetic_score != null}
      <div class="aes-overall">
        <span class="aes-overall-val">{aestheticsData.overall_aesthetic_score.toFixed(1)}</span>
        <span class="aes-overall-label">Aesthetic Score</span>
      </div>
    {/if}

    <!-- Individual metrics -->
    <div class="aes-metrics">
      {#each Object.entries(IDEALS) as [key, meta]}
        {@const val = aestheticsData[key]}
        {@const status = inRange(key, val)}
        <div class="aes-metric" class:optimal={status === 'optimal'} class:suboptimal={status === 'suboptimal'}>
          <div class="aes-metric-header">
            <span class="aes-metric-label">{meta.label}</span>
            <span class="aes-metric-indicator" class:optimal={status === 'optimal'} class:suboptimal={status === 'suboptimal'}>
              {status === 'optimal' ? '✓' : status === 'suboptimal' ? '△' : '—'}
            </span>
          </div>
          <div class="aes-metric-row">
            <span class="aes-metric-val">{formatVal(key, val)}</span>
            <span class="aes-metric-range">ideal: {meta.min}–{meta.max}{meta.unit}</span>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .aes-panel { display: flex; flex-direction: column; gap: 8px; }
  .aes-header { display: flex; justify-content: space-between; align-items: center; }
  .aes-title { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--sov-text-tertiary); }

  .aes-empty { display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 20px 0; color: var(--sov-text-muted); font-size: 12px; text-align: center; }
  .aes-empty-icon { font-size: 24px; opacity: 0.4; }
  .aes-empty-sub { font-size: 10px; }

  .aes-overall { text-align: center; padding: 12px 0 8px; }
  .aes-overall-val { font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 700; color: var(--sov-accent); display: block; }
  .aes-overall-label { font-size: 10px; color: var(--sov-text-muted); text-transform: uppercase; letter-spacing: 0.04em; }

  .aes-metrics { display: flex; flex-direction: column; gap: 6px; }
  .aes-metric { padding: 6px 8px; background: var(--sov-bg-root); border: 1px solid var(--sov-border-subtle); border-radius: 4px; }
  .aes-metric.optimal { border-left: 2px solid #10B981; }
  .aes-metric.suboptimal { border-left: 2px solid #F59E0B; }
  .aes-metric-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px; }
  .aes-metric-label { font-size: 10px; font-weight: 600; color: var(--sov-text-secondary); }
  .aes-metric-indicator { font-size: 11px; }
  .aes-metric-indicator.optimal { color: #10B981; }
  .aes-metric-indicator.suboptimal { color: #F59E0B; }
  .aes-metric-row { display: flex; justify-content: space-between; align-items: baseline; }
  .aes-metric-val { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; color: var(--sov-text-primary); }
  .aes-metric-range { font-size: 9px; color: var(--sov-text-muted); font-family: 'JetBrains Mono', monospace; }
</style>
