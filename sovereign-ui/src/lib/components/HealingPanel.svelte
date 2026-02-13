<script>
  /**
   * HealingPanel — displays healing timeline milestones and edema curve.
   * Shows recovery progression with milestones, edema fraction, and structural integrity.
   */

  /** @type {import('$lib/api-client-ext').HealingTimeline | null} */
  export let healingData = null;

  /** @type {boolean} */
  export let loading = false;

  /** @type {() => void} */
  export let onRunHealing = () => {};

  $: hasData = healingData != null && !healingData.error;
  $: milestones = healingData?.milestones ?? [];
  $: edemaCurve = healingData?.edema_curve ?? [];

  // Build a simple ASCII sparkline for edema curve
  $: sparkline = (() => {
    if (edemaCurve.length === 0) return '';
    const vals = edemaCurve.map(p => p.edema_fraction);
    const maxV = Math.max(...vals);
    const chars = '▁▂▃▄▅▆▇█';
    return vals.map(v => {
      const idx = Math.min(Math.floor((v / (maxV || 1)) * (chars.length - 1)), chars.length - 1);
      return chars[idx];
    }).join('');
  })();
</script>

<div class="heal-panel">
  <div class="heal-header">
    <span class="heal-title">Healing Timeline</span>
  </div>

  {#if !hasData}
    <div class="heal-empty">
      <span class="heal-empty-icon">🩹</span>
      <span>No healing prediction available</span>
      <span class="heal-empty-sub">Generate healing timeline with edema curve and milestones</span>
      <button class="sov-btn sov-btn-primary sov-btn-sm" on:click={onRunHealing} disabled={loading}>
        {loading ? 'Computing…' : 'Generate Healing Timeline'}
      </button>
    </div>
  {:else}
    <!-- Edema sparkline -->
    {#if sparkline}
      <div class="heal-sparkline">
        <span class="heal-spark-label">Edema</span>
        <span class="heal-spark-chars">{sparkline}</span>
        <span class="heal-spark-axis">
          <span>Day 0</span>
          <span>Day {edemaCurve[edemaCurve.length - 1]?.day ?? '—'}</span>
        </span>
      </div>
    {/if}

    <!-- Milestones -->
    <div class="heal-milestones">
      {#each milestones as ms, i}
        <div class="heal-milestone">
          <div class="heal-ms-marker">
            <div class="heal-ms-dot" style="background: {ms.edema_fraction > 0.5 ? '#F59E0B' : ms.edema_fraction > 0.2 ? '#3B82F6' : '#10B981'};">
            </div>
            {#if i < milestones.length - 1}
              <div class="heal-ms-line"></div>
            {/if}
          </div>
          <div class="heal-ms-content">
            <div class="heal-ms-header">
              <span class="heal-ms-day">Day {ms.day}</span>
              <span class="heal-ms-label">{ms.label}</span>
            </div>
            {#if ms.description}
              <span class="heal-ms-desc">{ms.description}</span>
            {/if}
            <div class="heal-ms-metrics">
              <span class="heal-ms-metric">
                Edema: <strong>{((ms.edema_fraction ?? 0) * 100).toFixed(0)}%</strong>
              </span>
              {#if ms.structural_integrity != null}
                <span class="heal-ms-metric">
                  Integrity: <strong>{((ms.structural_integrity ?? 0) * 100).toFixed(0)}%</strong>
                </span>
              {/if}
            </div>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .heal-panel { display: flex; flex-direction: column; gap: 8px; }
  .heal-header { display: flex; justify-content: space-between; align-items: center; }
  .heal-title { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--sov-text-tertiary); }

  .heal-empty { display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 20px 0; color: var(--sov-text-muted); font-size: 12px; text-align: center; }
  .heal-empty-icon { font-size: 24px; opacity: 0.4; }
  .heal-empty-sub { font-size: 10px; }

  .heal-sparkline { padding: 8px 0; border-bottom: 1px solid var(--sov-border-subtle); }
  .heal-spark-label { font-size: 9px; color: var(--sov-text-muted); text-transform: uppercase; display: block; margin-bottom: 4px; }
  .heal-spark-chars { font-size: 14px; letter-spacing: 1px; color: var(--sov-accent); font-family: monospace; display: block; }
  .heal-spark-axis { display: flex; justify-content: space-between; font-size: 8px; color: var(--sov-text-muted); margin-top: 2px; font-family: 'JetBrains Mono', monospace; }

  .heal-milestones { display: flex; flex-direction: column; gap: 0; }
  .heal-milestone { display: flex; gap: 10px; }
  .heal-ms-marker { display: flex; flex-direction: column; align-items: center; min-width: 14px; }
  .heal-ms-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .heal-ms-line { width: 1px; flex: 1; background: var(--sov-border-subtle); margin: 2px 0; }
  .heal-ms-content { padding-bottom: 10px; flex: 1; }
  .heal-ms-header { display: flex; gap: 8px; align-items: baseline; }
  .heal-ms-day { font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 600; color: var(--sov-accent); min-width: 42px; }
  .heal-ms-label { font-size: 11px; font-weight: 500; color: var(--sov-text-primary); }
  .heal-ms-desc { font-size: 10px; color: var(--sov-text-muted); display: block; margin-top: 1px; }
  .heal-ms-metrics { display: flex; gap: 12px; margin-top: 3px; }
  .heal-ms-metric { font-size: 9px; color: var(--sov-text-tertiary); }
  .heal-ms-metric strong { color: var(--sov-text-secondary); font-family: 'JetBrains Mono', monospace; }
</style>
