<script>
  import { onMount } from 'svelte';
  import {
    casesStore,
    activePlan,
    operatorsStore,
    whatIfStore,
    sweepStore,
    loadOperators,
    runWhatIf,
    runSweep,
    operatorSchemas,
  } from '$lib/stores';
  import ParamEditor from '$lib/components/ParamEditor.svelte';

  // ── State ──────────────────────────────────────────────────
  let selectedCaseId = '';
  let activeTab = 'whatif';  // 'whatif' | 'sweep'

  // What-If state
  let selectedOp = '';
  let modifiedParams = {};

  // Sweep state
  let sweepOp = '';
  let sweepParam = '';
  let sweepMin = 0;
  let sweepMax = 10;
  let sweepSteps = 5;

  onMount(async () => {
    await loadOperators();
    if ($casesStore.data?.cases?.[0]) {
      selectedCaseId = $casesStore.data.cases[0].case_id;
    }
  });

  // ── Derived ────────────────────────────────────────────────
  $: plan = $activePlan;
  $: cases = $casesStore.data?.cases ?? [];
  $: operators = $operatorSchemas;
  $: planSteps = plan?.steps?.map(s => s.name || s.operator) ?? [];
  $: hasPlan = plan && plan.n_steps > 0;

  // Selected operator's param_defs
  $: selectedOpSchema = operators[selectedOp] ?? null;
  $: selectedParamDefs = selectedOpSchema?.param_defs ?? {};
  $: sweepParamOptions = Object.entries(selectedParamDefs)
    .filter(([_, d]) => d.param_type === 'float' || d.param_type === 'number' || d.param_type === 'int')
    .map(([k, d]) => ({ key: k, def: d }));

  // Sweep operator's param_defs
  $: sweepOpSchema = operators[sweepOp] ?? null;
  $: sweepParamDefs = sweepOpSchema?.param_defs ?? {};
  $: sweepableParams = Object.entries(sweepParamDefs)
    .filter(([_, d]) => d.param_type === 'float' || d.param_type === 'number' || d.param_type === 'int')
    .map(([k, d]) => ({ key: k, def: d }));

  // Auto-set sweep bounds from param_defs
  $: {
    if (sweepParam && sweepOpSchema?.param_defs?.[sweepParam]) {
      const def = sweepOpSchema.param_defs[sweepParam];
      if (def.min_value != null) sweepMin = def.min_value;
      if (def.max_value != null) sweepMax = def.max_value;
    }
  }

  // ── Actions ────────────────────────────────────────────────
  async function handleWhatIf() {
    if (!selectedCaseId || !selectedOp) return;
    await runWhatIf(selectedCaseId, { [selectedOp]: modifiedParams });
  }

  async function handleSweep() {
    if (!selectedCaseId || !sweepOp || !sweepParam) return;
    const step = (sweepMax - sweepMin) / Math.max(sweepSteps - 1, 1);
    const values = Array.from({ length: sweepSteps }, (_, i) =>
      Math.round((sweepMin + i * step) * 100) / 100
    );
    await runSweep(selectedCaseId, sweepOp, sweepParam, values);
  }

  function formatName(n) {
    if (!n) return '';
    return n.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  // ── Sweep chart ────────────────────────────────────────────
  $: sweepData = $sweepStore.data;
  $: chartPoints = sweepData?.results?.map(r => ({
    x: r.value,
    y: typeof r.result === 'object'
      ? Object.values(r.result).find(v => typeof v === 'number') ?? 0
      : r.result ?? 0,
  })) ?? [];

  $: chartXMin = chartPoints.length > 0 ? Math.min(...chartPoints.map(p => p.x)) : 0;
  $: chartXMax = chartPoints.length > 0 ? Math.max(...chartPoints.map(p => p.x)) : 1;
  $: chartYMin = chartPoints.length > 0 ? Math.min(...chartPoints.map(p => p.y)) : 0;
  $: chartYMax = chartPoints.length > 0 ? Math.max(...chartPoints.map(p => p.y)) : 1;

  function chartX(val) {
    if (chartXMax === chartXMin) return 50;
    return ((val - chartXMin) / (chartXMax - chartXMin)) * 100;
  }
  function chartY(val) {
    if (chartYMax === chartYMin) return 50;
    return 100 - ((val - chartYMin) / (chartYMax - chartYMin)) * 100;
  }
  $: svgPath = chartPoints.length > 1
    ? chartPoints.map((p, i) =>
        `${i === 0 ? 'M' : 'L'} ${chartX(p.x) * 4.6 + 40} ${chartY(p.y) * 1.8 + 10}`
      ).join(' ')
    : '';
</script>

<div class="sov-page-header">
  <h1 class="sov-page-title">What-If Console</h1>
  <p class="sov-page-subtitle">Parameter exploration and sensitivity analysis</p>
</div>

<!-- Pre-check -->
{#if !hasPlan}
  <div class="sov-error-banner" style="border-color: var(--sov-warning); background: #F59E0B08; color: var(--sov-warning);">
    <span>⚠</span>
    <span>No active plan. Go to <a href="/plan" style="color: var(--sov-accent);">Plan Editor</a> to build one first.</span>
  </div>
{/if}

<!-- Toolbar -->
<div class="sov-toolbar">
  <select class="sov-select" style="width: 220px;" bind:value={selectedCaseId}>
    <option value="">Select case...</option>
    {#each cases as c}
      <option value={c.case_id}>{c.case_id.substring(0, 12)}...</option>
    {/each}
  </select>

  <div class="sov-toolbar-spacer"></div>

  <button class="sov-btn sov-btn-sm" class:sov-btn-primary={activeTab === 'whatif'}
    class:sov-btn-secondary={activeTab !== 'whatif'}
    on:click={() => activeTab = 'whatif'}>What-If</button>
  <button class="sov-btn sov-btn-sm" class:sov-btn-primary={activeTab === 'sweep'}
    class:sov-btn-secondary={activeTab !== 'sweep'}
    on:click={() => activeTab = 'sweep'}>Sweep</button>
</div>

{#if activeTab === 'whatif'}
  <!-- What-If Panel -->
  <div class="consult-layout">
    <div class="sov-card">
      <div class="sov-card-header">
        <span class="sov-card-title">Parameter Override</span>
      </div>
      <div class="sov-card-body">
        <div style="margin-bottom: 12px;">
          <label class="sov-label">Operator to modify</label>
          <select class="sov-select" style="width: 100%;" bind:value={selectedOp}>
            <option value="">Choose operator...</option>
            {#each planSteps as step}
              <option value={step}>{formatName(step)}</option>
            {/each}
          </select>
        </div>

        {#if selectedOp && selectedParamDefs && Object.keys(selectedParamDefs).length > 0}
          <ParamEditor
            paramDefs={selectedParamDefs}
            bind:values={modifiedParams}
          />
          <div style="margin-top: 12px; text-align: right;">
            <button class="sov-btn sov-btn-primary"
              disabled={!selectedCaseId || !selectedOp || $whatIfStore.loading}
              on:click={handleWhatIf}>
              {$whatIfStore.loading ? 'Running...' : '⚡ Run What-If'}
            </button>
          </div>
        {:else if selectedOp}
          <p class="text-muted" style="font-size: 12px;">No configurable parameters for this operator.</p>
        {/if}
      </div>
    </div>

    <div class="sov-card">
      <div class="sov-card-header">
        <span class="sov-card-title">Results</span>
      </div>
      <div class="sov-card-body">
        {#if $whatIfStore.loading}
          <div class="sov-loading"><div class="sov-spinner"></div></div>
        {:else if $whatIfStore.error}
          <div class="sov-error-banner"><span>⚠</span><span>{$whatIfStore.error}</span></div>
        {:else if $whatIfStore.data}
          <div class="result-section">
            <span class="sov-label">Scenario</span>
            <span class="font-data" style="font-size: 12px; color: var(--sov-accent);">
              {$whatIfStore.data.scenario}
            </span>
          </div>
          <div class="result-section">
            <span class="sov-label">Modified Operators</span>
            <div style="display: flex; gap: 4px; flex-wrap: wrap;">
              {#each $whatIfStore.data.modified_operators ?? [] as op}
                <span class="sov-badge sov-badge-accent">{formatName(op)}</span>
              {/each}
            </div>
          </div>
          <div class="result-section">
            <span class="sov-label">Result</span>
            <div class="compile-result-grid">
              {#each Object.entries($whatIfStore.data.result ?? {}) as [key, val]}
                <div class="compile-result-item">
                  <span class="compile-result-label">{formatName(key)}</span>
                  <span class="compile-result-value font-data">
                    {typeof val === 'object' ? JSON.stringify(val).substring(0, 50) : val}
                  </span>
                </div>
              {/each}
            </div>
          </div>
        {:else}
          <div class="sov-empty" style="padding: 24px;">
            <p style="font-size: 12px;">Select an operator, modify parameters, and run.</p>
          </div>
        {/if}
      </div>
    </div>
  </div>

{:else}
  <!-- Sweep Panel -->
  <div class="consult-layout">
    <div class="sov-card">
      <div class="sov-card-header">
        <span class="sov-card-title">Sweep Configuration</span>
      </div>
      <div class="sov-card-body">
        <div style="margin-bottom: 12px;">
          <label class="sov-label">Operator</label>
          <select class="sov-select" style="width: 100%;" bind:value={sweepOp}>
            <option value="">Choose operator...</option>
            {#each planSteps as step}
              <option value={step}>{formatName(step)}</option>
            {/each}
          </select>
        </div>

        {#if sweepOp && sweepableParams.length > 0}
          <div style="margin-bottom: 12px;">
            <label class="sov-label">Parameter to sweep</label>
            <select class="sov-select" style="width: 100%;" bind:value={sweepParam}>
              <option value="">Choose parameter...</option>
              {#each sweepableParams as { key, def }}
                <option value={key}>{formatName(def.name || key)} ({def.unit || '—'})</option>
              {/each}
            </select>
          </div>

          {#if sweepParam}
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 12px;">
              <div>
                <label class="sov-label">Min</label>
                <input class="sov-input" type="number" step="0.1" bind:value={sweepMin} />
              </div>
              <div>
                <label class="sov-label">Max</label>
                <input class="sov-input" type="number" step="0.1" bind:value={sweepMax} />
              </div>
              <div>
                <label class="sov-label">Steps</label>
                <input class="sov-input" type="number" min="2" max="20" bind:value={sweepSteps} />
              </div>
            </div>
          {/if}

          <div style="text-align: right;">
            <button class="sov-btn sov-btn-primary"
              disabled={!selectedCaseId || !sweepParam || $sweepStore.loading}
              on:click={handleSweep}>
              {$sweepStore.loading ? 'Sweeping...' : '⚡ Run Sweep'}
            </button>
          </div>
        {:else if sweepOp}
          <p class="text-muted" style="font-size: 12px;">No numeric parameters to sweep.</p>
        {/if}
      </div>
    </div>

    <div class="sov-card">
      <div class="sov-card-header">
        <span class="sov-card-title">Sweep Results</span>
        {#if sweepData}
          <span class="sov-badge sov-badge-default">{sweepData.n_points} points</span>
        {/if}
      </div>
      <div class="sov-card-body">
        {#if $sweepStore.loading}
          <div class="sov-loading"><div class="sov-spinner"></div></div>
        {:else if $sweepStore.error}
          <div class="sov-error-banner"><span>⚠</span><span>{$sweepStore.error}</span></div>
        {:else if sweepData && chartPoints.length > 0}
          <!-- SVG Chart -->
          <div class="sweep-chart-wrap">
            <svg viewBox="0 0 500 200" class="sweep-chart">
              <!-- Grid lines -->
              {#each [0, 25, 50, 75, 100] as pct}
                <line x1="40" y1={pct * 1.8 + 10} x2="500" y2={pct * 1.8 + 10}
                  stroke="#1F2937" stroke-width="0.5" />
              {/each}

              <!-- Axis labels -->
              <text x="4" y="14" fill="#4B5563" font-size="8" font-family="JetBrains Mono">
                {chartYMax.toFixed(1)}
              </text>
              <text x="4" y="192" fill="#4B5563" font-size="8" font-family="JetBrains Mono">
                {chartYMin.toFixed(1)}
              </text>

              <!-- Line -->
              {#if svgPath}
                <path d={svgPath} fill="none" stroke="#3B82F6" stroke-width="2"
                  stroke-linecap="round" stroke-linejoin="round" />
              {/if}

              <!-- Points -->
              {#each chartPoints as p}
                <circle cx={chartX(p.x) * 4.6 + 40} cy={chartY(p.y) * 1.8 + 10}
                  r="4" fill="#3B82F6" stroke="#08080A" stroke-width="1.5" />
              {/each}

              <!-- X axis labels -->
              {#each chartPoints as p, i}
                {#if i === 0 || i === chartPoints.length - 1 || i === Math.floor(chartPoints.length / 2)}
                  <text x={chartX(p.x) * 4.6 + 40} y="198" fill="#4B5563" font-size="8"
                    font-family="JetBrains Mono" text-anchor="middle">
                    {p.x.toFixed(1)}
                  </text>
                {/if}
              {/each}
            </svg>
          </div>

          <!-- Data table -->
          <table class="sov-table" style="font-size: 11px; margin-top: 12px;">
            <thead>
              <tr>
                <th>{formatName(sweepData.sweep_param)}</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {#each sweepData.results as r}
                <tr>
                  <td class="font-data">{r.value}</td>
                  <td class="font-data">
                    {typeof r.result === 'object' ? JSON.stringify(r.result).substring(0, 60) : r.result}
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        {:else}
          <div class="sov-empty" style="padding: 24px;">
            <p style="font-size: 12px;">Configure sweep parameters and run.</p>
          </div>
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  .consult-layout {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }

  .result-section {
    margin-bottom: 12px;
  }

  .compile-result-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 6px;
    margin-top: 4px;
  }

  .compile-result-item {
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .compile-result-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--sov-text-muted, #4B5563);
  }

  .compile-result-value {
    font-size: 12px;
    color: var(--sov-text-primary, #E8E8EC);
  }

  .sweep-chart-wrap {
    background: var(--sov-bg-root, #08080A);
    border: 1px solid var(--sov-border-subtle, #151820);
    border-radius: 4px;
    padding: 8px;
  }

  .sweep-chart {
    width: 100%;
    height: auto;
  }

  @media (max-width: 900px) {
    .consult-layout { grid-template-columns: 1fr; }
  }
</style>
