<script>
  import { onMount } from 'svelte';
  import {
    casesStore,
    compareStore,
    comparePlans,
    compareCases,
  } from '$lib/stores';

  // ── State ──────────────────────────────────────────────────
  let activeMode = 'plans';  // 'plans' | 'cases'

  // Plan compare
  let planIdA = '';
  let planIdB = '';

  // Case compare
  let caseIdA = '';
  let caseIdB = '';

  onMount(() => {
    const c = $casesStore.data?.cases ?? [];
    if (c.length >= 2) {
      caseIdA = c[0].case_id;
      caseIdB = c[1].case_id;
    } else if (c.length === 1) {
      caseIdA = c[0].case_id;
    }
  });

  $: cases = $casesStore.data?.cases ?? [];
  $: result = $compareStore.data;

  async function handleComparePlans() {
    if (!planIdA || !planIdB) return;
    await comparePlans(planIdA, planIdB);
  }

  async function handleCompareCases() {
    if (!caseIdA || !caseIdB) return;
    await compareCases(caseIdA, caseIdB);
  }

  function formatName(n) {
    if (!n) return '';
    return n.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  function truncId(id) {
    return id ? id.substring(0, 10) + '...' : '';
  }

  // Flatten comparison result for display
  function flattenResult(obj, prefix = '') {
    const items = [];
    if (!obj || typeof obj !== 'object') return items;
    for (const [key, val] of Object.entries(obj)) {
      const fullKey = prefix ? `${prefix}.${key}` : key;
      if (val && typeof val === 'object' && !Array.isArray(val)) {
        items.push(...flattenResult(val, fullKey));
      } else {
        items.push({ key: fullKey, value: val });
      }
    }
    return items;
  }

  $: flatResults = result ? flattenResult(result) : [];
</script>

<div class="sov-page-header">
  <h1 class="sov-page-title">Compare</h1>
  <p class="sov-page-subtitle">Side-by-side plan and case comparison</p>
</div>

<!-- Mode toggle -->
<div class="sov-toolbar">
  <button class="sov-btn sov-btn-sm"
    class:sov-btn-primary={activeMode === 'plans'}
    class:sov-btn-secondary={activeMode !== 'plans'}
    on:click={() => activeMode = 'plans'}>
    Plan vs Plan
  </button>
  <button class="sov-btn sov-btn-sm"
    class:sov-btn-primary={activeMode === 'cases'}
    class:sov-btn-secondary={activeMode !== 'cases'}
    on:click={() => activeMode = 'cases'}>
    Case vs Case
  </button>
</div>

{#if activeMode === 'plans'}
  <!-- Plan Compare -->
  <div class="compare-config">
    <div class="sov-card compare-side">
      <div class="sov-card-header">
        <span class="sov-card-title" style="color: var(--sov-accent);">Plan A</span>
      </div>
      <div class="sov-card-body">
        <label class="sov-label">Plan ID or Content Hash</label>
        <input class="sov-input" type="text"
          placeholder="Enter plan ID..."
          bind:value={planIdA} />
        <p class="text-muted" style="font-size: 10px; margin-top: 6px;">
          Use content hash from Plan Editor compilation
        </p>
      </div>
    </div>

    <div class="compare-vs">
      <span class="compare-vs-text">VS</span>
    </div>

    <div class="sov-card compare-side">
      <div class="sov-card-header">
        <span class="sov-card-title" style="color: var(--sov-warning);">Plan B</span>
      </div>
      <div class="sov-card-body">
        <label class="sov-label">Plan ID or Content Hash</label>
        <input class="sov-input" type="text"
          placeholder="Enter plan ID..."
          bind:value={planIdB} />
      </div>
    </div>
  </div>

  <div style="text-align: center; margin: 16px 0;">
    <button class="sov-btn sov-btn-primary"
      disabled={!planIdA || !planIdB || $compareStore.loading}
      on:click={handleComparePlans}>
      {$compareStore.loading ? 'Comparing...' : '⟺ Compare Plans'}
    </button>
  </div>

{:else}
  <!-- Case Compare -->
  <div class="compare-config">
    <div class="sov-card compare-side">
      <div class="sov-card-header">
        <span class="sov-card-title" style="color: var(--sov-accent);">Case A</span>
      </div>
      <div class="sov-card-body">
        <label class="sov-label">Case</label>
        <select class="sov-select" style="width: 100%;" bind:value={caseIdA}>
          <option value="">Select case...</option>
          {#each cases as c}
            <option value={c.case_id}>
              {truncId(c.case_id)} — {formatName(c.procedure_type)}
            </option>
          {/each}
        </select>
      </div>
    </div>

    <div class="compare-vs">
      <span class="compare-vs-text">VS</span>
    </div>

    <div class="sov-card compare-side">
      <div class="sov-card-header">
        <span class="sov-card-title" style="color: var(--sov-warning);">Case B</span>
      </div>
      <div class="sov-card-body">
        <label class="sov-label">Case</label>
        <select class="sov-select" style="width: 100%;" bind:value={caseIdB}>
          <option value="">Select case...</option>
          {#each cases.filter(c => c.case_id !== caseIdA) as c}
            <option value={c.case_id}>
              {truncId(c.case_id)} — {formatName(c.procedure_type)}
            </option>
          {/each}
        </select>
      </div>
    </div>
  </div>

  <div style="text-align: center; margin: 16px 0;">
    <button class="sov-btn sov-btn-primary"
      disabled={!caseIdA || !caseIdB || $compareStore.loading}
      on:click={handleCompareCases}>
      {$compareStore.loading ? 'Comparing...' : '⟺ Compare Cases'}
    </button>
  </div>
{/if}

<!-- Error -->
{#if $compareStore.error}
  <div class="sov-error-banner">
    <span>⚠</span><span>{$compareStore.error}</span>
  </div>
{/if}

<!-- Results -->
{#if $compareStore.loading}
  <div class="sov-loading" style="padding: 32px;">
    <div class="sov-spinner"></div>
    <span>Running comparison...</span>
  </div>
{:else if result}
  <div class="sov-card">
    <div class="sov-card-header">
      <span class="sov-card-title">Comparison Results</span>
      <span class="sov-badge sov-badge-default">{flatResults.length} fields</span>
    </div>
    <div class="sov-card-body">
      {#if flatResults.length > 0}
        <div class="sov-table-wrap">
          <table class="sov-table">
            <thead>
              <tr>
                <th>Field</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {#each flatResults as item}
                <tr>
                  <td style="white-space: nowrap;">
                    <span class="font-data" style="font-size: 11px; color: var(--sov-accent);">
                      {item.key}
                    </span>
                  </td>
                  <td>
                    <span class="font-data" style="font-size: 11px;">
                      {#if Array.isArray(item.value)}
                        [{item.value.map(v => typeof v === 'object' ? JSON.stringify(v) : v).join(', ')}]
                      {:else if typeof item.value === 'object' && item.value !== null}
                        {JSON.stringify(item.value).substring(0, 80)}
                      {:else if typeof item.value === 'boolean'}
                        <span class="sov-badge" class:sov-badge-success={item.value} class:sov-badge-error={!item.value}>
                          {item.value ? 'Yes' : 'No'}
                        </span>
                      {:else if typeof item.value === 'number'}
                        <span style="color: var(--sov-text-primary);">{item.value}</span>
                      {:else}
                        {item.value}
                      {/if}
                    </span>
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {:else}
        <div class="sov-empty" style="padding: 24px;">
          <div class="sov-empty-title">Empty Result</div>
          <p style="font-size: 12px;">Comparison returned no diff fields.</p>
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  .compare-config {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 16px;
    align-items: start;
  }

  .compare-side {
    min-width: 0;
  }

  .compare-vs {
    display: flex;
    align-items: center;
    justify-content: center;
    padding-top: 40px;
  }

  .compare-vs-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 700;
    color: var(--sov-text-muted, #4B5563);
    background: var(--sov-bg-elevated, #161920);
    padding: 6px 12px;
    border-radius: 4px;
    border: 1px solid var(--sov-border, #1F2937);
  }

  @media (max-width: 700px) {
    .compare-config {
      grid-template-columns: 1fr;
    }
    .compare-vs { padding: 8px 0; }
  }
</style>
