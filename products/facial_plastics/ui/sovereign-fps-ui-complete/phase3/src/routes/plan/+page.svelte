<script>
  import { onMount } from 'svelte';
  import {
    contractStore,
    casesStore,
    operatorsStore,
    templatesStore,
    activePlan,
    compileResultStore,
    loadOperators,
    loadTemplates,
    createFromTemplate,
    createCustom,
    compilePlan,
    procedureTypes,
    operatorSchemas,
    templateRegistry,
  } from '$lib/stores';
  import OperatorPalette from '$lib/components/OperatorPalette.svelte';
  import PlanSteps from '$lib/components/PlanSteps.svelte';

  // ── State ──────────────────────────────────────────────────
  let selectedProcedure = '';
  let searchText = '';
  let selectedCaseId = '';
  let planName = 'Untitled Plan';
  let showTemplateModal = false;

  // ── Load data ──────────────────────────────────────────────
  onMount(async () => {
    await Promise.all([
      loadOperators(),
      loadTemplates(),
    ]);
    // Default to first procedure
    if ($procedureTypes.length > 0 && !selectedProcedure) {
      selectedProcedure = $procedureTypes[0];
    }
    // Default to first case
    if ($casesStore.data?.cases?.length > 0 && !selectedCaseId) {
      selectedCaseId = $casesStore.data.cases[0].case_id;
    }
  });

  // Reload operators when procedure changes
  $: if (selectedProcedure) {
    loadOperators(selectedProcedure);
  }

  // ── Derived ────────────────────────────────────────────────
  $: operators = $operatorSchemas;
  $: templates = $templateRegistry;
  $: plan = $activePlan;
  $: compileResult = $compileResultStore.data;
  $: cases = $casesStore.data?.cases ?? [];
  $: procedures = $procedureTypes;

  // Group templates by category
  $: templateEntries = Object.entries(templates);

  // ── Actions ────────────────────────────────────────────────
  function handleAddOperator(e) {
    const { operator, params } = e.detail;
    const opSchema = operators[operator];

    if (!plan) {
      // Create new custom plan with this operator
      createCustom(planName, selectedProcedure, [{ operator, params }]);
    } else {
      // Add step to existing plan — update activePlan directly
      const newSteps = [
        ...(plan.steps || []),
        {
          name: operator,
          operator,
          category: opSchema?.category || selectedProcedure,
          params: params || {},
          affected_structures: opSchema?.affected_structures || [],
          param_defs: opSchema?.param_defs || {},
        },
      ];
      $activePlan = {
        ...plan,
        steps: newSteps,
        n_steps: newSteps.length,
        content_hash: '', // Stale until recompile
      };
    }
  }

  async function handleSelectTemplate(category, template) {
    await createFromTemplate(category, template);
    showTemplateModal = false;
  }

  function handleReorder(e) {
    if (!plan?.steps) return;
    const { from, to } = e.detail;
    const steps = [...plan.steps];
    const [moved] = steps.splice(from, 1);
    steps.splice(to, 0, moved);
    $activePlan = { ...plan, steps, n_steps: steps.length, content_hash: '' };
  }

  function handleRemoveStep(e) {
    if (!plan?.steps) return;
    const { index } = e.detail;
    const steps = plan.steps.filter((_, i) => i !== index);
    $activePlan = { ...plan, steps, n_steps: steps.length, content_hash: '' };
  }

  async function handleCompile() {
    if (!selectedCaseId) return;
    await compilePlan(selectedCaseId);
  }

  function handleNewPlan() {
    $activePlan = null;
    planName = 'Untitled Plan';
  }

  function formatName(n) {
    if (!n) return '';
    return n.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }
</script>

<!-- Page Header -->
<div class="sov-page-header">
  <h1 class="sov-page-title">Surgical Plan Editor</h1>
  <p class="sov-page-subtitle">Build and compile surgical plans from parameterized operators</p>
</div>

<!-- Toolbar -->
<div class="sov-toolbar">
  <div style="display: flex; align-items: center; gap: 8px;">
    <label class="sov-label" style="margin: 0; white-space: nowrap;">Case</label>
    <select class="sov-select" style="width: 220px;" bind:value={selectedCaseId}>
      <option value="">Select case...</option>
      {#each cases as c}
        <option value={c.case_id}>
          {c.case_id.substring(0, 12)}... — {formatName(c.procedure_type)}
        </option>
      {/each}
    </select>
  </div>

  <div style="display: flex; align-items: center; gap: 8px;">
    <label class="sov-label" style="margin: 0; white-space: nowrap;">Procedure</label>
    <select class="sov-select" style="width: 180px;" bind:value={selectedProcedure}>
      {#each procedures as proc}
        <option value={proc}>{formatName(proc)}</option>
      {/each}
    </select>
  </div>

  <div class="sov-toolbar-spacer"></div>

  <button class="sov-btn sov-btn-secondary"
    on:click={() => showTemplateModal = true}>
    ◈ Templates
  </button>

  <button class="sov-btn sov-btn-secondary"
    on:click={handleNewPlan}>
    + New Plan
  </button>
</div>

<!-- Error Banner -->
{#if $operatorsStore.error}
  <div class="sov-error-banner">
    <span>⚠</span>
    <span>Operators: {$operatorsStore.error}</span>
  </div>
{/if}

{#if $compileResultStore.error}
  <div class="sov-error-banner">
    <span>⚠</span>
    <span>Compile: {$compileResultStore.error}</span>
  </div>
{/if}

<!-- Main layout -->
<div class="plan-layout">
  <!-- Left: Operator Palette -->
  <div class="plan-palette-col">
    <div class="sov-card" style="height: 100%;">
      <div class="sov-card-header">
        <span class="sov-card-title">Operators</span>
        {#if $operatorsStore.data}
          <span class="sov-badge sov-badge-default">{$operatorsStore.data.count}</span>
        {/if}
      </div>
      <div style="padding: 8px 12px 4px;">
        <input class="sov-input" type="text" placeholder="Search operators..."
          style="height: 28px; font-size: 12px;"
          bind:value={searchText} />
      </div>
      <div class="sov-card-body" style="overflow-y: auto; max-height: 600px;">
        {#if $operatorsStore.loading}
          <div class="sov-loading">
            <div class="sov-spinner"></div>
          </div>
        {:else}
          <OperatorPalette
            {operators}
            procedureFilter={selectedProcedure}
            {searchText}
            on:addOperator={handleAddOperator}
          />
        {/if}
      </div>
    </div>
  </div>

  <!-- Right: Plan Builder -->
  <div class="plan-builder-col">
    <!-- Plan info bar -->
    <div class="sov-card plan-info-card">
      <div class="plan-info-row">
        <div class="plan-info-name">
          {#if plan}
            <input class="plan-name-input" type="text"
              value={plan.name || planName}
              on:input={(e) => {
                planName = e.target.value;
                if (plan) $activePlan = { ...plan, name: e.target.value };
              }}
            />
          {:else}
            <input class="plan-name-input" type="text"
              bind:value={planName}
              placeholder="Plan name..."
            />
          {/if}
        </div>

        <div class="plan-info-stats">
          {#if plan}
            <span class="plan-info-stat">
              <span class="plan-info-val">{plan.n_steps}</span> steps
            </span>
            <span class="plan-info-stat">
              <span class="sov-badge sov-badge-accent">{formatName(plan.procedure)}</span>
            </span>
            {#if plan.content_hash}
              <span class="plan-info-hash font-data"
                title="Content hash: {plan.content_hash}">
                #{plan.content_hash.substring(0, 8)}
              </span>
            {/if}
          {:else}
            <span class="text-muted" style="font-size: 12px;">No plan loaded</span>
          {/if}
        </div>
      </div>
    </div>

    <!-- Steps -->
    <div class="sov-card" style="flex: 1;">
      <div class="sov-card-header">
        <span class="sov-card-title">Plan Steps</span>
        {#if plan}
          <span class="sov-badge sov-badge-default">{plan.n_steps} steps</span>
        {/if}
      </div>
      <div class="sov-card-body" style="overflow-y: auto; max-height: 450px;">
        <PlanSteps
          {plan}
          operatorSchemas={operators}
          on:reorder={handleReorder}
          on:remove={handleRemoveStep}
        />
      </div>
    </div>

    <!-- Compile Section -->
    <div class="sov-card">
      <div class="sov-card-header">
        <span class="sov-card-title">Compilation</span>
      </div>
      <div class="sov-card-body">
        <div class="compile-row">
          <div class="compile-info">
            {#if !selectedCaseId}
              <span class="text-muted" style="font-size: 12px;">Select a case to compile against</span>
            {:else if !plan || plan.n_steps === 0}
              <span class="text-muted" style="font-size: 12px;">Add operators to the plan before compiling</span>
            {:else}
              <span style="font-size: 12px; color: var(--sov-text-secondary);">
                Ready to compile {plan.n_steps} steps against case {selectedCaseId.substring(0, 8)}...
              </span>
            {/if}
          </div>
          <button class="sov-btn sov-btn-primary"
            disabled={!selectedCaseId || !plan || plan.n_steps === 0 || $compileResultStore.loading}
            on:click={handleCompile}>
            {$compileResultStore.loading ? 'Compiling...' : '⚡ Compile Plan'}
          </button>
        </div>

        {#if compileResult}
          <div class="compile-results">
            <div class="compile-result-grid">
              {#each Object.entries(compileResult) as [key, val]}
                <div class="compile-result-item">
                  <span class="compile-result-label">{formatName(key)}</span>
                  <span class="compile-result-value font-data">
                    {typeof val === 'object' ? JSON.stringify(val).substring(0, 40) : val}
                  </span>
                </div>
              {/each}
            </div>
          </div>
        {/if}
      </div>
    </div>
  </div>
</div>

<!-- Template Modal -->
{#if showTemplateModal}
  <div class="sov-modal-overlay" on:click|self={() => showTemplateModal = false}
    role="dialog" aria-modal="true">
    <div class="sov-modal" style="max-width: 560px;">
      <div class="sov-modal-header">
        <span class="sov-modal-title">Plan Templates</span>
        <button class="sov-btn sov-btn-ghost sov-btn-sm"
          on:click={() => showTemplateModal = false}>✕</button>
      </div>
      <div class="sov-modal-body" style="max-height: 400px; overflow-y: auto;">
        {#if $templatesStore.loading}
          <div class="sov-loading">
            <div class="sov-spinner"></div>
          </div>
        {:else if templateEntries.length === 0}
          <div class="sov-empty">
            <div class="sov-empty-title">No Templates</div>
          </div>
        {:else}
          {#each templateEntries as [category, templateList]}
            <div class="template-category">
              <div class="template-cat-name">{formatName(category)}</div>
              <div class="template-grid">
                {#each templateList as tmpl}
                  <button class="template-card"
                    on:click={() => handleSelectTemplate(category, tmpl)}>
                    <span class="template-card-name">{formatName(tmpl)}</span>
                    <span class="template-card-cat">{formatName(category)}</span>
                  </button>
                {/each}
              </div>
            </div>
          {/each}
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  /* ── Plan Layout ──────────────────────────────────────────── */
  .plan-layout {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 16px;
    min-height: 500px;
  }

  .plan-palette-col {
    min-width: 0;
  }

  .plan-builder-col {
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-width: 0;
  }

  /* ── Plan Info Bar ────────────────────────────────────────── */
  .plan-info-card {
    padding: 10px 16px;
  }

  .plan-info-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .plan-info-name {
    flex: 1;
  }

  .plan-name-input {
    width: 100%;
    background: transparent;
    border: none;
    font-family: var(--font-ui, inherit);
    font-size: 16px;
    font-weight: 600;
    color: var(--sov-text-primary, #E8E8EC);
    outline: none;
    padding: 2px 0;
    border-bottom: 1px solid transparent;
    transition: border-color 120ms ease-out;
  }

  .plan-name-input:focus {
    border-bottom-color: var(--sov-accent, #3B82F6);
  }

  .plan-info-stats {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
  }

  .plan-info-stat {
    font-size: 12px;
    color: var(--sov-text-tertiary, #6B7280);
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .plan-info-val {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: var(--sov-text-primary, #E8E8EC);
  }

  .plan-info-hash {
    font-size: 10px;
    color: var(--sov-text-muted, #4B5563);
    background: var(--sov-bg-elevated, #161920);
    padding: 2px 6px;
    border-radius: 3px;
  }

  /* ── Compile Section ──────────────────────────────────────── */
  .compile-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .compile-info {
    flex: 1;
  }

  .compile-results {
    margin-top: 12px;
    padding: 10px;
    background: var(--sov-bg-root, #08080A);
    border: 1px solid var(--sov-border-subtle, #151820);
    border-radius: 4px;
  }

  .compile-result-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 8px;
  }

  .compile-result-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .compile-result-label {
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--sov-text-muted, #4B5563);
  }

  .compile-result-value {
    font-size: 13px;
    color: var(--sov-text-primary, #E8E8EC);
  }

  /* ── Template Modal ───────────────────────────────────────── */
  .template-category {
    margin-bottom: 16px;
  }

  .template-cat-name {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--sov-text-tertiary, #6B7280);
    margin-bottom: 8px;
  }

  .template-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 8px;
  }

  .template-card {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 12px;
    background: var(--sov-bg-root, #08080A);
    border: 1px solid var(--sov-border, #1F2937);
    border-radius: 6px;
    cursor: pointer;
    text-align: left;
    font-family: inherit;
    transition: all 120ms ease-out;
  }

  .template-card:hover {
    border-color: var(--sov-accent, #3B82F6);
    background: var(--sov-accent-glow, rgba(59, 130, 246, 0.06));
  }

  .template-card-name {
    font-size: 12px;
    font-weight: 500;
    color: var(--sov-text-primary, #E8E8EC);
  }

  .template-card-cat {
    font-size: 10px;
    color: var(--sov-text-muted, #4B5563);
  }

  /* ── Responsive ────────────────────────────────────────────── */
  @media (max-width: 900px) {
    .plan-layout {
      grid-template-columns: 1fr;
    }
  }
</style>
