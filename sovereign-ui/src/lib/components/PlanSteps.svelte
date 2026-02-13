<script>
  import ParamEditor from './ParamEditor.svelte';

  /** @type {import('$lib/api-client').PlanDict | null} */
  export let plan = null;

  /** @type {Record<string, import('$lib/api-client').OperatorSchema>} */
  export let operatorSchemas = {};

  /** @type {boolean} */
  export let readonly = false;

  /** @type {(detail: { from: number, to: number }) => void} */
  export let onReorder = () => {};

  /** @type {(detail: { index: number }) => void} */
  export let onRemove = () => {};

  /** @type {(detail: { key: string, value: any, allValues: Record<string, any> }) => void} */
  export let onChange = () => {};

  let expandedStep = null;

  function formatName(n) {
    if (!n) return '';
    return n.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  function moveStep(index, direction) {
    if (!plan?.steps) return;
    const newIdx = index + direction;
    if (newIdx < 0 || newIdx >= plan.steps.length) return;
    onReorder({ from: index, to: newIdx });
  }

  function removeStep(index) {
    onRemove({ index });
  }

  function getParamDefs(step) {
    // Look up from operator schemas
    const name = step.name || step.operator;
    if (operatorSchemas[name]) return operatorSchemas[name].param_defs || {};
    // Schema not found — return null to distinguish from "has schema, no params"
    return null;
  }

  $: steps = plan?.steps ?? [];
</script>

<div class="steps-list">
  {#if steps.length === 0}
    <div class="steps-empty">
      <div class="steps-empty-icon">▦</div>
      <div class="sov-empty-title">No Steps</div>
      <p>Add operators from the palette or select a template to build your plan.</p>
    </div>
  {:else}
    {#each steps as step, i}
      <div class="step-item" class:expanded={expandedStep === i}>
        <!-- Step header -->
        <div class="step-header">
          <div class="step-number">{i + 1}</div>

          <button class="step-info" on:click={() => expandedStep = expandedStep === i ? null : i}>
            <span class="step-name">{formatName(step.name || step.operator)}</span>
            <span class="step-meta">
              {#if step.category}
                <span class="sov-badge sov-badge-accent" style="font-size: 9px;">
                  {formatName(step.category)}
                </span>
              {/if}
              {#if step.params}
                <span class="step-param-count">
                  {Object.keys(step.params || {}).length} params
                </span>
              {/if}
            </span>
          </button>

          {#if !readonly}
            <div class="step-actions">
              <button class="step-action-btn"
                disabled={i === 0}
                on:click={() => moveStep(i, -1)}
                title="Move up">↑</button>
              <button class="step-action-btn"
                disabled={i === steps.length - 1}
                on:click={() => moveStep(i, 1)}
                title="Move down">↓</button>
              <button class="step-action-btn danger"
                on:click={() => removeStep(i)}
                title="Remove">✕</button>
            </div>
          {/if}
        </div>

        <!-- Expanded: show params -->
        {#if expandedStep === i}
          <div class="step-body">
            {#if step.affected_structures?.length > 0}
              <div class="step-structures">
                <span class="pe-label-text">Affected Structures</span>
                <div style="display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px;">
                  {#each step.affected_structures as s}
                    <span class="sov-badge sov-badge-default" style="font-size: 9px;">
                      {formatName(s)}
                    </span>
                  {/each}
                </div>
              </div>
            {/if}

            {#if step.params && Object.keys(step.params).length > 0}
              {@const defs = getParamDefs(step)}
              {#if defs !== null}
                <ParamEditor
                  paramDefs={defs}
                  values={step.params}
                  {readonly}
                  compact={true}
                  onChange={(detail) => onChange(detail)}
                />
              {:else}
                <!-- Schema not found — show raw params as JSON -->
                <div class="step-raw-params">
                  <span class="pe-label-text" style="margin-bottom: 4px; display: block;">Raw Parameters (schema not found)</span>
                  <pre style="font-size: 10px; color: var(--sov-text-secondary); margin: 0; white-space: pre-wrap;">{JSON.stringify(step.params, null, 2)}</pre>
                </div>
              {/if}
            {:else}
              <p class="text-muted" style="font-size: 11px;">No configurable parameters.</p>
            {/if}
          </div>
        {/if}
      </div>
    {/each}
  {/if}
</div>

<style>
  .steps-list {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .steps-empty {
    text-align: center;
    padding: 32px 16px;
    color: var(--sov-text-tertiary, #6B7280);
  }

  .steps-empty-icon {
    font-size: 28px;
    opacity: 0.3;
    margin-bottom: 8px;
  }

  .steps-empty p {
    font-size: 12px;
    margin-top: 4px;
    color: var(--sov-text-muted, #4B5563);
  }

  .step-item {
    background: var(--sov-bg-card, #111318);
    border: 1px solid var(--sov-border-subtle, #151820);
    border-radius: 6px;
    transition: all 100ms ease-out;
  }

  .step-item:hover {
    border-color: var(--sov-border, #1F2937);
  }

  .step-item.expanded {
    border-color: var(--sov-accent, #3B82F6)30;
  }

  .step-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
  }

  .step-number {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    color: var(--sov-accent, #3B82F6);
    background: var(--sov-accent-dim, #1E3A5F);
    border-radius: 4px;
    flex-shrink: 0;
  }

  .step-info {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
    background: none;
    border: none;
    cursor: pointer;
    font-family: inherit;
    text-align: left;
    padding: 0;
    min-width: 0;
  }

  .step-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--sov-text-primary, #E8E8EC);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .step-meta {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .step-param-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--sov-text-muted, #4B5563);
  }

  .step-actions {
    display: flex;
    gap: 2px;
    flex-shrink: 0;
  }

  .step-action-btn {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 3px;
    color: var(--sov-text-muted, #4B5563);
    cursor: pointer;
    font-size: 12px;
    transition: all 100ms ease-out;
  }

  .step-action-btn:hover:not(:disabled) {
    background: var(--sov-bg-hover, #1C1F28);
    border-color: var(--sov-border, #1F2937);
    color: var(--sov-text-secondary, #9CA3AF);
  }

  .step-action-btn.danger:hover:not(:disabled) {
    color: var(--sov-error, #EF4444);
    border-color: var(--sov-error, #EF4444)40;
    background: #EF444410;
  }

  .step-action-btn:disabled {
    opacity: 0.2;
    cursor: not-allowed;
  }

  .step-body {
    padding: 4px 10px 12px 44px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    border-top: 1px solid var(--sov-border-subtle, #151820);
  }

  .step-structures {
    padding-top: 4px;
  }
</style>
