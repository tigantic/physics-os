<script>
  import ParamEditor from './ParamEditor.svelte';

  /** @type {Record<string, import('$lib/api-client').OperatorSchema>} */
  export let operators = {};

  /** @type {string} */
  export let procedureFilter = '';

  /** @type {string} */
  export let searchText = '';

  /** @type {(detail: { operator: string, params: Record<string, any> }) => void} */
  export let onAddOperator = () => {};

  let expandedOp = null;
  let addParams = {};

  // Category colors
  const CATEGORY_COLORS = {
    rhinoplasty: '#3B82F6',
    facelift: '#10B981',
    necklift: '#10B981',
    blepharoplasty: '#F59E0B',
    filler: '#8B5CF6',
    injection: '#8B5CF6',
  };

  function catColor(cat) {
    if (!cat) return '#6B7280';
    const key = cat.toLowerCase();
    for (const [k, v] of Object.entries(CATEGORY_COLORS)) {
      if (key.includes(k)) return v;
    }
    return '#6B7280';
  }

  function formatName(n) {
    if (!n) return '';
    return n.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  function toggleExpand(opKey) {
    if (expandedOp === opKey) {
      expandedOp = null;
    } else {
      expandedOp = opKey;
      addParams = {};
    }
  }

  function handleAdd(opKey) {
    onAddOperator({
      operator: opKey,
      params: { ...addParams },
    });
    expandedOp = null;
    addParams = {};
  }

  $: entries = Object.entries(operators);

  $: filtered = entries.filter(([key, op]) => {
    if (procedureFilter && op.procedure !== procedureFilter) return false;
    if (searchText) {
      const q = searchText.toLowerCase();
      return key.toLowerCase().includes(q) ||
             op.name?.toLowerCase().includes(q) ||
             op.category?.toLowerCase().includes(q) ||
             op.description?.toLowerCase().includes(q);
    }
    return true;
  });

  // Group by category
  $: grouped = filtered.reduce((acc, [key, op]) => {
    const cat = op.category || op.procedure || 'other';
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push([key, op]);
    return acc;
  }, {});
</script>

<div class="op-palette">
  {#each Object.entries(grouped) as [category, ops]}
    <div class="op-category">
      <div class="op-cat-header">
        <span class="op-cat-dot" style="background: {catColor(category)};"></span>
        <span class="op-cat-name">{formatName(category)}</span>
        <span class="op-cat-count">{ops.length}</span>
      </div>

      {#each ops as [key, op]}
        <div class="op-item" class:expanded={expandedOp === key}>
          <button class="op-item-header" on:click={() => toggleExpand(key)}>
            <div class="op-item-info">
              <span class="op-item-name">{formatName(op.name || key)}</span>
              <span class="op-item-params-count">
                {Object.keys(op.param_defs || {}).length} params
              </span>
            </div>
            <span class="op-item-chevron">{expandedOp === key ? '−' : '+'}</span>
          </button>

          {#if expandedOp === key}
            <div class="op-item-body">
              {#if op.description}
                <p class="op-item-desc">{op.description}</p>
              {/if}

              {#if op.affected_structures?.length > 0}
                <div class="op-item-structures">
                  {#each op.affected_structures as s}
                    <span class="sov-badge sov-badge-default" style="font-size: 9px;">
                      {formatName(s)}
                    </span>
                  {/each}
                </div>
              {/if}

              {#if op.param_defs && Object.keys(op.param_defs).length > 0}
                <div class="op-item-params">
                  <ParamEditor
                    paramDefs={op.param_defs}
                    bind:values={addParams}
                    compact={true}
                  />
                </div>
              {/if}

              <button class="sov-btn sov-btn-primary sov-btn-sm op-add-btn"
                on:click={() => handleAdd(key)}>
                + Add to Plan
              </button>
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {/each}

  {#if filtered.length === 0}
    <div class="sov-empty" style="padding: 20px;">
      <div class="sov-empty-title">No operators found</div>
      <p style="font-size: 12px;">Try adjusting procedure filter or search.</p>
    </div>
  {/if}
</div>

<style>
  .op-palette {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .op-category {
    margin-bottom: 8px;
  }

  .op-cat-header {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 0;
    border-bottom: 1px solid var(--sov-border-subtle, #151820);
    margin-bottom: 2px;
  }

  .op-cat-dot {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex-shrink: 0;
  }

  .op-cat-name {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--sov-text-tertiary, #6B7280);
  }

  .op-cat-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--sov-text-muted, #4B5563);
    margin-left: auto;
  }

  .op-item {
    border: 1px solid transparent;
    border-radius: 4px;
    transition: all 100ms ease-out;
  }

  .op-item:hover {
    background: var(--sov-bg-hover, #1C1F28);
  }

  .op-item.expanded {
    background: var(--sov-bg-card, #111318);
    border-color: var(--sov-border, #1F2937);
    margin: 4px 0;
  }

  .op-item-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    padding: 7px 8px;
    background: none;
    border: none;
    cursor: pointer;
    font-family: inherit;
    text-align: left;
  }

  .op-item-info {
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .op-item-name {
    font-size: 12px;
    font-weight: 500;
    color: var(--sov-text-primary, #E8E8EC);
  }

  .op-item-params-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--sov-text-muted, #4B5563);
  }

  .op-item-chevron {
    font-size: 14px;
    color: var(--sov-text-muted, #4B5563);
    width: 20px;
    text-align: center;
  }

  .op-item-body {
    padding: 4px 8px 10px 8px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .op-item-desc {
    font-size: 11px;
    color: var(--sov-text-tertiary, #6B7280);
    line-height: 1.4;
  }

  .op-item-structures {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }

  .op-item-params {
    padding: 8px;
    background: var(--sov-bg-root, #08080A);
    border-radius: 4px;
    border: 1px solid var(--sov-border-subtle, #151820);
  }

  .op-add-btn {
    align-self: flex-end;
  }
</style>
