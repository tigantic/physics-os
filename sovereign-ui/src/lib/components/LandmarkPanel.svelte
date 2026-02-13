<script>
  /** @type {import('$lib/api-client').LandmarksResponse | null} */
  export let landmarkData = null;

  /** @type {(lm: any) => void} */
  export let onLandmarkClick = () => {};

  let sortBy = 'type';     // 'type' | 'confidence'
  let sortAsc = true;
  let filterText = '';

  function formatType(t) {
    if (!t) return '—';
    return t.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  function toggleSort(col) {
    if (sortBy === col) sortAsc = !sortAsc;
    else { sortBy = col; sortAsc = col === 'type'; }
  }

  $: landmarks = landmarkData?.landmarks ?? [];

  $: filtered = landmarks.filter(lm => {
    if (!filterText) return true;
    return lm.type.toLowerCase().includes(filterText.toLowerCase());
  });

  $: sorted = [...filtered].sort((a, b) => {
    let cmp = 0;
    if (sortBy === 'type') cmp = a.type.localeCompare(b.type);
    else cmp = a.confidence - b.confidence;
    return sortAsc ? cmp : -cmp;
  });

  $: highConf = landmarks.filter(l => l.confidence >= 0.8).length;
  $: medConf = landmarks.filter(l => l.confidence >= 0.5 && l.confidence < 0.8).length;
  $: lowConf = landmarks.filter(l => l.confidence < 0.5).length;
</script>

<div class="lm-panel">
  <!-- Summary bar -->
  <div class="lm-summary">
    <span class="lm-summary-item">
      <span class="lm-dot high"></span>
      <span class="lm-summary-val">{highConf}</span> high
    </span>
    <span class="lm-summary-item">
      <span class="lm-dot med"></span>
      <span class="lm-summary-val">{medConf}</span> med
    </span>
    <span class="lm-summary-item">
      <span class="lm-dot low"></span>
      <span class="lm-summary-val">{lowConf}</span> low
    </span>
  </div>

  <!-- Filter -->
  <input
    class="sov-input lm-filter"
    type="text"
    placeholder="Filter landmarks..."
    bind:value={filterText}
  />

  <!-- List -->
  <div class="lm-list">
    <!-- Header -->
    <div class="lm-row lm-header">
      <button class="lm-col-type lm-sort-btn" on:click={() => toggleSort('type')}>
        Landmark {sortBy === 'type' ? (sortAsc ? '↑' : '↓') : ''}
      </button>
      <button class="lm-col-pos">Position</button>
      <button class="lm-col-conf lm-sort-btn" on:click={() => toggleSort('confidence')}>
        Conf {sortBy === 'confidence' ? (sortAsc ? '↑' : '↓') : ''}
      </button>
    </div>

    {#each sorted as lm}
      <button
        class="lm-row lm-data"
        on:click={() => onLandmarkClick(lm)}
      >
        <span class="lm-col-type">
          <span class="lm-dot"
            class:high={lm.confidence >= 0.8}
            class:med={lm.confidence >= 0.5 && lm.confidence < 0.8}
            class:low={lm.confidence < 0.5}
          ></span>
          {formatType(lm.type)}
        </span>
        <span class="lm-col-pos font-data">
          [{lm.position.map(v => v.toFixed(1)).join(', ')}]
        </span>
        <span class="lm-col-conf">
          <span class="lm-conf-bar">
            <span class="lm-conf-fill"
              class:high={lm.confidence >= 0.8}
              class:med={lm.confidence >= 0.5 && lm.confidence < 0.8}
              class:low={lm.confidence < 0.5}
              style="width: {lm.confidence * 100}%"
            ></span>
          </span>
          <span class="lm-conf-val">{(lm.confidence * 100).toFixed(0)}%</span>
        </span>
      </button>
    {/each}

    {#if sorted.length === 0}
      <div class="lm-empty">
        {filterText ? 'No matches' : 'No landmarks detected'}
      </div>
    {/if}
  </div>
</div>

<style>
  .lm-panel {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .lm-summary {
    display: flex;
    gap: 12px;
    padding: 6px 0;
  }

  .lm-summary-item {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: var(--sov-text-tertiary, #6B7280);
  }

  .lm-summary-val {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    color: var(--sov-text-secondary, #9CA3AF);
  }

  .lm-filter {
    height: 28px !important;
    font-size: 12px !important;
  }

  .lm-list {
    display: flex;
    flex-direction: column;
    max-height: 400px;
    overflow-y: auto;
  }

  .lm-row {
    display: grid;
    grid-template-columns: 1fr 120px 80px;
    align-items: center;
    gap: 4px;
    padding: 5px 6px;
    font-size: 11px;
    border: none;
    background: transparent;
    font-family: inherit;
    text-align: left;
    cursor: pointer;
    border-radius: 3px;
    width: 100%;
  }

  .lm-header {
    color: var(--sov-text-muted, #4B5563);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 10px;
    cursor: default;
    border-bottom: 1px solid var(--sov-border-subtle, #151820);
    padding-bottom: 6px;
    margin-bottom: 2px;
  }

  .lm-data {
    color: var(--sov-text-secondary, #9CA3AF);
    transition: background 100ms ease-out;
  }

  .lm-data:hover {
    background: var(--sov-bg-hover, #1C1F28);
    color: var(--sov-text-primary, #E8E8EC);
  }

  .lm-sort-btn {
    cursor: pointer;
    background: none;
    border: none;
    color: inherit;
    font: inherit;
    text-transform: inherit;
    letter-spacing: inherit;
    padding: 0;
    text-align: left;
  }

  .lm-col-type {
    display: flex;
    align-items: center;
    gap: 6px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 11px;
  }

  .lm-col-pos {
    font-size: 9px;
    color: var(--sov-text-muted, #4B5563);
    background: none;
    border: none;
    font-family: 'JetBrains Mono', monospace;
    text-align: left;
    padding: 0;
  }

  .lm-col-conf {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .lm-conf-bar {
    flex: 1;
    height: 3px;
    background: var(--sov-bg-elevated, #161920);
    border-radius: 2px;
    overflow: hidden;
  }

  .lm-conf-fill {
    display: block;
    height: 100%;
    border-radius: 2px;
    transition: width 200ms ease-out;
  }

  .lm-conf-fill.high { background: #10B981; }
  .lm-conf-fill.med  { background: #F59E0B; }
  .lm-conf-fill.low  { background: #EF4444; }

  .lm-conf-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    min-width: 28px;
    text-align: right;
  }

  .lm-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .lm-dot.high { background: #10B981; box-shadow: 0 0 4px #10B98180; }
  .lm-dot.med  { background: #F59E0B; box-shadow: 0 0 4px #F59E0B80; }
  .lm-dot.low  { background: #EF4444; box-shadow: 0 0 4px #EF444480; }

  .lm-empty {
    text-align: center;
    padding: 24px;
    color: var(--sov-text-muted, #4B5563);
    font-size: 12px;
  }
</style>
