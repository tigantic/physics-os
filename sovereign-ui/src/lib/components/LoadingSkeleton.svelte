<script>
  /** @type {'card' | 'table' | 'stat-row' | 'viewer' | 'text' | 'inline'} */
  export let variant = 'card';

  /** Number of skeleton rows for table variant */
  export let rows = 5;

  /** Number of stat boxes for stat-row variant */
  export let count = 4;
</script>

{#if variant === 'stat-row'}
  <div class="sk-stat-row">
    {#each Array(count) as _}
      <div class="sk-stat">
        <div class="sk-line sk-shimmer" style="width: 60%; height: 24px;"></div>
        <div class="sk-line sk-shimmer" style="width: 80%; height: 10px; margin-top: 6px;"></div>
      </div>
    {/each}
  </div>

{:else if variant === 'table'}
  <div class="sk-card">
    <div class="sk-card-header">
      <div class="sk-line sk-shimmer" style="width: 120px; height: 12px;"></div>
    </div>
    <div class="sk-table">
      <div class="sk-table-head">
        {#each Array(4) as _}
          <div class="sk-line sk-shimmer" style="width: 80px; height: 10px;"></div>
        {/each}
      </div>
      {#each Array(rows) as _, i}
        <div class="sk-table-row" style="animation-delay: {i * 60}ms;">
          {#each Array(4) as _}
            <div class="sk-line sk-shimmer" style="width: {60 + Math.random() * 40}%; height: 12px;"></div>
          {/each}
        </div>
      {/each}
    </div>
  </div>

{:else if variant === 'viewer'}
  <div class="sk-viewer">
    <div class="sk-viewer-inner">
      <div class="sk-viewer-cube sk-shimmer"></div>
      <div class="sk-line sk-shimmer" style="width: 140px; height: 10px; margin-top: 12px;"></div>
    </div>
  </div>

{:else if variant === 'text'}
  <div class="sk-text">
    {#each Array(rows) as _, i}
      <div class="sk-line sk-shimmer" style="width: {70 + Math.random() * 30}%; height: 12px; animation-delay: {i * 40}ms;"></div>
    {/each}
  </div>

{:else if variant === 'inline'}
  <span class="sk-inline sk-shimmer"></span>

{:else}
  <!-- card variant -->
  <div class="sk-card">
    <div class="sk-card-header">
      <div class="sk-line sk-shimmer" style="width: 100px; height: 12px;"></div>
    </div>
    <div class="sk-card-body">
      <div class="sk-line sk-shimmer" style="width: 80%; height: 14px;"></div>
      <div class="sk-line sk-shimmer" style="width: 60%; height: 14px;"></div>
      <div class="sk-line sk-shimmer" style="width: 90%; height: 14px;"></div>
    </div>
  </div>
{/if}

<style>
  .sk-shimmer {
    background: linear-gradient(
      90deg,
      var(--sov-bg-elevated, #161920) 25%,
      var(--sov-bg-hover, #1C1F28) 50%,
      var(--sov-bg-elevated, #161920) 75%
    );
    background-size: 200% 100%;
    animation: sk-shimmer 1.5s ease-in-out infinite;
    border-radius: 3px;
  }

  @keyframes sk-shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }

  .sk-line { display: block; }

  .sk-stat-row { display: flex; gap: 12px; margin-bottom: 20px; }
  .sk-stat {
    flex: 1; padding: 16px;
    background: var(--sov-bg-card, #111318);
    border: 1px solid var(--sov-border, #1F2937);
    border-radius: 6px;
  }

  .sk-card {
    background: var(--sov-bg-card, #111318);
    border: 1px solid var(--sov-border, #1F2937);
    border-radius: 8px; overflow: hidden;
  }
  .sk-card-header {
    padding: 10px 16px;
    border-bottom: 1px solid var(--sov-border-subtle, #151820);
  }
  .sk-card-body { padding: 16px; display: flex; flex-direction: column; gap: 10px; }

  .sk-table { padding: 0; }
  .sk-table-head {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 8px; padding: 8px 16px;
    border-bottom: 1px solid var(--sov-border, #1F2937);
  }
  .sk-table-row {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 8px; padding: 10px 16px;
    border-bottom: 1px solid var(--sov-border-subtle, #151820);
    animation: sk-fade-in 300ms ease-out both;
  }
  @keyframes sk-fade-in {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .sk-viewer {
    background: var(--sov-bg-card, #111318);
    border: 1px solid var(--sov-border, #1F2937);
    border-radius: 8px; min-height: 400px;
    display: flex; align-items: center; justify-content: center;
  }
  .sk-viewer-inner { text-align: center; }
  .sk-viewer-cube {
    width: 48px; height: 48px; border-radius: 8px;
    margin: 0 auto;
  }

  .sk-text { display: flex; flex-direction: column; gap: 8px; }

  .sk-inline {
    display: inline-block; width: 60px; height: 14px;
    vertical-align: middle;
  }
</style>
