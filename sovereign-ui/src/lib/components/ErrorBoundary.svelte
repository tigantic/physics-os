<script>
  /**
   * ErrorBoundary — Display component for rendering error states.
   *
   * This is NOT a true try/catch error boundary (Svelte does not support
   * component-level error boundaries like React). It renders an error UI
   * when the `error` prop is non-null, and falls through to <slot /> otherwise.
   *
   * Usage:
   *   <ErrorBoundary error={$store.error} onRetry={reload} title="Load failed">
   *     <NormalContent />
   *   </ErrorBoundary>
   */

  /** @type {string} */
  export let title = 'Something went wrong';

  /** @type {string | null} */
  export let error = null;

  /** @type {() => void} */
  export let onRetry = () => {};

  /** @type {boolean} */
  export let showDetails = false;
</script>

{#if error}
  <div class="eb">
    <div class="eb-inner">
      <div class="eb-icon">⚠</div>
      <div class="eb-title">{title}</div>
      <div class="eb-message">{error}</div>
      {#if showDetails}
        <pre class="eb-stack font-data">{error}</pre>
      {/if}
      <div class="eb-actions">
        <button class="sov-btn sov-btn-primary sov-btn-sm" on:click={onRetry}>
          ⟳ Retry
        </button>
        <button class="sov-btn sov-btn-ghost sov-btn-sm"
          on:click={() => showDetails = !showDetails}>
          {showDetails ? 'Hide' : 'Show'} Details
        </button>
      </div>
    </div>
  </div>
{:else}
  <slot />
{/if}

<style>
  .eb {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 200px;
    padding: 24px;
  }

  .eb-inner {
    text-align: center;
    max-width: 400px;
  }

  .eb-icon {
    font-size: 28px;
    margin-bottom: 8px;
    opacity: 0.6;
  }

  .eb-title {
    font-size: 15px;
    font-weight: 600;
    color: var(--sov-text-primary, #E8E8EC);
    margin-bottom: 4px;
  }

  .eb-message {
    font-size: 13px;
    color: var(--sov-error, #EF4444);
    margin-bottom: 12px;
    word-break: break-word;
  }

  .eb-stack {
    text-align: left;
    font-size: 10px;
    color: var(--sov-text-muted, #4B5563);
    background: var(--sov-bg-root, #08080A);
    border: 1px solid var(--sov-border, #1F2937);
    border-radius: 4px;
    padding: 8px;
    margin-bottom: 12px;
    max-height: 120px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
  }

  .eb-actions {
    display: flex;
    justify-content: center;
    gap: 8px;
  }
</style>
