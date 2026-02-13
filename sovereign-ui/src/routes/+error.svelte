<script>
  import { page } from '$app/stores';

  $: status = $page.status;
  $: message = $page.error?.message ?? 'An unexpected error occurred';
</script>

<div class="error-page">
  <div class="error-inner">
    <div class="error-code">{status}</div>
    <h1 class="error-title">
      {status === 404 ? 'Page Not Found' : 'Something Went Wrong'}
    </h1>
    <p class="error-message">{message}</p>
    <div class="error-actions">
      <a href="/cases" class="sov-btn sov-btn-primary">Go to Case Library</a>
      <button class="sov-btn sov-btn-ghost" on:click={() => window.location.reload()}>
        Reload Page
      </button>
    </div>
    {#if status === 404}
      <p class="error-hint">Check the URL or navigate using the sidebar.</p>
    {/if}
  </div>
</div>

<style>
  .error-page {
    min-height: 80vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }

  .error-inner {
    text-align: center;
    max-width: 420px;
  }

  .error-code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 64px;
    font-weight: 700;
    color: var(--sov-accent, #3B82F6);
    opacity: 0.3;
    line-height: 1;
    margin-bottom: 8px;
  }

  .error-title {
    font-size: 20px;
    font-weight: 600;
    color: var(--sov-text-primary, #E8E8EC);
    margin-bottom: 8px;
  }

  .error-message {
    font-size: 13px;
    color: var(--sov-text-secondary, #9CA3AF);
    margin-bottom: 20px;
    word-break: break-word;
  }

  .error-actions {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-bottom: 16px;
  }

  .error-hint {
    font-size: 11px;
    color: var(--sov-text-muted, #7B8494);
  }
</style>
