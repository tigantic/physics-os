<script>
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { initApp, contractStore, casesStore } from '$lib/stores';
  import '../sovereign.css';

  let connected = false;
  let initError = '';

  onMount(async () => {
    try {
      await initApp();
      connected = true;
    } catch (err) {
      initError = err instanceof Error ? err.message : 'Connection failed';
      connected = false;
    }
  });

  // Reactive: track connection from contract store
  $: if ($contractStore.error) {
    connected = false;
    initError = $contractStore.error;
  }
  $: if ($contractStore.data) {
    connected = true;
    initError = '';
  }

  // Nav items — maps to pages built in each phase
  const navSections = [
    {
      label: 'Clinical',
      items: [
        { href: '/cases', label: 'Case Library', icon: '⬡' },
        { href: '/twin', label: 'Digital Twin', icon: '◇' },
        { href: '/plan', label: 'Plan Editor', icon: '▦' },
      ],
    },
    {
      label: 'Analysis',
      items: [
        { href: '/consult', label: 'What-If', icon: '⟁' },
        { href: '/report', label: 'Reports', icon: '▤' },
        { href: '/compare', label: 'Compare', icon: '⟺' },
      ],
    },
    {
      label: 'System',
      items: [
        { href: '/governance', label: 'Governance', icon: '⛋' },
      ],
    },
  ];

  $: currentPath = $page.url.pathname;

  function isActive(href) {
    if (href === '/cases') return currentPath === '/' || currentPath.startsWith('/cases');
    return currentPath.startsWith(href);
  }
</script>

{#if $contractStore.loading}
  <!-- Full-screen loading state while contract loads -->
  <div class="sov-boot">
    <div class="sov-boot-inner">
      <div class="sov-boot-mark">FPS</div>
      <div class="sov-boot-title">Sovereign</div>
      <div class="sov-boot-sub">Initializing platform...</div>
      <div class="sov-spinner" style="margin-top: 16px;"></div>
    </div>
  </div>

{:else if !connected && initError}
  <!-- Connection failure state -->
  <div class="sov-boot">
    <div class="sov-boot-inner">
      <div class="sov-boot-mark" style="background: var(--sov-error);">!</div>
      <div class="sov-boot-title">Connection Failed</div>
      <div class="sov-boot-sub">{initError}</div>
      <div class="sov-boot-hint">
        Backend not running. Start with:<br />
        <code>python -m products.facial_plastics.ui.server --port 8420</code>
      </div>
      <button class="sov-btn sov-btn-primary" style="margin-top: 16px;"
        on:click={() => { $contractStore.loading = true; initApp(); }}>
        Retry
      </button>
    </div>
  </div>

{:else}
  <!-- App shell -->
  <div class="sov-shell">
    <!-- Sidebar -->
    <nav class="sov-sidebar">
      <div class="sov-logo">
        <div class="sov-logo-mark">FPS</div>
        <span class="sov-logo-text">Sovereign</span>
      </div>

      {#each navSections as section}
        <div class="sov-nav-section">
          <div class="sov-nav-label">{section.label}</div>
        </div>
        {#each section.items as item}
          <a
            href={item.href}
            class="sov-nav-item"
            class:active={isActive(item.href)}
          >
            <span class="sov-nav-icon">{item.icon}</span>
            {item.label}
          </a>
        {/each}
      {/each}

      <!-- Bottom: connection status -->
      <div style="margin-top: auto; padding: 12px 16px;">
        <div class="sov-connection">
          <span class="sov-connection-dot" class:offline={!connected}></span>
          {connected ? 'Connected' : 'Offline'}
          {#if connected && $contractStore.data}
            <span style="margin-left: auto;">v{$contractStore.data.version}</span>
          {/if}
        </div>
      </div>
    </nav>

    <!-- Header -->
    <header class="sov-header">
      <div class="sov-breadcrumb">
        <a href="/cases">Sovereign</a>
        <span class="sov-breadcrumb-sep">›</span>
        <span>{currentPath === '/' ? 'Case Library' : currentPath.split('/').filter(Boolean)[0]}</span>
      </div>

      <div class="sov-connection" style="font-size: 10px;">
        {#if $casesStore.data}
          <span class="font-data">{$casesStore.data.total} cases</span>
          <span style="color: var(--sov-text-muted);">|</span>
        {/if}
        {#if $contractStore.data}
          <span class="font-data">{$contractStore.data.operators.count} operators</span>
        {/if}
      </div>
    </header>

    <!-- Main content -->
    <main class="sov-main">
      <slot />
    </main>
  </div>
{/if}

<style>
  .sov-boot {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--sov-bg-root);
  }

  .sov-boot-inner {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }

  .sov-boot-mark {
    width: 48px;
    height: 48px;
    background: var(--sov-accent);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-data);
    font-weight: 700;
    font-size: 16px;
    color: white;
    margin-bottom: 8px;
  }

  .sov-boot-title {
    font-size: 20px;
    font-weight: 600;
    color: var(--sov-text-primary);
  }

  .sov-boot-sub {
    font-size: 13px;
    color: var(--sov-text-tertiary);
  }

  .sov-boot-hint {
    margin-top: 16px;
    padding: 12px 16px;
    background: var(--sov-bg-card);
    border: 1px solid var(--sov-border);
    border-radius: 6px;
    font-size: 12px;
    color: var(--sov-text-secondary);
    text-align: center;
    line-height: 1.6;
  }

  .sov-boot-hint code {
    font-family: var(--font-data);
    font-size: 11px;
    color: var(--sov-accent);
    background: var(--sov-bg-root);
    padding: 2px 6px;
    border-radius: 3px;
  }
</style>
