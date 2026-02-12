<script>
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { initApp, contractStore, casesStore } from '$lib/stores';
  import PageTransition from '$lib/components/PageTransition.svelte';
  import '../sovereign.css';

  let connected = false;
  let initError = '';
  let bootPhase = 'contract'; // 'contract' | 'cases' | 'done'

  onMount(async () => {
    try {
      bootPhase = 'contract';
      await initApp();
      bootPhase = 'done';
      connected = true;
    } catch (err) {
      initError = err instanceof Error ? err.message : 'Connection failed';
      connected = false;
    }
  });

  $: if ($contractStore.error) { connected = false; initError = $contractStore.error; }
  $: if ($contractStore.data) { connected = true; initError = ''; }

  const navSections = [
    {
      label: 'Clinical',
      items: [
        { href: '/cases', label: 'Case Library', icon: '⬡', shortcut: '1' },
        { href: '/plan', label: 'Plan Editor', icon: '▦', shortcut: '2' },
        { href: '/consult', label: 'What-If', icon: '⟁', shortcut: '3' },
      ],
    },
    {
      label: 'Output',
      items: [
        { href: '/report', label: 'Reports', icon: '▤', shortcut: '4' },
        { href: '/compare', label: 'Compare', icon: '⟺', shortcut: '5' },
      ],
    },
    {
      label: 'System',
      items: [
        { href: '/governance', label: 'Governance', icon: '⛋', shortcut: '6' },
      ],
    },
  ];

  $: currentPath = $page.url.pathname;

  function isActive(href) {
    if (href === '/cases') return currentPath === '/' || currentPath.startsWith('/cases');
    return currentPath.startsWith(href);
  }

  // Keyboard shortcuts
  function handleKeydown(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
    if (!e.altKey) return;

    const shortcuts = {
      '1': '/cases', '2': '/plan', '3': '/consult',
      '4': '/report', '5': '/compare', '6': '/governance',
    };

    if (shortcuts[e.key]) {
      e.preventDefault();
      goto(shortcuts[e.key]);
    }
  }
</script>

<svelte:window on:keydown={handleKeydown} />

{#if $contractStore.loading}
  <div class="sov-boot">
    <div class="sov-boot-inner">
      <div class="sov-boot-mark">FPS</div>
      <div class="sov-boot-title">Sovereign</div>
      <div class="sov-boot-sub">
        {bootPhase === 'contract' ? 'Loading platform contract...' : 'Loading case library...'}
      </div>
      <div class="sov-boot-progress">
        <div class="sov-boot-bar" style="width: {bootPhase === 'contract' ? '30' : '70'}%;"></div>
      </div>
    </div>
  </div>

{:else if !connected && initError}
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
  <div class="sov-shell">
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
          <a href={item.href} class="sov-nav-item" class:active={isActive(item.href)}>
            <span class="sov-nav-icon">{item.icon}</span>
            <span style="flex: 1;">{item.label}</span>
            <span class="nav-shortcut">⌥{item.shortcut}</span>
          </a>
        {/each}
      {/each}

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

    <header class="sov-header">
      <div class="sov-breadcrumb">
        <a href="/cases">Sovereign</a>
        <span class="sov-breadcrumb-sep">›</span>
        <span>{currentPath === '/' ? 'Case Library' : currentPath.split('/').filter(Boolean)[0]}</span>
        {#if currentPath.includes('/cases/') && currentPath.split('/').length > 2}
          <span class="sov-breadcrumb-sep">›</span>
          <span class="font-data" style="font-size: 11px;">{currentPath.split('/')[2]?.substring(0, 8)}...</span>
        {/if}
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

    <main class="sov-main">
      <PageTransition key={currentPath}>
        <slot />
      </PageTransition>
    </main>
  </div>
{/if}

<style>
  .sov-boot {
    min-height: 100vh; display: flex; align-items: center; justify-content: center;
    background: var(--sov-bg-root);
  }
  .sov-boot-inner { display: flex; flex-direction: column; align-items: center; gap: 8px; }
  .sov-boot-mark {
    width: 48px; height: 48px; background: var(--sov-accent); border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-data); font-weight: 700; font-size: 16px; color: white; margin-bottom: 8px;
  }
  .sov-boot-title { font-size: 20px; font-weight: 600; color: var(--sov-text-primary); }
  .sov-boot-sub { font-size: 13px; color: var(--sov-text-tertiary); }
  .sov-boot-hint {
    margin-top: 16px; padding: 12px 16px; background: var(--sov-bg-card);
    border: 1px solid var(--sov-border); border-radius: 6px;
    font-size: 12px; color: var(--sov-text-secondary); text-align: center; line-height: 1.6;
  }
  .sov-boot-hint code {
    font-family: var(--font-data); font-size: 11px; color: var(--sov-accent);
    background: var(--sov-bg-root); padding: 2px 6px; border-radius: 3px;
  }
  .sov-boot-progress {
    width: 200px; height: 3px; background: var(--sov-bg-elevated);
    border-radius: 2px; margin-top: 12px; overflow: hidden;
  }
  .sov-boot-bar {
    height: 100%; background: var(--sov-accent); border-radius: 2px;
    transition: width 400ms ease-out;
  }

  .nav-shortcut {
    font-family: var(--font-data); font-size: 9px; color: var(--sov-text-muted);
    opacity: 0; transition: opacity 120ms ease-out;
  }
  .sov-nav-item:hover .nav-shortcut { opacity: 1; }
</style>
