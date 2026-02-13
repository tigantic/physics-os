<script>
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import {
    selectCase,
    loadMesh,
    loadVisualization,
    twinStore,
    meshStore,
    landmarksStore,
    timelineStore,
    visualizationStore,
    activeCaseId,
    clearActiveCase,
  } from '$lib/stores';
  import MeshViewer from '$lib/components/MeshViewer.svelte';
  import RegionLegend from '$lib/components/RegionLegend.svelte';
  import LandmarkPanel from '$lib/components/LandmarkPanel.svelte';

  // ── State ──────────────────────────────────────────────────
  let viewer;
  let activeTab = 'regions';   // 'regions' | 'landmarks' | 'timeline'
  let showLandmarks = true;
  let showWireframe = false;
  let highlightRegion = null;
  let meshLoaded = false;

  $: caseId = $page.params.id;

  // ── Load case data on mount ────────────────────────────────
  onMount(async () => {
    if (caseId) {
      await selectCase(caseId);
      // Auto-load mesh if twin has geometry
      if ($twinStore.data?.mesh) {
        await loadMesh(caseId);
      }
    }
    return () => clearActiveCase();
  });

  // Load visualization for region colors
  $: if (caseId && $twinStore.data?.mesh) {
    loadVisualization(caseId);
  }

  // ── Derived state ──────────────────────────────────────────
  $: twin = $twinStore.data;
  $: mesh = $meshStore.data;
  $: landmarks = $landmarksStore.data;
  $: timeline = $timelineStore.data;
  $: viz = $visualizationStore.data;
  $: regionColors = viz?.region_colors ?? null;

  $: hasMesh = twin?.mesh != null;
  $: meshStats = twin?.mesh ?? null;

  // ── Actions ────────────────────────────────────────────────
  async function handleLoadMesh() {
    if (caseId) await loadMesh(caseId);
  }

  function handleMeshLoaded(e) {
    meshLoaded = true;
  }

  function handleLandmarkClick(e) {
    // Future: fly camera to landmark position
    const lm = e.detail;
    console.log('Focus landmark:', lm.type, lm.position);
  }

  function handleRegionHover(e) {
    highlightRegion = e.detail;
  }

  // Format timestamps
  function timeAgo(ts) {
    if (!ts) return '';
    try {
      const d = new Date(ts);
      return d.toLocaleString();
    } catch { return String(ts); }
  }
</script>

<!-- Breadcrumb -->
<div class="sov-page-header">
  <div class="sov-breadcrumb" style="margin-bottom: 8px;">
    <a href="/cases">Cases</a>
    <span class="sov-breadcrumb-sep">›</span>
    <span class="font-data text-accent">{caseId?.substring(0, 12)}...</span>
  </div>
  <h1 class="sov-page-title">Digital Twin Inspection</h1>
  <p class="sov-page-subtitle">Biomechanical twin analysis and visualization</p>
</div>

{#if $twinStore.loading}
  <div class="sov-loading" style="min-height: 400px;">
    <div class="sov-spinner"></div>
    <span>Loading twin data...</span>
  </div>

{:else if $twinStore.error}
  <div class="sov-error-banner">
    <span>⚠</span>
    <span>{$twinStore.error}</span>
  </div>

{:else if twin}
  <!-- Stats Row -->
  <div class="sov-stat-row">
    <div class="sov-stat">
      <span class="sov-stat-value accent">
        {meshStats ? meshStats.n_nodes.toLocaleString() : '—'}
      </span>
      <span class="sov-stat-label">Nodes</span>
    </div>
    <div class="sov-stat">
      <span class="sov-stat-value">
        {meshStats ? meshStats.n_elements.toLocaleString() : '—'}
      </span>
      <span class="sov-stat-label">Elements</span>
    </div>
    <div class="sov-stat">
      <span class="sov-stat-value">
        {meshStats ? meshStats.n_regions : '—'}
      </span>
      <span class="sov-stat-label">Regions</span>
    </div>
    <div class="sov-stat">
      <span class="sov-stat-value">
        {meshStats?.element_type ?? '—'}
      </span>
      <span class="sov-stat-label">Element Type</span>
    </div>
    <div class="sov-stat">
      <span class="sov-stat-value">
        {landmarks ? landmarks.landmarks.length : '—'}
      </span>
      <span class="sov-stat-label">Landmarks</span>
    </div>
  </div>

  <!-- Main layout: viewer + sidebar -->
  <div class="twin-layout">
    <!-- 3D Viewer -->
    <div class="twin-viewer-col">
      <div class="sov-card twin-viewer-card">
        <div class="sov-card-header">
          <span class="sov-card-title">3D Twin Viewer</span>
          <div style="display: flex; gap: 6px;">
            <button class="sov-btn sov-btn-ghost sov-btn-sm"
              class:active-toggle={showWireframe}
              on:click={() => { showWireframe = !showWireframe; viewer?.toggleWireframe(); }}>
              Wireframe
            </button>
            <button class="sov-btn sov-btn-ghost sov-btn-sm"
              class:active-toggle={showLandmarks}
              on:click={() => { showLandmarks = !showLandmarks; viewer?.toggleLandmarks(); }}>
              Landmarks
            </button>
            <button class="sov-btn sov-btn-ghost sov-btn-sm"
              on:click={() => viewer?.resetView()}>
              Reset
            </button>
          </div>
        </div>

        <div class="viewer-area">
          {#if $meshStore.loading}
            <div class="sov-loading">
              <div class="sov-spinner"></div>
              <span>Loading mesh geometry...</span>
            </div>
          {:else if $meshStore.error}
            <div class="viewer-error">
              <span>⚠ {$meshStore.error}</span>
              <button class="sov-btn sov-btn-secondary sov-btn-sm" on:click={handleLoadMesh}>
                Retry
              </button>
            </div>
          {:else if !hasMesh}
            <div class="viewer-empty">
              <div class="sov-empty-title">No Twin Geometry</div>
              <p>This case has no mesh data. Run curation to generate the digital twin.</p>
            </div>
          {:else}
            <MeshViewer
              bind:this={viewer}
              meshData={mesh}
              landmarkData={landmarks}
              {regionColors}
              {showLandmarks}
              {showWireframe}
              {highlightRegion}
              on:meshLoaded={handleMeshLoaded}
            />
          {/if}
        </div>
      </div>
    </div>

    <!-- Sidebar: tabs for regions, landmarks, timeline -->
    <div class="twin-sidebar-col">
      <!-- Tab bar -->
      <div class="twin-tabs">
        <button class="twin-tab" class:active={activeTab === 'regions'}
          on:click={() => activeTab = 'regions'}>
          Regions
          {#if meshStats}<span class="twin-tab-count">{meshStats.n_regions}</span>{/if}
        </button>
        <button class="twin-tab" class:active={activeTab === 'landmarks'}
          on:click={() => activeTab = 'landmarks'}>
          Landmarks
          {#if landmarks}<span class="twin-tab-count">{landmarks.landmarks.length}</span>{/if}
        </button>
        <button class="twin-tab" class:active={activeTab === 'timeline'}
          on:click={() => activeTab = 'timeline'}>
          Timeline
          {#if timeline}<span class="twin-tab-count">{timeline.n_events}</span>{/if}
        </button>
      </div>

      <!-- Tab content -->
      <div class="sov-card twin-sidebar-card">
        <div class="sov-card-body">
          {#if activeTab === 'regions'}
            <RegionLegend
              {twin}
              {highlightRegion}
              on:regionHover={handleRegionHover}
              on:regionClick={(e) => highlightRegion = highlightRegion === e.detail ? null : e.detail}
            />

          {:else if activeTab === 'landmarks'}
            <LandmarkPanel
              landmarkData={landmarks}
              on:landmarkClick={handleLandmarkClick}
            />

          {:else if activeTab === 'timeline'}
            {#if timeline && timeline.events.length > 0}
              <div class="timeline-list">
                {#each timeline.events as event, i}
                  <div class="timeline-item">
                    <div class="timeline-marker">
                      <div class="timeline-dot"></div>
                      {#if i < timeline.events.length - 1}
                        <div class="timeline-line"></div>
                      {/if}
                    </div>
                    <div class="timeline-content">
                      {#if typeof event === 'object'}
                        {#if event.action}
                          <span class="timeline-action">{event.action}</span>
                        {/if}
                        {#if event.timestamp}
                          <span class="timeline-time">{timeAgo(event.timestamp)}</span>
                        {/if}
                        {#if event.detail || event.message}
                          <span class="timeline-detail">{event.detail || event.message}</span>
                        {/if}
                        {#if !event.action && !event.timestamp}
                          <span class="timeline-detail font-data">
                            {JSON.stringify(event).substring(0, 100)}
                          </span>
                        {/if}
                      {:else}
                        <span class="timeline-detail">{String(event)}</span>
                      {/if}
                    </div>
                  </div>
                {/each}
              </div>
            {:else}
              <div class="sov-empty" style="padding: 24px;">
                <div class="sov-empty-title">No Events</div>
                <p>No audit events recorded for this case.</p>
              </div>
            {/if}
          {/if}
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  /* ── Twin layout ──────────────────────────────────────────── */
  .twin-layout {
    display: grid;
    grid-template-columns: 1fr 320px;
    gap: 16px;
    min-height: 500px;
  }

  .twin-viewer-col {
    min-width: 0;
  }

  .twin-sidebar-col {
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  .twin-viewer-card {
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .viewer-area {
    flex: 1;
    min-height: 400px;
    position: relative;
  }

  .viewer-error, .viewer-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    height: 100%;
    min-height: 400px;
    color: var(--sov-text-tertiary);
    font-size: 13px;
    text-align: center;
    padding: 24px;
  }

  .viewer-error {
    color: var(--sov-error);
  }

  /* ── Viewer toggle buttons ────────────────────────────────── */
  .active-toggle {
    background: var(--sov-accent-glow) !important;
    color: var(--sov-accent) !important;
    border-color: var(--sov-accent) !important;
  }

  /* ── Tabs ──────────────────────────────────────────────────── */
  .twin-tabs {
    display: flex;
    border-bottom: 1px solid var(--sov-border);
    background: var(--sov-bg-card);
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    border: 1px solid var(--sov-border);
    border-bottom: none;
  }

  .twin-tab {
    flex: 1;
    padding: 10px 8px;
    font-family: inherit;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--sov-text-tertiary);
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    transition: all 120ms ease-out;
  }

  .twin-tab:hover {
    color: var(--sov-text-secondary);
    background: var(--sov-bg-hover);
  }

  .twin-tab.active {
    color: var(--sov-accent);
    border-bottom-color: var(--sov-accent);
  }

  .twin-tab-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    background: var(--sov-bg-elevated);
    padding: 0 5px;
    border-radius: 3px;
    color: var(--sov-text-muted);
  }

  .twin-tab.active .twin-tab-count {
    background: var(--sov-accent-dim);
    color: var(--sov-accent);
  }

  .twin-sidebar-card {
    border-radius: 0 0 var(--radius-lg) var(--radius-lg);
    flex: 1;
    overflow: hidden;
  }

  .twin-sidebar-card .sov-card-body {
    max-height: 500px;
    overflow-y: auto;
  }

  /* ── Timeline ──────────────────────────────────────────────── */
  .timeline-list {
    display: flex;
    flex-direction: column;
  }

  .timeline-item {
    display: flex;
    gap: 10px;
    min-height: 36px;
  }

  .timeline-marker {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 12px;
    flex-shrink: 0;
    padding-top: 4px;
  }

  .timeline-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--sov-accent);
    flex-shrink: 0;
  }

  .timeline-line {
    width: 1px;
    flex: 1;
    background: var(--sov-border);
    margin: 3px 0;
  }

  .timeline-content {
    display: flex;
    flex-direction: column;
    gap: 1px;
    padding-bottom: 10px;
    flex: 1;
  }

  .timeline-action {
    font-size: 12px;
    font-weight: 500;
    color: var(--sov-text-primary);
  }

  .timeline-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--sov-text-muted);
  }

  .timeline-detail {
    font-size: 11px;
    color: var(--sov-text-tertiary);
    word-break: break-word;
  }

  /* ── Responsive ────────────────────────────────────────────── */
  @media (max-width: 900px) {
    .twin-layout {
      grid-template-columns: 1fr;
    }

    .twin-sidebar-col {
      order: -1;
    }
  }
</style>
