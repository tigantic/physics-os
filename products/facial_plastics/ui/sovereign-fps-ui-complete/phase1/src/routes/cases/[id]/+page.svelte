<script>
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { selectCase, twinStore, landmarksStore, timelineStore, activeCaseId } from '$lib/stores';

  $: caseId = $page.params.id;

  onMount(() => {
    if (caseId) selectCase(caseId);
  });

  $: twin = $twinStore.data;
  $: landmarks = $landmarksStore.data;
  $: timeline = $timelineStore.data;
</script>

<div class="sov-page-header">
  <div class="sov-breadcrumb" style="margin-bottom: 8px;">
    <a href="/cases">Cases</a>
    <span class="sov-breadcrumb-sep">›</span>
    <span class="font-data text-accent">{caseId?.substring(0, 12)}...</span>
  </div>
  <h1 class="sov-page-title">Case Detail</h1>
  <p class="sov-page-subtitle">Digital twin inspection and plan authoring</p>
</div>

{#if $twinStore.loading}
  <div class="sov-loading">
    <div class="sov-spinner"></div>
    <span>Loading twin data...</span>
  </div>

{:else if $twinStore.error}
  <div class="sov-error-banner">
    <span>⚠</span>
    <span>{$twinStore.error}</span>
  </div>

{:else if twin}
  <!-- Stats Row — real data from getTwinSummary -->
  <div class="sov-stat-row">
    <div class="sov-stat">
      <span class="sov-stat-value accent">
        {twin.mesh ? twin.mesh.n_nodes.toLocaleString() : '—'}
      </span>
      <span class="sov-stat-label">Mesh Nodes</span>
    </div>
    <div class="sov-stat">
      <span class="sov-stat-value">
        {twin.mesh ? twin.mesh.n_elements.toLocaleString() : '—'}
      </span>
      <span class="sov-stat-label">Elements</span>
    </div>
    <div class="sov-stat">
      <span class="sov-stat-value">
        {twin.mesh ? twin.mesh.n_regions : '—'}
      </span>
      <span class="sov-stat-label">Regions</span>
    </div>
    <div class="sov-stat">
      <span class="sov-stat-value">
        {landmarks ? landmarks.landmarks.length : '—'}
      </span>
      <span class="sov-stat-label">Landmarks</span>
    </div>
  </div>

  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
    <!-- Regions Card -->
    <div class="sov-card">
      <div class="sov-card-header">
        <span class="sov-card-title">Regions</span>
        {#if twin.mesh}
          <span class="sov-badge sov-badge-default">{twin.mesh.n_regions} regions</span>
        {/if}
      </div>
      <div class="sov-card-body">
        {#if twin.mesh && twin.mesh.regions}
          <table class="sov-table" style="font-size: 12px;">
            <thead>
              <tr>
                <th>ID</th>
                <th>Structure</th>
                <th>Material</th>
              </tr>
            </thead>
            <tbody>
              {#each Object.entries(twin.mesh.regions) as [id, region]}
                <tr>
                  <td class="font-data">{id}</td>
                  <td>{region.structure}</td>
                  <td><span class="sov-badge sov-badge-accent">{region.material}</span></td>
                </tr>
              {/each}
            </tbody>
          </table>
        {:else}
          <div class="sov-empty">
            <div class="sov-empty-title">No mesh data</div>
            <p>Run curation to generate twin geometry.</p>
          </div>
        {/if}
      </div>
    </div>

    <!-- Landmarks Card -->
    <div class="sov-card">
      <div class="sov-card-header">
        <span class="sov-card-title">Landmarks</span>
        {#if landmarks}
          <span class="sov-badge sov-badge-default">{landmarks.landmarks.length} detected</span>
        {/if}
      </div>
      <div class="sov-card-body">
        {#if landmarks && landmarks.landmarks.length > 0}
          <table class="sov-table" style="font-size: 12px;">
            <thead>
              <tr>
                <th>Type</th>
                <th>Position</th>
                <th>Conf.</th>
              </tr>
            </thead>
            <tbody>
              {#each landmarks.landmarks as lm}
                <tr>
                  <td>{lm.type}</td>
                  <td class="font-data" style="font-size: 10px;">
                    [{lm.position.map(v => v.toFixed(2)).join(', ')}]
                  </td>
                  <td>
                    <span class="sov-badge"
                      class:sov-badge-success={lm.confidence >= 0.8}
                      class:sov-badge-warning={lm.confidence >= 0.5 && lm.confidence < 0.8}
                      class:sov-badge-error={lm.confidence < 0.5}>
                      {(lm.confidence * 100).toFixed(0)}%
                    </span>
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        {:else}
          <div class="sov-empty">
            <div class="sov-empty-title">No landmarks</div>
          </div>
        {/if}
      </div>
    </div>
  </div>

  <!-- Timeline Card -->
  {#if timeline}
    <div class="sov-card" style="margin-top: 16px;">
      <div class="sov-card-header">
        <span class="sov-card-title">Audit Timeline</span>
        <span class="sov-badge sov-badge-default">{timeline.n_events} events</span>
      </div>
      <div class="sov-card-body">
        {#if timeline.events.length > 0}
          <div class="timeline-list">
            {#each timeline.events.slice(0, 20) as event, i}
              <div class="timeline-item">
                <div class="timeline-dot"></div>
                <div class="timeline-content">
                  <span class="font-data" style="font-size: 11px; color: var(--sov-text-tertiary);">
                    {JSON.stringify(event).substring(0, 120)}
                  </span>
                </div>
              </div>
            {/each}
          </div>
        {:else}
          <p class="text-muted" style="font-size: 13px;">No events recorded yet.</p>
        {/if}
      </div>
    </div>
  {/if}

  <!-- Phase 2 placeholder -->
  <div class="sov-card" style="margin-top: 16px; border-style: dashed;">
    <div class="sov-card-body" style="text-align: center; padding: 32px;">
      <div class="sov-empty-title">3D Twin Viewer</div>
      <p style="color: var(--sov-text-tertiary); font-size: 13px;">
        Phase 2 — Three.js mesh rendering, landmark overlays, region visualization
      </p>
    </div>
  </div>
{/if}

<style>
  .timeline-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .timeline-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
  }

  .timeline-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--sov-accent);
    margin-top: 5px;
    flex-shrink: 0;
  }

  .timeline-content {
    flex: 1;
  }
</style>
