<script>
  import { onMount, onDestroy } from 'svelte';
  import { page } from '$app/stores';
  import {
    selectCase, loadMesh, loadVisualization,
    twinStore, meshStore, landmarksStore, timelineStore, visualizationStore,
    activeCaseId, activePlan, clearActiveCase,
  } from '$lib/stores';
  import {
    loadTissueLayers, loadCfdResults, loadFemResults, loadDicomData,
    runCfdSimulation, runFemSimulation, executeIncision, executeOsteotomy,
    placeGraft, generatePostOpPrediction, loadPostOpPrediction,
  } from '$lib/api-client-ext';

  import LayeredTissueViewer from '$lib/components/LayeredTissueViewer.svelte';
  import SurgicalToolbar from '$lib/components/SurgicalToolbar.svelte';
  import LayerControlPanel from '$lib/components/LayerControlPanel.svelte';
  import CfdPanel from '$lib/components/CfdPanel.svelte';
  import FemPanel from '$lib/components/FemPanel.svelte';
  import DicomViewer from '$lib/components/DicomViewer.svelte';
  import ColorBar from '$lib/components/ColorBar.svelte';
  import PrePostView from '$lib/components/PrePostView.svelte';
  import RegionLegend from '$lib/components/RegionLegend.svelte';
  import LandmarkPanel from '$lib/components/LandmarkPanel.svelte';
  import MeshViewer from '$lib/components/MeshViewer.svelte';

  // ── State ──────────────────────────────────────────────────
  let viewer;
  let postViewer;
  $: caseId = $page.params.id;

  // Data stores
  let layerData = null;
  let layerNames = [];
  let cfdData = null;
  let femData = null;
  let dicomData = null;
  let postOpData = null;

  // Loading states
  let loadingLayers = false;
  let loadingCfd = false;
  let loadingFem = false;
  let loadingDicom = false;
  let loadingPostOp = false;
  let runningCfd = false;
  let runningFem = false;

  // Viewer state
  let layerConfig = {};
  let interactionMode = 'orbit';
  let highlightLayer = null;
  let showLandmarks = true;
  let showLabels = true;
  let showStreamlines = true;
  let showScalarField = false;
  let showIncisions = true;
  let clipEnabled = false;
  let clipConfig = { axis: 'y', position: 0, enabled: false };

  // FEM state
  let femActiveField = 'stress';
  let femColormap = 'stress';

  // Incision / surgery state
  let incisionPaths = [];
  let incisionDrawing = [];
  let canCommitIncision = false;
  let canUndoIncision = false;

  // Layout
  let activeRightTab = 'layers';
  let activeLeftTab = 'dicom';
  let showPrePost = false;
  let splitPosition = 50;

  // ── Data Loading ───────────────────────────────────────────
  async function loadAllData() {
    if (!caseId) return;

    // Load base twin data
    await selectCase(caseId);

    // Load tissue layers (new endpoint — graceful fallback to mesh)
    loadingLayers = true;
    const layers = await loadTissueLayers(caseId);
    if (layers) {
      layerData = layers.layers;
      layerNames = layers.layer_order || Object.keys(layers.layers);
      initLayerConfig();
    } else {
      // Fallback: load single mesh as one layer
      await loadMesh(caseId);
      if ($meshStore.data) {
        layerData = { tissue: $meshStore.data };
        layerNames = ['tissue'];
        initLayerConfig();
      }
    }
    loadingLayers = false;

    // Load visualization config
    loadVisualization(caseId);

    // Load CFD (may not exist yet)
    loadingCfd = true;
    cfdData = await loadCfdResults(caseId);
    loadingCfd = false;

    // Load FEM (may not exist yet)
    loadingFem = true;
    femData = await loadFemResults(caseId);
    loadingFem = false;

    // Load DICOM (may not exist)
    loadingDicom = true;
    dicomData = await loadDicomData(caseId);
    loadingDicom = false;

    // Load post-op prediction if plan exists
    if ($activePlan?.content_hash) {
      loadingPostOp = true;
      postOpData = await loadPostOpPrediction(caseId, $activePlan.content_hash);
      loadingPostOp = false;
    }
  }

  function initLayerConfig() {
    const DEFAULTS = {
      bone: { visible: true, opacity: 1.0, wireframe: false },
      cartilage: { visible: true, opacity: 0.85, wireframe: false },
      muscle: { visible: true, opacity: 0.7, wireframe: false },
      skin: { visible: true, opacity: 0.5, wireframe: false },
      fascia: { visible: true, opacity: 0.35, wireframe: false },
      soft_tissue: { visible: true, opacity: 0.6, wireframe: false },
      tissue: { visible: true, opacity: 1.0, wireframe: false },
    };
    for (const name of layerNames) {
      const key = Object.keys(DEFAULTS).find(k => name.toLowerCase().includes(k));
      layerConfig[name] = DEFAULTS[key || 'tissue'];
    }
    layerConfig = { ...layerConfig };
  }

  // ── Simulation Triggers ────────────────────────────────────
  async function handleRunCfd() {
    runningCfd = true;
    cfdData = await runCfdSimulation(caseId, $activePlan?.content_hash);
    runningCfd = false;
  }

  async function handleRunFem() {
    runningFem = true;
    femData = await runFemSimulation(caseId, $activePlan?.content_hash);
    runningFem = false;
  }

  async function handleGeneratePostOp() {
    if (!$activePlan?.content_hash) return;
    loadingPostOp = true;
    postOpData = await generatePostOpPrediction(caseId, $activePlan.content_hash);
    loadingPostOp = false;
  }

  // ── Surgical Tool Handlers ─────────────────────────────────
  function handleIncisionPoint(e) {
    canCommitIncision = (e.detail.pathLength >= 2);
    canUndoIncision = (e.detail.pathLength > 0);
  }

  async function handleIncisionCommit(e) {
    const result = await executeIncision(caseId, e.detail.points, 5.0);
    if (result) {
      incisionPaths = [...incisionPaths, { points: e.detail.points, color: '#EF4444', depth: 5.0 }];
      // Update layers if backend returned updated mesh
      if (result.updated_mesh) {
        layerData = { ...layerData, ...result.updated_mesh };
      }
    }
    canCommitIncision = false;
    canUndoIncision = false;
  }

  function handleIncisionCancel() {
    viewer?.clearIncision();
    canCommitIncision = false;
    canUndoIncision = false;
  }

  async function handleGraftPlace(e) {
    const result = await placeGraft(
      caseId, e.detail.position, e.detail.normal,
      'septal', { length: 15, width: 10, thickness: 2 }
    );
    if (result?.graft_mesh) {
      layerData = { ...layerData, [`graft_${result.graft_id}`]: result.graft_mesh };
      layerNames = [...layerNames, `graft_${result.graft_id}`];
      layerConfig[`graft_${result.graft_id}`] = { visible: true, opacity: 0.9, wireframe: false };
      layerConfig = { ...layerConfig };
    }
  }

  // ── Toolbar Handlers ───────────────────────────────────────
  function handleModeChange(e) { interactionMode = e.detail; }
  function handleViewPreset(e) {
    // Camera presets
    // Would call viewer.setCameraPreset(e.detail) with known angles
  }

  function handleCommit() {
    if (interactionMode === 'incision') viewer?.commitIncision();
  }

  function handleUndo() {
    if (interactionMode === 'incision') {
      viewer?.clearIncision();
      canCommitIncision = false;
    }
    if (incisionPaths.length > 0) {
      incisionPaths = incisionPaths.slice(0, -1);
      canUndoIncision = incisionPaths.length > 0;
    }
  }

  function handleCancel() {
    interactionMode = 'orbit';
    viewer?.clearIncision();
    viewer?.clearMeasurement();
  }

  // ── FEM Field Change ───────────────────────────────────────
  function handleFemFieldChange(e) {
    femActiveField = e.detail.field;
    femColormap = e.detail.colormap;
    // Apply scalar field to viewer
    if (femData?.fields?.[femActiveField]) {
      // The viewer will pick this up reactively
    }
  }

  // ── Computed ───────────────────────────────────────────────
  $: twin = $twinStore.data;
  $: landmarks = $landmarksStore.data;
  $: timeline = $timelineStore.data;
  $: viz = $visualizationStore.data;

  $: streamlineData = (showStreamlines && cfdData) ? cfdData.streamlines || cfdData : null;
  $: scalarFieldData = (showScalarField && femData?.fields?.[femActiveField]) ? femData.fields[femActiveField] : null;

  $: hasPlan = $activePlan != null;
  $: meshStats = twin?.mesh ?? null;
  $: totalVerts = layerData ? Object.values(layerData).reduce((sum, l) => sum + (l.n_vertices || l.positions?.length || 0), 0) : 0;
  $: totalTris = layerData ? Object.values(layerData).reduce((sum, l) => sum + (l.n_triangles || l.indices?.length || 0), 0) : 0;

  onMount(() => { loadAllData(); });
  onDestroy(() => { clearActiveCase(); });
</script>

<!-- Top bar -->
<div class="sov-page-header cockpit-header">
  <div class="cockpit-breadcrumb">
    <a href="/cases" class="sov-breadcrumb">← Cases</a>
    <span class="sov-breadcrumb-sep">›</span>
    <span class="font-data text-accent">{caseId?.substring(0, 12)}…</span>
    {#if hasPlan}
      <span class="sov-breadcrumb-sep">›</span>
      <span class="sov-badge sov-badge-accent">{$activePlan.name || 'Active Plan'}</span>
    {/if}
  </div>
  <div class="cockpit-title-row">
    <h1 class="sov-page-title">Physics-Driven Digital Twin System</h1>
    <div class="cockpit-title-actions">
      <button class="sov-btn sov-btn-sm sov-btn-secondary" class:active={showPrePost}
        on:click={() => showPrePost = !showPrePost}>
        {showPrePost ? '◉ Single View' : '◫ Pre/Post'}
      </button>
      {#if hasPlan && !postOpData}
        <button class="sov-btn sov-btn-sm sov-btn-primary" on:click={handleGeneratePostOp}
          disabled={loadingPostOp}>
          {loadingPostOp ? 'Computing…' : '▶ Generate Post-Op'}
        </button>
      {/if}
    </div>
  </div>
</div>

<!-- Stats ribbon -->
<div class="cockpit-ribbon">
  <div class="ribbon-stat"><span class="ribbon-val">{totalVerts.toLocaleString()}</span><span class="ribbon-label">Vertices</span></div>
  <div class="ribbon-sep"></div>
  <div class="ribbon-stat"><span class="ribbon-val">{totalTris.toLocaleString()}</span><span class="ribbon-label">Triangles</span></div>
  <div class="ribbon-sep"></div>
  <div class="ribbon-stat"><span class="ribbon-val">{layerNames.length}</span><span class="ribbon-label">Layers</span></div>
  <div class="ribbon-sep"></div>
  <div class="ribbon-stat"><span class="ribbon-val">{landmarks?.landmarks?.length ?? '—'}</span><span class="ribbon-label">Landmarks</span></div>
  <div class="ribbon-sep"></div>
  <div class="ribbon-stat">
    <span class="ribbon-val" class:active-val={cfdData != null}>{cfdData ? '✓' : '—'}</span>
    <span class="ribbon-label">CFD</span>
  </div>
  <div class="ribbon-sep"></div>
  <div class="ribbon-stat">
    <span class="ribbon-val" class:active-val={femData != null}>{femData ? '✓' : '—'}</span>
    <span class="ribbon-label">FEM</span>
  </div>
  <div class="ribbon-sep"></div>
  <div class="ribbon-stat">
    <span class="ribbon-val" class:active-val={dicomData != null}>{dicomData ? '✓' : '—'}</span>
    <span class="ribbon-label">DICOM</span>
  </div>
  <div class="ribbon-sep"></div>
  <div class="ribbon-stat">
    <span class="ribbon-val" class:active-val={postOpData != null}>{postOpData ? '✓' : '—'}</span>
    <span class="ribbon-label">Post-Op</span>
  </div>
  {#if incisionPaths.length > 0}
    <div class="ribbon-sep"></div>
    <div class="ribbon-stat"><span class="ribbon-val" style="color: #EF4444;">{incisionPaths.length}</span><span class="ribbon-label">Incisions</span></div>
  {/if}
</div>

<!-- Surgical toolbar -->
<div class="sov-card cockpit-toolbar-card">
  <SurgicalToolbar
    mode={interactionMode}
    canCommit={canCommitIncision}
    canUndo={canUndoIncision || incisionPaths.length > 0}
    {showStreamlines}
    {showScalarField}
    {showLandmarks}
    {showLabels}
    {showIncisions}
    {clipEnabled}
    scalarFieldType={femActiveField}
    on:modeChange={handleModeChange}
    on:commit={handleCommit}
    on:undo={handleUndo}
    on:cancel={handleCancel}
    on:streamlinesToggle={(e) => showStreamlines = e.detail}
    on:scalarToggle={(e) => showScalarField = e.detail}
    on:landmarksToggle={(e) => showLandmarks = e.detail}
    on:labelsToggle={(e) => showLabels = e.detail}
    on:incisionsToggle={(e) => showIncisions = e.detail}
    on:clipToggle={(e) => { clipEnabled = e.detail; clipConfig = { ...clipConfig, enabled: e.detail }; }}
    on:scalarTypeChange={(e) => { femActiveField = e.detail; }}
    on:resetView={() => viewer?.resetView()}
    on:viewPreset={handleViewPreset}
  />
</div>

<!-- Main cockpit layout -->
<div class="cockpit-layout">
  <!-- Left panel: DICOM + Timeline -->
  <div class="cockpit-left">
    <div class="cockpit-left-tabs">
      {#each [['dicom','Imaging'],['timeline','Timeline'],['plan','Plan']] as [key, lbl]}
        <button class="cockpit-tab" class:active={activeLeftTab === key}
          on:click={() => activeLeftTab = key}>
          {lbl}
        </button>
      {/each}
    </div>

    <div class="sov-card cockpit-left-card">
      <div class="sov-card-body" style="overflow-y: auto; max-height: 600px;">
        {#if activeLeftTab === 'dicom'}
          <DicomViewer dicomData={dicomData} on:uploadDicom />
        {:else if activeLeftTab === 'timeline'}
          {#if timeline?.events?.length > 0}
            <div class="timeline-list">
              {#each timeline.events as event, i}
                <div class="timeline-item">
                  <div class="timeline-marker">
                    <div class="timeline-dot"></div>
                    {#if i < timeline.events.length - 1}<div class="timeline-line"></div>{/if}
                  </div>
                  <div class="timeline-content">
                    {#if typeof event === 'object'}
                      {#if event.action}<span class="timeline-action">{event.action}</span>{/if}
                      {#if event.timestamp}<span class="timeline-time">{new Date(event.timestamp).toLocaleString()}</span>{/if}
                      {#if event.detail}<span class="timeline-detail">{event.detail}</span>{/if}
                    {:else}
                      <span class="timeline-detail">{String(event)}</span>
                    {/if}
                  </div>
                </div>
              {/each}
            </div>
          {:else}
            <div class="sov-empty" style="padding: 24px;"><div class="sov-empty-title">No events</div></div>
          {/if}
        {:else if activeLeftTab === 'plan'}
          {#if hasPlan}
            <div class="plan-summary">
              <div class="plan-name">{$activePlan.name}</div>
              <div class="plan-meta">
                {#if $activePlan.procedure}<span class="sov-badge sov-badge-accent">{$activePlan.procedure}</span>{/if}
                <span class="font-data" style="font-size: 10px; color: var(--sov-text-muted);">
                  {$activePlan.steps?.length ?? 0} steps
                </span>
              </div>
              {#if $activePlan.steps}
                <div class="plan-steps-mini">
                  {#each $activePlan.steps as step, i}
                    <div class="plan-step-mini">
                      <span class="plan-step-num">{i + 1}</span>
                      <span class="plan-step-name">{step.operator || step.name}</span>
                    </div>
                  {/each}
                </div>
              {/if}
            </div>
          {:else}
            <div class="sov-empty" style="padding: 24px;">
              <div class="sov-empty-title">No Active Plan</div>
              <a href="/plan" class="sov-btn sov-btn-secondary sov-btn-sm" style="margin-top: 8px;">Open Plan Editor</a>
            </div>
          {/if}
        {/if}
      </div>
    </div>

    <!-- FEM results (below DICOM) -->
    <div class="sov-card cockpit-left-card" style="margin-top: 8px;">
      <div class="sov-card-body" style="overflow-y: auto; max-height: 400px;">
        <FemPanel
          femData={femData}
          visible={showScalarField}
          activeField={femActiveField}
          colormap={femColormap}
          on:visibilityChange={(e) => showScalarField = e.detail}
          on:fieldChange={handleFemFieldChange}
          on:runFem={handleRunFem}
        />
      </div>
    </div>
  </div>

  <!-- Center: 3D Viewer (main) -->
  <div class="cockpit-center">
    {#if loadingLayers}
      <div class="sov-card" style="height: 100%; display: flex; align-items: center; justify-content: center;">
        <div class="sov-spinner"></div><span style="color: var(--sov-text-muted); margin-left: 8px;">Loading tissue layers...</span>
      </div>
    {:else if showPrePost && postOpData}
      <!-- Pre/Post split view -->
      <PrePostView mode="slider" bind:splitPosition>
        <div slot="pre" style="height: 100%;">
          <LayeredTissueViewer
            bind:this={viewer}
            {layerData}
            landmarkData={landmarks}
            {layerConfig}
            streamlines={streamlineData}
            scalarField={scalarFieldData}
            incisionPaths={showIncisions ? incisionPaths : null}
            clipConfig={clipConfig}
            {interactionMode}
            {showLandmarks}
            {showLabels}
            {highlightLayer}
            on:measurement
            on:incisionPoint={handleIncisionPoint}
            on:incisionCommit={handleIncisionCommit}
            on:graftPlace={handleGraftPlace}
            on:modeChange={(e) => interactionMode = e.detail}
          />
        </div>
        <div slot="post" style="height: 100%;">
          <LayeredTissueViewer
            bind:this={postViewer}
            layerData={postOpData.layers}
            landmarkData={postOpData.landmarks_predicted}
            {layerConfig}
            streamlines={postOpData.cfd_predicted?.streamlines}
            scalarField={null}
            clipConfig={clipConfig}
            interactionMode="orbit"
            {showLandmarks}
            {showLabels}
          />
        </div>
      </PrePostView>
    {:else}
      <!-- Single viewer -->
      <div class="sov-card cockpit-viewer-card">
        <LayeredTissueViewer
          bind:this={viewer}
          {layerData}
          landmarkData={landmarks}
          {layerConfig}
          streamlines={streamlineData}
          scalarField={scalarFieldData}
          incisionPaths={showIncisions ? incisionPaths : null}
          clipConfig={clipConfig}
          {interactionMode}
          {showLandmarks}
          {showLabels}
          {highlightLayer}
          on:measurement
          on:incisionPoint={handleIncisionPoint}
          on:incisionCommit={handleIncisionCommit}
          on:graftPlace={handleGraftPlace}
          on:modeChange={(e) => interactionMode = e.detail}
        />

        <!-- Scalar field colorbar overlay -->
        {#if showScalarField && scalarFieldData}
          <div class="cockpit-colorbar-overlay">
            <ColorBar
              label={scalarFieldData.field || femActiveField}
              unit={femActiveField === 'stress' ? 'MPa' : femActiveField === 'displacement' ? 'mm' : ''}
              min={scalarFieldData.min ?? 0}
              max={scalarFieldData.max ?? 1}
              colormap={femColormap}
              orientation="vertical"
              compact
            />
          </div>
        {/if}

        <!-- CFD colorbar overlay -->
        {#if showStreamlines && cfdData}
          <div class="cockpit-colorbar-overlay" style="right: 60px;">
            <ColorBar
              label="Velocity"
              unit="m/s"
              min={cfdData.velocity_min ?? 0}
              max={cfdData.velocity_max ?? 5}
              colormap="jet"
              orientation="vertical"
              compact
            />
          </div>
        {/if}
      </div>
    {/if}
  </div>

  <!-- Right panel: Layers + CFD + Landmarks -->
  <div class="cockpit-right">
    <div class="cockpit-right-tabs">
      {#each [['layers','Layers'],['cfd','CFD'],['landmarks','LM'],['regions','Regions']] as [key, lbl]}
        <button class="cockpit-tab" class:active={activeRightTab === key}
          on:click={() => activeRightTab = key}>
          {lbl}
        </button>
      {/each}
    </div>

    <div class="sov-card cockpit-right-card">
      <div class="sov-card-body" style="overflow-y: auto; max-height: 600px;">
        {#if activeRightTab === 'layers'}
          <LayerControlPanel
            {layerConfig}
            {layerNames}
            {highlightLayer}
            on:change={(e) => { layerConfig = e.detail; }}
          />
        {:else if activeRightTab === 'cfd'}
          <CfdPanel
            cfdData={cfdData}
            visible={showStreamlines}
            on:visibilityChange={(e) => showStreamlines = e.detail}
            on:runCfd={handleRunCfd}
          />
        {:else if activeRightTab === 'landmarks'}
          <LandmarkPanel landmarkData={landmarks} />
        {:else if activeRightTab === 'regions'}
          <RegionLegend twin={twin} {highlightLayer}
            on:regionHover={(e) => highlightLayer = e.detail}
            on:regionClick={(e) => highlightLayer = highlightLayer === e.detail ? null : e.detail}
          />
        {/if}
      </div>
    </div>

    <!-- Multi-plan comparison mini -->
    {#if showStreamlines && cfdData?.resistance}
      <div class="sov-card" style="margin-top: 8px;">
        <div class="sov-card-header"><span class="sov-card-title">Airflow</span></div>
        <div class="sov-card-body">
          <div class="cfd-mini">
            <div class="cfd-mini-row">
              <span>Peak</span>
              <span class="font-data" style="color: var(--sov-accent);">{cfdData.summary?.peak_velocity?.toFixed(2) ?? '—'} m/s</span>
            </div>
            <div class="cfd-mini-row">
              <span>L/R Ratio</span>
              <span class="font-data">{cfdData.resistance ? (cfdData.resistance.left / cfdData.resistance.right).toFixed(2) : '—'}</span>
            </div>
            <div class="cfd-mini-row">
              <span>Total R</span>
              <span class="font-data">{cfdData.resistance?.total?.toFixed(2) ?? '—'} Pa·s/mL</span>
            </div>
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>

<svelte:window on:keydown={(e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
  const keyMap = { v: 'orbit', m: 'measure', i: 'incision', o: 'osteotomy', g: 'graft' };
  if (keyMap[e.key]) { interactionMode = keyMap[e.key]; }
}} />

<style>
  /* Cockpit layout */
  .cockpit-header { margin-bottom: 8px; }
  .cockpit-breadcrumb { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
  .cockpit-title-row { display: flex; justify-content: space-between; align-items: center; }
  .cockpit-title-actions { display: flex; gap: 6px; }

  .cockpit-ribbon {
    display: flex; align-items: center; gap: 8px; padding: 6px 12px;
    background: var(--sov-bg-card); border: 1px solid var(--sov-border); border-radius: 6px;
    margin-bottom: 8px; overflow-x: auto;
  }
  .ribbon-stat { display: flex; flex-direction: column; align-items: center; min-width: 50px; }
  .ribbon-val { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; color: var(--sov-text-primary); }
  .ribbon-val.active-val { color: #10B981; }
  .ribbon-label { font-size: 9px; color: var(--sov-text-muted); text-transform: uppercase; letter-spacing: 0.03em; }
  .ribbon-sep { width: 1px; height: 24px; background: var(--sov-border-subtle); flex-shrink: 0; }

  .cockpit-toolbar-card { margin-bottom: 8px; }

  .cockpit-layout {
    display: grid;
    grid-template-columns: 260px 1fr 280px;
    gap: 8px;
    min-height: 600px;
  }

  .cockpit-left, .cockpit-right { display: flex; flex-direction: column; gap: 0; min-width: 0; }
  .cockpit-center { position: relative; min-height: 500px; }

  .cockpit-left-tabs, .cockpit-right-tabs {
    display: flex; border-bottom: 1px solid var(--sov-border);
    background: var(--sov-bg-card); border-radius: 6px 6px 0 0;
    border: 1px solid var(--sov-border); border-bottom: none;
  }
  .cockpit-tab {
    flex: 1; padding: 7px 4px; font-family: inherit; font-size: 10px; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.03em; color: var(--sov-text-muted);
    background: transparent; border: none; border-bottom: 2px solid transparent;
    cursor: pointer; text-align: center; transition: all 80ms;
  }
  .cockpit-tab:hover { color: var(--sov-text-secondary); }
  .cockpit-tab.active { color: var(--sov-accent); border-bottom-color: var(--sov-accent); }

  .cockpit-left-card, .cockpit-right-card { border-radius: 0 0 6px 6px; flex: 1; }

  .cockpit-viewer-card { height: 100%; position: relative; overflow: hidden; }

  .cockpit-colorbar-overlay {
    position: absolute; bottom: 20px; right: 16px; z-index: 5;
    padding: 6px 8px; background: rgba(8,10,15,0.8); border-radius: 4px;
    backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,0.05);
  }

  /* Timeline */
  .timeline-list { display: flex; flex-direction: column; }
  .timeline-item { display: flex; gap: 8px; min-height: 32px; }
  .timeline-marker { display: flex; flex-direction: column; align-items: center; width: 10px; flex-shrink: 0; padding-top: 3px; }
  .timeline-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--sov-accent); }
  .timeline-line { width: 1px; flex: 1; background: var(--sov-border); margin: 2px 0; }
  .timeline-content { display: flex; flex-direction: column; gap: 1px; padding-bottom: 8px; }
  .timeline-action { font-size: 11px; font-weight: 500; color: var(--sov-text-primary); }
  .timeline-time { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--sov-text-muted); }
  .timeline-detail { font-size: 10px; color: var(--sov-text-tertiary); }

  /* Plan mini */
  .plan-summary { display: flex; flex-direction: column; gap: 6px; }
  .plan-name { font-size: 14px; font-weight: 600; color: var(--sov-text-primary); }
  .plan-meta { display: flex; align-items: center; gap: 6px; }
  .plan-steps-mini { display: flex; flex-direction: column; gap: 3px; margin-top: 4px; }
  .plan-step-mini { display: flex; align-items: center; gap: 6px; padding: 4px 6px; background: var(--sov-bg-root); border-radius: 3px; }
  .plan-step-num { font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 600; color: var(--sov-accent); width: 16px; }
  .plan-step-name { font-size: 11px; color: var(--sov-text-secondary); }

  /* CFD mini */
  .cfd-mini { display: flex; flex-direction: column; gap: 4px; }
  .cfd-mini-row { display: flex; justify-content: space-between; font-size: 11px; color: var(--sov-text-secondary); }

  /* Responsive */
  @media (max-width: 1100px) {
    .cockpit-layout { grid-template-columns: 220px 1fr 240px; }
  }
  @media (max-width: 900px) {
    .cockpit-layout { grid-template-columns: 1fr; }
    .cockpit-left, .cockpit-right { max-height: 300px; }
  }
</style>
