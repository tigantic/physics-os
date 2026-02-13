<script>
  /**
   * Phase 7 — Full Surgical Cockpit
   *
   * 3-column layout: left (DICOM/Timeline/Plan + FEM), center (LayeredTissueViewer),
   * right (Layers/CFD/Landmarks/Regions). Includes SurgicalToolbar, Pre/Post comparison,
   * and complete simulation trigger flows (CFD, FEM, incision, osteotomy, graft, post-op).
   *
   * Tree-shaking safe: uses Svelte action instead of onMount/onDestroy,
   * callback props instead of on:event.
   */
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
    evaluateSafety, evaluateAesthetics, evaluateFunctional, getHealingTimeline,
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
  import SafetyPanel from '$lib/components/SafetyPanel.svelte';
  import AestheticsPanel from '$lib/components/AestheticsPanel.svelte';
  import HealingPanel from '$lib/components/HealingPanel.svelte';

  // ── Svelte action: replaces onMount/onDestroy ──────────────
  function pageSetup(node) {
    loadAllData();
    return {
      destroy() { clearActiveCase(); }
    };
  }

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

  // Phase 8-9 analytics data
  let safetyData = null;
  let aestheticsData = null;
  let healingData = null;

  // Loading states
  let loadingLayers = false;
  let loadingCfd = false;
  let loadingFem = false;
  let loadingDicom = false;
  let loadingPostOp = false;
  let runningCfd = false;
  let runningFem = false;
  let loadingSafety = false;
  let loadingAesthetics = false;
  let loadingHealing = false;

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
    try {
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
    } catch {
      // Graceful fallback to single mesh
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
    try { cfdData = await loadCfdResults(caseId); } catch { cfdData = null; }
    loadingCfd = false;

    // Load FEM (may not exist yet)
    loadingFem = true;
    try { femData = await loadFemResults(caseId); } catch { femData = null; }
    loadingFem = false;

    // Load DICOM (may not exist)
    loadingDicom = true;
    try { dicomData = await loadDicomData(caseId); } catch { dicomData = null; }
    loadingDicom = false;

    // Load post-op prediction if plan exists
    if ($activePlan?.content_hash) {
      loadingPostOp = true;
      try { postOpData = await loadPostOpPrediction(caseId, $activePlan.content_hash); } catch { postOpData = null; }
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
      layerConfig[name] = { ...(DEFAULTS[key || 'tissue']) };
    }
    layerConfig = { ...layerConfig };
  }

  // ── Simulation Triggers ────────────────────────────────────
  async function handleRunCfd() {
    runningCfd = true;
    try { cfdData = await runCfdSimulation(caseId, $activePlan?.content_hash); } catch { /* empty state */ }
    runningCfd = false;
  }

  async function handleRunFem() {
    runningFem = true;
    try { femData = await runFemSimulation(caseId, $activePlan?.content_hash); } catch { /* empty state */ }
    runningFem = false;
  }

  async function handleGeneratePostOp() {
    if (!$activePlan) return;
    // Use content_hash if available, otherwise fall back to plan name as identifier
    const planHash = $activePlan.content_hash || $activePlan.name || 'default';
    loadingPostOp = true;
    try { postOpData = await generatePostOpPrediction(caseId, planHash); } catch { postOpData = null; }
    loadingPostOp = false;
  }

  // ── Phase 8 Analytics Triggers ─────────────────────────────
  async function handleRunSafety() {
    loadingSafety = true;
    try { safetyData = await evaluateSafety(caseId, $activePlan?.content_hash); } catch { safetyData = null; }
    loadingSafety = false;
  }

  async function handleRunAesthetics() {
    loadingAesthetics = true;
    try { aestheticsData = await evaluateAesthetics(caseId); } catch { aestheticsData = null; }
    loadingAesthetics = false;
  }

  async function handleRunHealing() {
    loadingHealing = true;
    try { healingData = await getHealingTimeline(caseId, $activePlan?.content_hash); } catch { healingData = null; }
    loadingHealing = false;
  }

  // ── Surgical Tool Handlers ─────────────────────────────────
  // NOTE: callback props pass values directly, not wrapped in e.detail
  function handleIncisionPoint(detail) {
    canCommitIncision = (detail.pathLength >= 2);
    canUndoIncision = (detail.pathLength > 0);
  }

  async function handleIncisionCommit(detail) {
    try {
      const result = await executeIncision(caseId, detail.points, 5.0);
      if (result) {
        incisionPaths = [...incisionPaths, { points: detail.points, color: '#EF4444', depth: 5.0 }];
        if (result.updated_mesh) {
          layerData = { ...layerData, ...result.updated_mesh };
        }
      }
    } catch { /* graceful */ }
    canCommitIncision = false;
    canUndoIncision = false;
  }

  function handleIncisionCancel() {
    viewer?.clearIncision();
    canCommitIncision = false;
    canUndoIncision = false;
  }

  async function handleGraftPlace(detail) {
    try {
      const result = await placeGraft(
        caseId, detail.position, detail.normal,
        'septal', { length: 15, width: 10, thickness: 2 }
      );
      if (result?.graft_mesh) {
        layerData = { ...layerData, [`graft_${result.graft_id}`]: result.graft_mesh };
        layerNames = [...layerNames, `graft_${result.graft_id}`];
        layerConfig[`graft_${result.graft_id}`] = { visible: true, opacity: 0.9, wireframe: false };
        layerConfig = { ...layerConfig };
      }
    } catch { /* graceful */ }
  }

  // ── Toolbar Handlers (callback props pass values directly) ─
  function handleModeChange(mode) { interactionMode = mode; }
  function handleViewPreset(preset) { viewer?.setCameraPreset(preset); }
  function handleCommit() { if (interactionMode === 'incision') viewer?.commitIncision(); }

  function handleUndo() {
    if (interactionMode === 'incision') { viewer?.clearIncision(); canCommitIncision = false; }
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

  // ── FEM Field Change (callback prop passes value directly) ─
  function handleFemFieldChange(detail) {
    femActiveField = detail.field;
    femColormap = detail.colormap;
  }

  function timeAgo(ts) {
    if (!ts) return '';
    try { return new Date(ts).toLocaleString(); } catch { return String(ts); }
  }

  // ── Computed ───────────────────────────────────────────────
  $: twin = $twinStore.data;
  $: landmarks = $landmarksStore.data;
  $: timeline = $timelineStore.data;
  $: viz = $visualizationStore.data;

  // Build region data from visualization endpoint (twin summary has no mesh/regions)
  const REGION_NAMES = [
    { structure: 'Nasal Bone', material: 'bone' },
    { structure: 'Upper Lateral Cartilage', material: 'cartilage' },
    { structure: 'Lower Lateral Cartilage', material: 'cartilage' },
    { structure: 'Septal Cartilage', material: 'cartilage' },
    { structure: 'Dorsal Skin', material: 'skin' },
    { structure: 'Tip Skin', material: 'skin' },
    { structure: 'Soft Tissue', material: 'soft_tissue' },
    { structure: 'Alar Lobule', material: 'skin' },
    { structure: 'Columella', material: 'cartilage' },
  ];

  $: regionTwin = (() => {
    const rc = viz?.region_colors;
    if (!rc || Object.keys(rc).length === 0) return twin;
    // Build a twin-compatible object with mesh.regions
    const regions = {};
    for (const [id, color] of Object.entries(rc)) {
      const meta = REGION_NAMES[parseInt(id)] || { structure: `Region ${id}`, material: 'tissue' };
      regions[id] = { structure: meta.structure, material: meta.material, color };
    }
    return { ...twin, mesh: { ...(twin?.mesh || {}), regions } };
  })();

  $: streamlineData = (showStreamlines && cfdData) ? cfdData.streamlines || cfdData : null;
  $: scalarFieldData = (showScalarField && femData?.fields?.[femActiveField]) ? femData.fields[femActiveField] : null;

  $: hasPlan = $activePlan != null;
  $: meshStats = twin?.mesh ?? null;
  $: totalVerts = layerData ? Object.values(layerData).reduce((sum, l) => sum + (l.n_vertices || l.positions?.length || 0), 0) : 0;
  $: totalTris = layerData ? Object.values(layerData).reduce((sum, l) => sum + (l.n_triangles || l.indices?.length || 0), 0) : 0;
</script>

<!-- Top bar -->
<div class="sov-page-header cockpit-header" use:pageSetup>
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
        disabled={!hasPlan}
        title={!hasPlan ? 'Create a plan first to enable Pre/Post comparison' : postOpData ? 'Toggle Pre/Post comparison view' : 'Will generate post-op prediction first'}
        on:click={async () => {
          if (!showPrePost && !postOpData && hasPlan) {
            await handleGeneratePostOp();
          }
          showPrePost = !showPrePost;
        }}>
        {showPrePost ? '◉ Single View' : '◫ Pre/Post'}
      </button>
      {#if hasPlan && !postOpData}
        <button class="sov-btn sov-btn-sm sov-btn-primary" on:click={handleGeneratePostOp}
          disabled={loadingPostOp}>
          {loadingPostOp ? 'Computing…' : '▶ Generate Post-Op'}
        </button>
      {:else if hasPlan && postOpData}
        <span class="sov-badge sov-badge-accent" style="font-size: 10px;">Post-Op Ready</span>
      {/if}
    </div>
  </div>
</div>

<!-- Stats ribbon -->
<div class="cockpit-ribbon">
  <div class="ribbon-stat">
    <span class="ribbon-val ribbon-val-num">{totalVerts.toLocaleString()}</span>
    <span class="ribbon-label">Vertices</span>
  </div>
  <div class="ribbon-sep"></div>
  <div class="ribbon-stat">
    <span class="ribbon-val ribbon-val-num">{totalTris.toLocaleString()}</span>
    <span class="ribbon-label">Triangles</span>
  </div>
  <div class="ribbon-sep"></div>
  <div class="ribbon-stat">
    <span class="ribbon-val ribbon-val-num">{layerNames.length}</span>
    <span class="ribbon-label">Layers</span>
  </div>
  <div class="ribbon-sep"></div>
  <div class="ribbon-stat">
    <span class="ribbon-val ribbon-val-num">{landmarks?.landmarks?.length ?? '—'}</span>
    <span class="ribbon-label">Landmarks</span>
  </div>
  <div class="ribbon-sep"></div>
  <div class="ribbon-stat ribbon-stat-status">
    <span class="ribbon-indicator" class:ribbon-indicator-on={cfdData != null}></span>
    <span class="ribbon-label">CFD</span>
  </div>
  <div class="ribbon-sep"></div>
  <div class="ribbon-stat ribbon-stat-status">
    <span class="ribbon-indicator" class:ribbon-indicator-on={femData != null}></span>
    <span class="ribbon-label">FEM</span>
  </div>
  <div class="ribbon-sep"></div>
  <div class="ribbon-stat ribbon-stat-status">
    <span class="ribbon-indicator" class:ribbon-indicator-on={dicomData != null}></span>
    <span class="ribbon-label">DICOM</span>
  </div>
  <div class="ribbon-sep"></div>
  <div class="ribbon-stat ribbon-stat-status">
    <span class="ribbon-indicator" class:ribbon-indicator-on={postOpData != null}></span>
    <span class="ribbon-label">Post-Op</span>
  </div>
  {#if incisionPaths.length > 0}
    <div class="ribbon-sep"></div>
    <div class="ribbon-stat">
      <span class="ribbon-val ribbon-val-num" style="color: #EF4444;">{incisionPaths.length}</span>
      <span class="ribbon-label">Incisions</span>
    </div>
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
    onModeChange={handleModeChange}
    onCommit={handleCommit}
    onUndo={handleUndo}
    onCancel={handleCancel}
    onStreamlinesToggle={(val) => showStreamlines = val}
    onScalarToggle={(val) => showScalarField = val}
    onLandmarksToggle={(val) => showLandmarks = val}
    onLabelsToggle={(val) => showLabels = val}
    onIncisionsToggle={(val) => showIncisions = val}
    onClipToggle={(val) => { clipEnabled = val; clipConfig = { ...clipConfig, enabled: val }; }}
    onScalarTypeChange={(val) => { femActiveField = val; }}
    onResetView={() => viewer?.resetView()}
    onViewPreset={handleViewPreset}
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
          <DicomViewer dicomData={dicomData} />
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
                      {#if event.timestamp}<span class="timeline-time">{timeAgo(event.timestamp)}</span>{/if}
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
    <div class="sov-card cockpit-left-card cockpit-side-card-glass" style="margin-top: 8px;">
      <div class="sov-card-body" style="overflow-y: auto; max-height: 400px;">
        <FemPanel
          femData={femData}
          visible={showScalarField}
          activeField={femActiveField}
          colormap={femColormap}
          onVisibilityChange={(val) => showScalarField = val}
          onFieldChange={handleFemFieldChange}
          onRunFem={handleRunFem}
        />
      </div>
    </div>
  </div>

  <!-- Center: 3D Viewer (main) -->
  <div class="cockpit-center">
    {#if loadingLayers}
      <div class="cockpit-loading-card">
        <div class="cockpit-skeleton-grid">
          <div class="cockpit-skeleton cockpit-skeleton-lg"></div>
          <div class="cockpit-skeleton cockpit-skeleton-sm"></div>
          <div class="cockpit-skeleton cockpit-skeleton-md"></div>
        </div>
        <div class="cockpit-loading-label">
          <div class="cockpit-loading-pulse"></div>
          <span>Initializing tissue layers…</span>
        </div>
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
            onIncisionPoint={handleIncisionPoint}
            onIncisionCommit={handleIncisionCommit}
            onGraftPlace={handleGraftPlace}
            onModeChange={(mode) => interactionMode = mode}
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
          onIncisionPoint={handleIncisionPoint}
          onIncisionCommit={handleIncisionCommit}
          onGraftPlace={handleGraftPlace}
          onModeChange={(mode) => interactionMode = mode}
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

  <!-- Right panel: Layers + CFD + Landmarks + Analytics -->
  <div class="cockpit-right">
    <div class="cockpit-right-tabs">
      {#each [['layers','Layers'],['cfd','CFD'],['landmarks','LM'],['regions','Regions'],['safety','Safety'],['aesthetics','Aes'],['healing','Heal']] as [key, lbl]}
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
            onChange={(val) => { layerConfig = val; }}
          />
        {:else if activeRightTab === 'cfd'}
          <CfdPanel
            cfdData={cfdData}
            visible={showStreamlines}
            onVisibilityChange={(val) => showStreamlines = val}
            onRunCfd={handleRunCfd}
          />
        {:else if activeRightTab === 'landmarks'}
          <LandmarkPanel landmarkData={landmarks} />
        {:else if activeRightTab === 'regions'}
          <RegionLegend twin={regionTwin} highlightRegion={highlightLayer}
            onRegionHover={(id) => highlightLayer = id}
            onRegionClick={(id) => highlightLayer = highlightLayer === id ? null : id}
          />
        {:else if activeRightTab === 'safety'}
          <SafetyPanel
            {safetyData}
            loading={loadingSafety}
            onRunSafety={handleRunSafety}
          />
        {:else if activeRightTab === 'aesthetics'}
          <AestheticsPanel
            {aestheticsData}
            loading={loadingAesthetics}
            onRunAesthetics={handleRunAesthetics}
          />
        {:else if activeRightTab === 'healing'}
          <HealingPanel
            {healingData}
            loading={loadingHealing}
            onRunHealing={handleRunHealing}
          />
        {/if}
      </div>
    </div>

    <!-- Multi-plan comparison mini -->
    {#if showStreamlines && cfdData?.resistance}
      <div class="sov-card cockpit-side-card-glass" style="margin-top: 8px;">
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
  /* ── Keyframes ────────────────────────────────────────────── */
  @keyframes cockpit-fade-in {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes cockpit-shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
  }
  @keyframes cockpit-pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.4; }
  }
  @keyframes cockpit-indicator-glow {
    0%, 100% { box-shadow: 0 0 4px 1px rgba(16,185,129,0.3); }
    50%      { box-shadow: 0 0 8px 3px rgba(16,185,129,0.5); }
  }

  /* ── Cockpit Header ───────────────────────────────────────── */
  .cockpit-header {
    margin-bottom: 8px;
    animation: cockpit-fade-in 0.3s ease-out;
  }
  .cockpit-breadcrumb { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
  .cockpit-title-row { display: flex; justify-content: space-between; align-items: center; }
  .cockpit-title-actions { display: flex; gap: 6px; }

  /* ── Stats Ribbon — Glass morphism ────────────────────────── */
  .cockpit-ribbon {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 16px;
    background: linear-gradient(135deg, rgba(17,19,24,0.85) 0%, rgba(13,15,20,0.92) 100%);
    border: 1px solid rgba(59,130,246,0.1);
    border-radius: 8px;
    margin-bottom: 8px;
    overflow-x: auto;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow:
      0 1px 3px rgba(0,0,0,0.3),
      inset 0 1px 0 rgba(255,255,255,0.03);
    animation: cockpit-fade-in 0.35s ease-out 0.05s both;
  }
  .ribbon-stat {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 56px;
    gap: 2px;
    transition: transform 0.15s ease;
  }
  .ribbon-stat:hover { transform: translateY(-1px); }
  .ribbon-stat-status { min-width: 44px; }

  .ribbon-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: var(--sov-text-primary);
    line-height: 1.2;
  }
  .ribbon-val-num {
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
  }
  .ribbon-label {
    font-size: 9px;
    color: var(--sov-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
    line-height: 1;
  }
  .ribbon-sep {
    width: 1px;
    height: 28px;
    background: linear-gradient(180deg, transparent, var(--sov-border), transparent);
    flex-shrink: 0;
  }

  /* Status indicator dot (replaces ✓/— text) */
  .ribbon-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--sov-border);
    transition: background 0.3s ease, box-shadow 0.3s ease;
  }
  .ribbon-indicator-on {
    background: #10B981;
    animation: cockpit-indicator-glow 2.5s ease-in-out infinite;
  }

  /* ── Toolbar Card ─────────────────────────────────────────── */
  .cockpit-toolbar-card {
    margin-bottom: 8px;
    animation: cockpit-fade-in 0.35s ease-out 0.1s both;
  }

  /* ── Main Layout ──────────────────────────────────────────── */
  .cockpit-layout {
    display: grid;
    grid-template-columns: 260px 1fr 280px;
    gap: 8px;
    min-height: 600px;
    animation: cockpit-fade-in 0.35s ease-out 0.15s both;
  }

  .cockpit-left, .cockpit-right {
    display: flex;
    flex-direction: column;
    gap: 0;
    min-width: 0;
  }
  .cockpit-center { position: relative; min-height: 500px; }

  /* ── Tab Bars — refined with slide indicator ──────────────── */
  .cockpit-left-tabs, .cockpit-right-tabs {
    display: flex;
    border-bottom: 1px solid var(--sov-border);
    background: var(--sov-bg-card);
    border-radius: 8px 8px 0 0;
    border: 1px solid var(--sov-border);
    border-bottom: none;
    overflow: hidden;
  }
  .cockpit-tab {
    flex: 1;
    padding: 8px 4px;
    font-family: inherit;
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--sov-text-muted);
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    text-align: center;
    transition: color 0.2s ease, border-color 0.2s ease, background 0.2s ease;
    position: relative;
  }
  .cockpit-tab:hover {
    color: var(--sov-text-secondary);
    background: rgba(59,130,246,0.03);
  }
  .cockpit-tab.active {
    color: var(--sov-accent);
    border-bottom-color: var(--sov-accent);
    background: rgba(59,130,246,0.05);
  }

  /* ── Side Panel Cards — Glass effect ──────────────────────── */
  .cockpit-left-card, .cockpit-right-card {
    border-radius: 0 0 8px 8px;
    flex: 1;
    background: linear-gradient(180deg, rgba(17,19,24,0.95) 0%, rgba(13,15,20,0.98) 100%);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    transition: box-shadow 0.25s ease;
  }
  .cockpit-left-card:hover, .cockpit-right-card:hover {
    box-shadow: 0 0 0 1px rgba(59,130,246,0.08), 0 4px 16px rgba(0,0,0,0.2);
  }

  /* ── Viewer — Ambient glow frame ──────────────────────────── */
  .cockpit-viewer-card {
    height: 100%;
    position: relative;
    overflow: hidden;
    border-radius: 8px;
    border: 1px solid rgba(59,130,246,0.08);
    box-shadow:
      0 0 20px rgba(59,130,246,0.04),
      0 2px 8px rgba(0,0,0,0.3);
    transition: box-shadow 0.3s ease;
  }
  .cockpit-viewer-card:hover {
    box-shadow:
      0 0 30px rgba(59,130,246,0.07),
      0 4px 16px rgba(0,0,0,0.3);
  }

  /* ── Colorbar Overlay — Glass morphism ────────────────────── */
  .cockpit-colorbar-overlay {
    position: absolute;
    bottom: 20px;
    right: 16px;
    z-index: 5;
    padding: 8px 10px;
    background: linear-gradient(180deg, rgba(8,10,15,0.75) 0%, rgba(8,10,15,0.9) 100%);
    border-radius: 6px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }

  /* ── Skeleton Loading ─────────────────────────────────────── */
  .cockpit-loading-card {
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 24px;
    background: var(--sov-bg-card);
    border: 1px solid var(--sov-border);
    border-radius: 8px;
  }
  .cockpit-skeleton-grid {
    display: flex;
    flex-direction: column;
    gap: 12px;
    width: 60%;
    max-width: 320px;
  }
  .cockpit-skeleton {
    border-radius: 4px;
    background: linear-gradient(90deg, var(--sov-bg-elevated) 25%, var(--sov-bg-hover) 50%, var(--sov-bg-elevated) 75%);
    background-size: 200% 100%;
    animation: cockpit-shimmer 1.8s ease-in-out infinite;
  }
  .cockpit-skeleton-lg  { height: 120px; }
  .cockpit-skeleton-md  { height: 24px; width: 80%; }
  .cockpit-skeleton-sm  { height: 16px; width: 50%; }

  .cockpit-loading-label {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--sov-text-muted);
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.02em;
  }
  .cockpit-loading-pulse {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--sov-accent);
    animation: cockpit-pulse 1.5s ease-in-out infinite;
  }

  /* ── Timeline ─────────────────────────────────────────────── */
  .timeline-list { display: flex; flex-direction: column; }
  .timeline-item {
    display: flex;
    gap: 10px;
    min-height: 36px;
    transition: background 0.15s ease;
    padding: 2px 4px;
    border-radius: 4px;
  }
  .timeline-item:hover { background: rgba(59,130,246,0.03); }
  .timeline-marker { display: flex; flex-direction: column; align-items: center; width: 12px; flex-shrink: 0; padding-top: 4px; }
  .timeline-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--sov-accent);
    box-shadow: 0 0 6px rgba(59,130,246,0.3);
    flex-shrink: 0;
  }
  .timeline-line {
    width: 1px;
    flex: 1;
    background: linear-gradient(180deg, var(--sov-accent-dim), var(--sov-border));
    margin: 3px 0;
  }
  .timeline-content { display: flex; flex-direction: column; gap: 2px; padding-bottom: 10px; }
  .timeline-action { font-size: 11px; font-weight: 600; color: var(--sov-text-primary); letter-spacing: 0.01em; }
  .timeline-time { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--sov-text-muted); }
  .timeline-detail { font-size: 10px; color: var(--sov-text-tertiary); line-height: 1.4; }

  /* ── Plan Mini ────────────────────────────────────────────── */
  .plan-summary { display: flex; flex-direction: column; gap: 8px; }
  .plan-name {
    font-size: 14px;
    font-weight: 700;
    color: var(--sov-text-primary);
    letter-spacing: -0.01em;
  }
  .plan-meta { display: flex; align-items: center; gap: 6px; }
  .plan-steps-mini { display: flex; flex-direction: column; gap: 4px; margin-top: 4px; }
  .plan-step-mini {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 8px;
    background: var(--sov-bg-root);
    border-radius: 4px;
    border: 1px solid transparent;
    transition: border-color 0.15s ease, background 0.15s ease;
  }
  .plan-step-mini:hover {
    border-color: rgba(59,130,246,0.1);
    background: var(--sov-bg-elevated);
  }
  .plan-step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    color: var(--sov-accent);
    width: 18px;
    text-align: center;
  }
  .plan-step-name { font-size: 11px; color: var(--sov-text-secondary); }

  /* ── CFD Mini ─────────────────────────────────────────────── */
  .cfd-mini { display: flex; flex-direction: column; gap: 6px; }
  .cfd-mini-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    color: var(--sov-text-secondary);
    padding: 3px 0;
    border-bottom: 1px solid rgba(255,255,255,0.02);
  }
  .cfd-mini-row:last-child { border-bottom: none; }

  /* ── Glass card variant for secondary panels ──────────────── */
  .cockpit-side-card-glass {
    background: linear-gradient(180deg, rgba(17,19,24,0.9) 0%, rgba(13,15,20,0.95) 100%);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-color: rgba(59,130,246,0.06);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }
  .cockpit-side-card-glass:hover {
    border-color: rgba(59,130,246,0.12);
    box-shadow: 0 2px 12px rgba(0,0,0,0.15);
  }

  /* ── Responsive ───────────────────────────────────────────── */
  @media (max-width: 1100px) {
    .cockpit-layout { grid-template-columns: 220px 1fr 240px; }
  }
  @media (max-width: 900px) {
    .cockpit-layout {
      grid-template-columns: 1fr;
      gap: 6px;
    }
    .cockpit-left, .cockpit-right { max-height: 300px; }
    .cockpit-ribbon { gap: 8px; padding: 6px 12px; }
  }
</style>
