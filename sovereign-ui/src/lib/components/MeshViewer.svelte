<script>
  import { tick } from 'svelte';
  import { REGION_PALETTE, getRegionColor as _getRegionColor } from '$lib/constants';

  // ── Props ──────────────────────────────────────────────────
  /** @type {import('$lib/api-client').MeshData | null} */
  export let meshData = null;

  /** @type {import('$lib/api-client').LandmarksResponse | null} */
  export let landmarkData = null;

  /** @type {Record<string, string> | null} */
  export let regionColors = null;

  /** @type {boolean} */
  export let showLandmarks = true;

  /** @type {boolean} */
  export let showWireframe = false;

  /** @type {boolean} */
  export let showLabels = true;

  /** @type {number} tissue opacity 0-1 */
  export let tissueOpacity = 1.0;

  /** @type {string | null} */
  export let highlightRegion = null;

  /** @type {Set<string>} hidden region IDs */
  export let hiddenRegions = new Set();

  /** @type {boolean} */
  export let clipEnabled = false;

  /** @type {{ axis: 'x'|'y'|'z', position: number }} */
  export let clipConfig = { axis: 'y', position: 0 };

  /** @type {'orbit' | 'measure'} */
  export let interactionMode = 'orbit';

  /** @type {(stats: { vertices: number, triangles: number, regions: number, landmarks: number }) => void} */
  export let onMeshLoaded = () => {};

  /** @type {(detail: { distance: number, points: number[][] }) => void} */
  export let onMeasurement = () => {};

  // ── Internal State ─────────────────────────────────────────
  let container;
  let canvas;
  let labelOverlay;
  let THREE;
  let OrbitControls;
  let renderer, scene, camera, controls;
  let meshGroup, landmarkGroup, measureGroup;
  let clipPlane = null;
  let clipHelper = null;
  let animFrameId;
  let loaded = false;
  let loadError = '';

  /**
   * Drives {#await} in template — survives Svelte 5 runes:false tree-shaking.
   * onMount is stripped from compiled output; Svelte actions (use:) are
   * template-bound and cannot be tree-shaken.
   */
  let initPromise = new Promise(() => {});
  let stats = { vertices: 0, triangles: 0, regions: 0, landmarks: 0 };

  // Measurement state
  let measurePoints = [];
  let measureDistance = null;
  let raycaster = null;
  let mouse = null;

  // Label positions (projected to screen)
  let labelPositions = [];

  // ── Region color palette (from shared constants) ───────────
  function getRegionColor(regionId) {
    return _getRegionColor(regionId, regionColors);
  }

  // ── Three.js Setup ─────────────────────────────────────────
  async function initThree() {
    try {
      THREE = await import('three');
      const { OrbitControls: OC } = await import('three/examples/jsm/controls/OrbitControls.js');
      OrbitControls = OC;
    } catch (err) {
      if (window.THREE) THREE = window.THREE;
      else { loadError = 'Three.js not available. Install: npm install three'; return; }
    }

    scene = new THREE.Scene();
    scene.background = new THREE.Color('#08080A');

    const aspect = container.clientWidth / container.clientHeight;
    camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
    camera.position.set(0, 0, 200);

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    renderer.localClippingEnabled = true;

    if (OrbitControls) {
      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.rotateSpeed = 0.5;
      controls.zoomSpeed = 0.8;
      controls.panSpeed = 0.4;
      controls.minDistance = 10;
      controls.maxDistance = 500;
    }

    // Lighting
    scene.add(new THREE.AmbientLight('#404050', 0.6));
    const keyLight = new THREE.DirectionalLight('#E8E8EC', 0.9);
    keyLight.position.set(50, 80, 100);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight('#6B8CCC', 0.4);
    fillLight.position.set(-60, -30, -50);
    scene.add(fillLight);
    const rimLight = new THREE.DirectionalLight('#3B82F6', 0.3);
    rimLight.position.set(0, 100, -100);
    scene.add(rimLight);

    // Grid
    const grid = new THREE.GridHelper(200, 20, '#1F2937', '#111318');
    grid.position.y = -80;
    scene.add(grid);

    // Groups
    meshGroup = new THREE.Group();
    landmarkGroup = new THREE.Group();
    measureGroup = new THREE.Group();
    scene.add(meshGroup);
    scene.add(landmarkGroup);
    scene.add(measureGroup);

    // Raycaster for measurement
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    loaded = true;
    animate();
  }

  // ── Build mesh ─────────────────────────────────────────────
  function buildMesh() {
    if (!THREE || !meshGroup || !meshData) return;

    while (meshGroup.children.length > 0) {
      const c = meshGroup.children[0];
      c.geometry?.dispose();
      if (Array.isArray(c.material)) c.material.forEach(m => m.dispose());
      else c.material?.dispose();
      meshGroup.remove(c);
    }

    const { positions, indices, region_ids, n_vertices, n_triangles } = meshData;
    if (!positions || positions.length === 0) { loadError = 'No vertices'; return; }

    const posArray = new Float32Array(positions.length * 3);
    for (let i = 0; i < positions.length; i++) {
      posArray[i * 3] = positions[i][0];
      posArray[i * 3 + 1] = positions[i][1];
      posArray[i * 3 + 2] = positions[i][2];
    }

    const idxArray = new Uint32Array(indices.length * 3);
    for (let i = 0; i < indices.length; i++) {
      idxArray[i * 3] = indices[i][0];
      idxArray[i * 3 + 1] = indices[i][1];
      idxArray[i * 3 + 2] = indices[i][2];
    }

    // M5: Build indexed geometry, then convert to non-indexed to avoid
    // vertex color bleeding at shared edges between regions.
    let geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
    geometry.setIndex(new THREE.BufferAttribute(idxArray, 1));
    geometry.computeVertexNormals();
    geometry = geometry.toNonIndexed();

    // Allocate color buffer on non-indexed geometry (each face owns its vertices)
    const vertexCount = geometry.getAttribute('position').count;
    const colorArray = new Float32Array(vertexCount * 3);
    geometry.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));

    // Collect unique regions for stats
    const uniqueRegions = new Set();
    if (region_ids) {
      for (const id of region_ids) uniqueRegions.add(String(id ?? 0));
    }
    applyRegionColors(geometry, region_ids);

    geometry.computeBoundingSphere();

    // Clipping planes
    const clippingPlanes = [];
    if (clipEnabled && clipPlane) {
      clippingPlanes.push(clipPlane);
    }

    // Main material — tissue transparency support
    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      shininess: 30,
      specular: new THREE.Color('#222233'),
      flatShading: false,
      transparent: tissueOpacity < 1.0,
      opacity: tissueOpacity,
      depthWrite: tissueOpacity >= 0.95,
      clippingPlanes,
    });

    meshGroup.add(new THREE.Mesh(geometry, material));

    // Wireframe
    if (showWireframe) {
      const wireMat = new THREE.MeshBasicMaterial({
        color: '#1F2937',
        wireframe: true,
        transparent: true,
        opacity: 0.12,
        clippingPlanes,
      });
      const wire = new THREE.Mesh(geometry.clone(), wireMat);
      wire.name = 'wireframe';
      meshGroup.add(wire);
    }

    // Center camera on first build
    const sphere = geometry.boundingSphere;
    if (sphere && !meshGroup.userData.centered) {
      camera.position.set(sphere.center.x, sphere.center.y, sphere.center.z + sphere.radius * 2.5);
      if (controls) { controls.target.copy(sphere.center); controls.update(); }
      meshGroup.userData.centered = true;
    }

    stats = {
      vertices: n_vertices || positions.length,
      triangles: n_triangles || indices.length,
      regions: uniqueRegions.size,
      landmarks: landmarkData?.landmarks?.length ?? 0,
    };
    onMeshLoaded(stats);
  }

  /** Apply per-face region colors to a non-indexed geometry's color attribute. */
  function applyRegionColors(geometry, region_ids) {
    const colorAttr = geometry.getAttribute('color');
    if (!colorAttr) return;
    const arr = colorAttr.array;
    const faceCount = colorAttr.count / 3;

    if (region_ids && region_ids.length > 0) {
      for (let f = 0; f < faceCount; f++) {
        const regionId = String(region_ids[f] ?? 0);
        const isHidden = hiddenRegions.has(regionId);
        const isHighlighted = highlightRegion === regionId;
        const hex = getRegionColor(regionId);
        const color = new THREE.Color(hex);

        if (isHidden) {
          color.setRGB(0.04, 0.04, 0.06);
        } else if (isHighlighted) {
          color.lerp(new THREE.Color('#FFFFFF'), 0.3);
        }

        for (let v = 0; v < 3; v++) {
          const idx = (f * 3 + v) * 3;
          arr[idx]     = color.r;
          arr[idx + 1] = color.g;
          arr[idx + 2] = color.b;
        }
      }
    } else {
      const dc = new THREE.Color('#3B82F6');
      for (let i = 0; i < colorAttr.count; i++) {
        arr[i * 3]     = dc.r;
        arr[i * 3 + 1] = dc.g;
        arr[i * 3 + 2] = dc.b;
      }
    }
    colorAttr.needsUpdate = true;
  }

  /** M4: Update vertex colors and material opacity without rebuilding geometry. */
  function updateMeshColors() {
    if (!THREE || !meshGroup || !meshData || meshGroup.children.length === 0) return;
    const mainMesh = meshGroup.children.find(c => c.name !== 'wireframe');
    if (!mainMesh) return;

    applyRegionColors(mainMesh.geometry, meshData.region_ids);

    mainMesh.material.transparent = tissueOpacity < 1.0;
    mainMesh.material.opacity = tissueOpacity;
    mainMesh.material.depthWrite = tissueOpacity >= 0.95;
    mainMesh.material.needsUpdate = true;
  }

  // ── Clip plane ─────────────────────────────────────────────
  function updateClipPlane() {
    if (!THREE) return;
    const normals = { x: [1,0,0], y: [0,1,0], z: [0,0,1] };
    const n = normals[clipConfig.axis] || normals.y;
    clipPlane = new THREE.Plane(
      new THREE.Vector3(n[0], n[1], n[2]),
      -clipConfig.position
    );

    // Visual helper
    if (clipHelper) { scene.remove(clipHelper); clipHelper.geometry?.dispose(); clipHelper.material?.dispose(); }
    if (clipEnabled) {
      const helperGeo = new THREE.PlaneGeometry(200, 200);
      const helperMat = new THREE.MeshBasicMaterial({
        color: '#3B82F6',
        transparent: true,
        opacity: 0.06,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      clipHelper = new THREE.Mesh(helperGeo, helperMat);

      if (clipConfig.axis === 'x') clipHelper.rotation.y = Math.PI / 2;
      else if (clipConfig.axis === 'z') clipHelper.rotation.x = Math.PI / 2;

      clipHelper.position[clipConfig.axis] = clipConfig.position;
      scene.add(clipHelper);
    }
  }

  // ── Landmarks ──────────────────────────────────────────────
  // P2: Shared geometry/materials across all landmarks to reduce GPU allocations
  let lmSphereGeo = null;
  let lmMaterials = null;
  let lmRingMat = null;

  function buildLandmarks() {
    if (!THREE || !landmarkGroup || !landmarkData) return;

    while (landmarkGroup.children.length > 0) {
      const c = landmarkGroup.children[0];
      // Only dispose per-instance ring geometries; shared resources freed in onDestroy
      if (c.geometry?.type === 'RingGeometry') c.geometry.dispose();
      landmarkGroup.remove(c);
    }

    if (!showLandmarks || !landmarkData.landmarks) return;

    if (!lmSphereGeo) lmSphereGeo = new THREE.SphereGeometry(1.2, 12, 12);
    if (!lmMaterials) {
      lmMaterials = {
        high: new THREE.MeshBasicMaterial({ color: new THREE.Color('#10B981'), transparent: true, opacity: 0.85 }),
        mid:  new THREE.MeshBasicMaterial({ color: new THREE.Color('#F59E0B'), transparent: true, opacity: 0.85 }),
        low:  new THREE.MeshBasicMaterial({ color: new THREE.Color('#EF4444'), transparent: true, opacity: 0.85 }),
      };
    }
    if (!lmRingMat) {
      lmRingMat = new THREE.MeshBasicMaterial({ color: '#10B981', side: THREE.DoubleSide, transparent: true, opacity: 0.4 });
    }

    for (const lm of landmarkData.landmarks) {
      if (!lm.position || lm.position.length < 3) continue;
      const mat = lm.confidence >= 0.8 ? lmMaterials.high : lm.confidence >= 0.5 ? lmMaterials.mid : lmMaterials.low;

      const sphere = new THREE.Mesh(lmSphereGeo, mat);
      sphere.position.set(lm.position[0], lm.position[1], lm.position[2]);
      sphere.name = lm.type;
      sphere.userData = { type: lm.type, confidence: lm.confidence, position: lm.position };
      landmarkGroup.add(sphere);

      if (lm.confidence >= 0.9) {
        const ringGeo = new THREE.RingGeometry(1.8, 2.2, 16);
        const ring = new THREE.Mesh(ringGeo, lmRingMat);
        ring.position.copy(sphere.position);
        landmarkGroup.add(ring);
      }
    }
    stats.landmarks = landmarkData.landmarks.length;
  }

  // ── Measurement Tool ───────────────────────────────────────
  function handleCanvasClick(e) {
    if (interactionMode !== 'measure' || !raycaster || !camera) return;

    const rect = canvas.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const meshes = meshGroup.children.filter(c => c.name !== 'wireframe');
    const intersects = raycaster.intersectObjects(meshes);

    if (intersects.length > 0) {
      const point = intersects[0].point.clone();
      measurePoints = [...measurePoints, point];

      // Add sphere at click
      const dotGeo = new THREE.SphereGeometry(0.8, 8, 8);
      const dotMat = new THREE.MeshBasicMaterial({ color: '#F59E0B' });
      const dot = new THREE.Mesh(dotGeo, dotMat);
      dot.position.copy(point);
      measureGroup.add(dot);

      if (measurePoints.length === 2) {
        // Draw line
        const lineGeo = new THREE.BufferGeometry().setFromPoints(measurePoints);
        const lineMat = new THREE.LineBasicMaterial({ color: '#F59E0B', linewidth: 2 });
        measureGroup.add(new THREE.Line(lineGeo, lineMat));

        measureDistance = measurePoints[0].distanceTo(measurePoints[1]);
        onMeasurement({ distance: measureDistance, points: measurePoints.map(p => [p.x, p.y, p.z]) });
      }

      if (measurePoints.length > 2) {
        clearMeasurement();
      }
    }
  }

  export function clearMeasurement() {
    measurePoints = [];
    measureDistance = null;
    while (measureGroup.children.length > 0) {
      const c = measureGroup.children[0];
      c.geometry?.dispose(); c.material?.dispose();
      measureGroup.remove(c);
    }
  }

  // ── Label Projection ───────────────────────────────────────
  function updateLabelPositions() {
    if (!showLabels || !landmarkData?.landmarks || !camera || !container) {
      labelPositions = [];
      return;
    }

    const w = container.clientWidth;
    const h = container.clientHeight;
    const newPositions = [];

    for (const lm of landmarkData.landmarks) {
      if (!lm.position || lm.position.length < 3) continue;
      const vec = new THREE.Vector3(lm.position[0], lm.position[1], lm.position[2]);
      vec.project(camera);

      const x = (vec.x * 0.5 + 0.5) * w;
      const y = (-vec.y * 0.5 + 0.5) * h;

      if (vec.z < 1 && x > 0 && x < w && y > 0 && y < h) {
        newPositions.push({
          type: lm.type,
          confidence: lm.confidence,
          x, y,
          visible: showLandmarks,
        });
      }
    }

    labelPositions = newPositions;
  }

  // ── Animation ──────────────────────────────────────────────
  function animate() {
    animFrameId = requestAnimationFrame(animate);
    if (document.hidden) return; // P6: Skip rendering when tab is not visible
    if (controls) controls.update();

    // Face landmark rings toward camera
    if (landmarkGroup) {
      landmarkGroup.children.forEach(c => {
        if (c.geometry?.type === 'RingGeometry') c.lookAt(camera.position);
      });
    }

    renderer.render(scene, camera);

    // Update label overlays
    if (showLabels && landmarkData) updateLabelPositions();
  }

  function handleResize() {
    if (!container || !renderer || !camera) return;
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  // ── Public API ─────────────────────────────────────────────
  export function resetView() {
    if (!meshGroup || !controls || !camera) return;
    const mesh = meshGroup.children[0];
    if (mesh?.geometry?.boundingSphere) {
      const { center, radius } = mesh.geometry.boundingSphere;
      camera.position.set(center.x, center.y, center.z + radius * 2.5);
      controls.target.copy(center);
      controls.update();
    }
    meshGroup.userData.centered = false;
  }

  export function toggleWireframe() {
    showWireframe = !showWireframe;
    if (meshData) buildMesh();
  }

  export function toggleLandmarks() {
    showLandmarks = !showLandmarks;
    buildLandmarks();
  }

  // ── Reactivity ─────────────────────────────────────────────
  // Full rebuild only for structural changes (new mesh, clip, wireframe)
  $: if (loaded && meshData) buildMesh();
  $: if (loaded && landmarkData) buildLandmarks();
  $: if (loaded && landmarkGroup) landmarkGroup.visible = showLandmarks;
  $: if (loaded && clipEnabled !== undefined) { updateClipPlane(); if (meshData) buildMesh(); }
  $: if (loaded && clipConfig) { updateClipPlane(); if (meshData) buildMesh(); }
  // M4: Color-only updates — no geometry rebuild needed
  $: if (loaded && tissueOpacity !== undefined && meshData) updateMeshColors();
  $: if (loaded && hiddenRegions && meshData) updateMeshColors();
  $: if (loaded && highlightRegion !== undefined && meshData) updateMeshColors();
  $: if (loaded) {
    if (controls) controls.enabled = interactionMode === 'orbit';
  }

  /**
   * Svelte action — runs when the container div mounts, returns destroy for
   * cleanup.  Actions are template-bound and survive tree-shaking.
   */
  function viewerSetup(node) {
    container = node;
    if (!canvas) canvas = node.querySelector('canvas');

    initPromise = (async () => {
      await initThree();
      window.addEventListener('resize', handleResize);
    })();

    return {
      destroy() {
        if (animFrameId) cancelAnimationFrame(animFrameId);
        window.removeEventListener('resize', handleResize);
        if (renderer) { renderer.dispose(); renderer.forceContextLoss(); }
        if (controls) controls.dispose();
        if (lmSphereGeo) { lmSphereGeo.dispose(); lmSphereGeo = null; }
        if (lmMaterials) { lmMaterials.high.dispose(); lmMaterials.mid.dispose(); lmMaterials.low.dispose(); lmMaterials = null; }
        if (lmRingMat) { lmRingMat.dispose(); lmRingMat = null; }
      }
    };
  }
</script>

<div class="viewer-container" bind:this={container} use:viewerSetup role="img" aria-label="3D mesh viewer — {meshData ? stats.vertices.toLocaleString() + ' vertices, ' + stats.regions + ' regions' : 'No mesh loaded'}">
  <canvas bind:this={canvas} on:click={handleCanvasClick} aria-hidden="true"></canvas>

  <!-- Label overlay -->
  {#if showLabels && showLandmarks && labelPositions.length > 0}
    <div class="label-overlay" bind:this={labelOverlay}>
      {#each labelPositions as lbl}
        <div class="label-tag"
          style="left: {lbl.x + 8}px; top: {lbl.y - 6}px;"
          class:high={lbl.confidence >= 0.8}
          class:med={lbl.confidence >= 0.5 && lbl.confidence < 0.8}
          class:low={lbl.confidence < 0.5}>
          {lbl.type.replace(/_/g, ' ')}
        </div>
      {/each}
    </div>
  {/if}

  {#await initPromise}
    <div class="viewer-overlay">
      <div class="sov-spinner"></div>
      <span>Initializing 3D engine...</span>
    </div>
  {:catch}
    <div class="viewer-overlay error"><span>⚠ {loadError}</span></div>
  {/await}

  {#if !meshData && loaded}
    <div class="viewer-overlay">
      <span class="viewer-hint">No mesh data loaded</span>
      <span class="viewer-hint-sub">Select a case with twin geometry</span>
    </div>
  {/if}

  <!-- HUD -->
  {#if meshData && loaded}
    <div class="viewer-hud">
      <span class="hud-item"><span class="hud-val">{stats.vertices.toLocaleString()}</span> verts</span>
      <span class="hud-sep">|</span>
      <span class="hud-item"><span class="hud-val">{stats.triangles.toLocaleString()}</span> tris</span>
      <span class="hud-sep">|</span>
      <span class="hud-item"><span class="hud-val">{stats.regions}</span> regions</span>
      {#if showLandmarks && stats.landmarks > 0}
        <span class="hud-sep">|</span>
        <span class="hud-item"><span class="hud-val">{stats.landmarks}</span> lm</span>
      {/if}
      {#if tissueOpacity < 1.0}
        <span class="hud-sep">|</span>
        <span class="hud-item">α <span class="hud-val">{(tissueOpacity * 100).toFixed(0)}%</span></span>
      {/if}
      {#if clipEnabled}
        <span class="hud-sep">|</span>
        <span class="hud-item">✂ {clipConfig.axis.toUpperCase()}={clipConfig.position.toFixed(1)}</span>
      {/if}
    </div>

    <!-- Measurement display -->
    {#if measureDistance != null}
      <div class="measure-display">
        <span class="measure-icon">📏</span>
        <span class="measure-val">{measureDistance.toFixed(2)}</span>
        <span class="measure-unit">mm</span>
        <button class="measure-clear" on:click={clearMeasurement}>✕</button>
      </div>
    {/if}

    <!-- Mode indicator -->
    {#if interactionMode === 'measure'}
      <div class="mode-indicator">
        <span class="mode-dot"></span>
        MEASURE MODE — Click two points on mesh
      </div>
    {/if}

    <div class="viewer-controls-hint">
      LMB {interactionMode === 'orbit' ? 'rotate' : 'measure'} · Scroll zoom · MMB pan · R reset
    </div>
  {/if}
</div>

<svelte:window on:keydown={(e) => {
  const tag = e.target?.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.target?.isContentEditable) return;
  if (e.key === 'r' || e.key === 'R') resetView();
  if (e.key === 'Escape') { clearMeasurement(); interactionMode = 'orbit'; }
}} />

<style>
  .viewer-container {
    position: relative;
    width: 100%;
    height: 100%;
    min-height: 400px;
    background: #08080A;
    border-radius: var(--radius-lg, 8px);
    overflow: hidden;
  }

  canvas { display: block; width: 100%; height: 100%; }

  .viewer-overlay {
    position: absolute; inset: 0;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 8px; background: rgba(8,8,10,0.85); color: #9CA3AF; font-size: 13px; z-index: 10;
  }
  .viewer-overlay.error { color: #EF4444; }
  .viewer-hint { font-size: 14px; color: #6B7280; }
  .viewer-hint-sub { font-size: 12px; color: #4B5563; }

  .viewer-hud {
    position: absolute; top: 12px; left: 12px;
    display: flex; align-items: center; gap: 6px;
    padding: 4px 10px; background: rgba(13,15,20,0.85);
    border: 1px solid #1F2937; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #6B7280;
    z-index: 5; backdrop-filter: blur(8px);
  }
  .hud-val { color: #E8E8EC; font-weight: 500; }
  .hud-sep { color: #1F2937; }

  .viewer-controls-hint {
    position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%);
    padding: 3px 10px; background: rgba(13,15,20,0.7); border-radius: 3px;
    font-size: 10px; color: #4B5563; z-index: 5; white-space: nowrap;
  }

  /* Labels */
  .label-overlay {
    position: absolute; inset: 0; pointer-events: none; z-index: 4; overflow: hidden;
  }
  .label-tag {
    position: absolute; padding: 1px 5px;
    font-size: 9px; font-family: 'JetBrains Mono', monospace;
    border-radius: 2px; white-space: nowrap;
    background: rgba(13,15,20,0.8); color: #9CA3AF;
    border-left: 2px solid #6B7280;
    pointer-events: none;
  }
  .label-tag.high { border-left-color: #10B981; color: #A7F3D0; }
  .label-tag.med  { border-left-color: #F59E0B; color: #FDE68A; }
  .label-tag.low  { border-left-color: #EF4444; color: #FCA5A5; }

  /* Measurement */
  .measure-display {
    position: absolute; top: 12px; right: 12px;
    display: flex; align-items: center; gap: 6px;
    padding: 6px 10px; background: rgba(13,15,20,0.9);
    border: 1px solid #F59E0B40; border-radius: 4px;
    z-index: 5;
  }
  .measure-icon { font-size: 14px; }
  .measure-val {
    font-family: 'JetBrains Mono', monospace; font-size: 16px;
    font-weight: 600; color: #F59E0B;
  }
  .measure-unit { font-size: 11px; color: #6B7280; }
  .measure-clear {
    background: none; border: none; color: #6B7280; cursor: pointer;
    font-size: 12px; padding: 2px; margin-left: 4px;
  }
  .measure-clear:hover { color: #EF4444; }

  /* Mode indicator */
  .mode-indicator {
    position: absolute; top: 44px; left: 12px;
    display: flex; align-items: center; gap: 6px;
    padding: 3px 8px; background: rgba(245,158,11,0.12);
    border: 1px solid #F59E0B30; border-radius: 3px;
    font-size: 9px; font-family: 'JetBrains Mono', monospace;
    color: #F59E0B; letter-spacing: 0.04em; z-index: 5;
  }
  .mode-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: #F59E0B; animation: mode-blink 1s infinite;
  }
  @keyframes mode-blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }
</style>
