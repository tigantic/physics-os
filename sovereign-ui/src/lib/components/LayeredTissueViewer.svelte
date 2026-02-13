<script>
  /**
   * LayeredTissueViewer
   *
   * Renders anatomical layers as separate THREE.Mesh objects:
   *   - bone:      opaque white/cream, high specular
   *   - cartilage: semi-translucent blue-white, medium specular
   *   - muscle:    semi-translucent red, soft shading (SMAS, nasalis, etc.)
   *   - skin:      translucent peach, subsurface-scatter approximation
   *   - fascia:    thin translucent yellow layer
   *
   * Each layer independently: visible, opacity, wireframe, selectable.
   * Scalar fields (stress, displacement) overlay on any layer.
   * Streamlines render as tube geometry in the scene.
   * Incision paths render as line geometry with cutting plane preview.
   *
   * Tree-shaking safe: uses Svelte action instead of onMount/onDestroy,
   * callback props instead of createEventDispatcher.
   */

  // ── Props ──────────────────────────────────────────────────
  /** @type {Record<string, LayerMeshData> | null} keyed by layer name */
  export let layerData = null;

  /** @type {import('$lib/api-client').LandmarksResponse | null} */
  export let landmarkData = null;

  /** @type {Record<string, LayerConfig>} per-layer visibility + opacity */
  export let layerConfig = {};

  /** @type {ScalarFieldData | null} */
  export let scalarField = null;

  /** @type {StreamlineData | null} */
  export let streamlines = null;

  /** @type {IncisionPath[] | null} */
  export let incisionPaths = null;

  /** @type {{ axis: 'x'|'y'|'z', position: number, enabled: boolean }} */
  export let clipConfig = { axis: 'y', position: 0, enabled: false };

  /** @type {'orbit' | 'measure' | 'incision' | 'graft'} */
  export let interactionMode = 'orbit';

  /** @type {boolean} */
  export let showLandmarks = true;

  /** @type {boolean} */
  export let showLabels = true;

  /** @type {string | null} */
  export let highlightLayer = null;

  // ── Callback props (replaces createEventDispatcher) ────────
  /** @type {(stats: object) => void} */
  export let onLayersLoaded = () => {};
  /** @type {(detail: { distance: number, points: number[][] }) => void} */
  export let onMeasurement = () => {};
  /** @type {(detail: { position: number[], normal: number[], layer: string }) => void} */
  export let onGraftPlace = () => {};
  /** @type {(detail: { point: number[], pathLength: number }) => void} */
  export let onIncisionPoint = () => {};
  /** @type {(detail: { points: number[][] }) => void} */
  export let onIncisionCommit = () => {};
  /** @type {(mode: string) => void} */
  export let onModeChange = () => {};

  // ── Internal ───────────────────────────────────────────────
  let container;
  let canvas;
  let THREE;
  let OrbitControls;
  let RoomEnvironment;
  let renderer, scene, camera, controls;
  let envMap = null;
  let layerGroup, landmarkGroup, streamlineGroup, incisionGroup, measureGroup;
  let clipPlane = null;
  let animFrameId;
  let loaded = false;
  let loadError = '';
  let raycaster, mouse;

  // Layer meshes keyed by name
  let layerMeshes = {};

  // Measurement
  let measurePoints = [];
  let measureDistance = null;

  // Incision drawing
  let incisionDrawing = [];

  // Label positions
  let labelPositions = [];

  // Stats
  let stats = { totalVertices: 0, totalTriangles: 0, layers: 0, landmarks: 0 };

  // ── Layer material presets (PBR — MeshPhysicalMaterial) ────
  const LAYER_PRESETS = {
    bone: {
      color: '#F2EBD9', roughness: 0.65, metalness: 0.0,
      clearcoat: 0.15, clearcoatRoughness: 0.6,
      defaultOpacity: 1.0, emissive: '#000000',
      ior: 1.55, thickness: 0,
    },
    cartilage: {
      color: '#B8D4E3', roughness: 0.35, metalness: 0.0,
      clearcoat: 0.5, clearcoatRoughness: 0.25,
      defaultOpacity: 0.85, emissive: '#060A14',
      ior: 1.38, thickness: 0,
    },
    muscle: {
      color: '#C03030', roughness: 0.55, metalness: 0.0,
      clearcoat: 0.2, clearcoatRoughness: 0.5,
      defaultOpacity: 0.7, emissive: '#0A0202',
      ior: 1.4, thickness: 0,
    },
    skin: {
      color: '#E8B89D', roughness: 0.45, metalness: 0.0,
      clearcoat: 0.3, clearcoatRoughness: 0.35,
      defaultOpacity: 0.5, emissive: '#0F0805',
      // Subsurface scattering approximation via transmission + thickness
      ior: 1.4, thickness: 2.5,
      transmission: 0.08,
      sheen: 0.3, sheenRoughness: 0.4, sheenColor: '#FFCCAA',
    },
    fascia: {
      color: '#E8D888', roughness: 0.6, metalness: 0.0,
      clearcoat: 0.1, clearcoatRoughness: 0.7,
      defaultOpacity: 0.35, emissive: '#060602',
      ior: 1.35, thickness: 0,
    },
    soft_tissue: {
      color: '#D4888C', roughness: 0.5, metalness: 0.0,
      clearcoat: 0.2, clearcoatRoughness: 0.4,
      defaultOpacity: 0.6, emissive: '#080303',
      ior: 1.38, thickness: 0,
    },
  };

  function getPreset(layerName) {
    const key = layerName.toLowerCase();
    for (const [k, v] of Object.entries(LAYER_PRESETS)) {
      if (key.includes(k)) return v;
    }
    return LAYER_PRESETS.soft_tissue;
  }

  // ── Svelte action: replaces onMount/onDestroy ──────────────
  function viewerSetup(node) {
    container = node;
    canvas = node.querySelector('canvas');
    initThree();
    window.addEventListener('resize', handleResize);
    return {
      destroy() {
        if (animFrameId) cancelAnimationFrame(animFrameId);
        window.removeEventListener('resize', handleResize);
        renderer?.dispose();
        renderer?.forceContextLoss();
        controls?.dispose();
      }
    };
  }

  // ── Three.js Init ──────────────────────────────────────────
  async function initThree() {
    try {
      THREE = await import('three');
      const { OrbitControls: OC } = await import('three/examples/jsm/controls/OrbitControls.js');
      OrbitControls = OC;
      try {
        const { RoomEnvironment: RE } = await import('three/examples/jsm/environments/RoomEnvironment.js');
        RoomEnvironment = RE;
      } catch { RoomEnvironment = null; }
    } catch (err) {
      if (window.THREE) THREE = window.THREE;
      else { loadError = 'Three.js unavailable'; return; }
    }

    scene = new THREE.Scene();
    // Subtle dark gradient background
    scene.background = new THREE.Color('#06060A');

    camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 0.1, 2000);
    camera.position.set(0, 0, 250);

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'high-performance' });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    renderer.localClippingEnabled = true;
    renderer.sortObjects = true;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.VSMShadowMap;

    // PBR environment map via PMREMGenerator
    if (RoomEnvironment) {
      try {
        const pmremGenerator = new THREE.PMREMGenerator(renderer);
        pmremGenerator.compileEquirectangularShader();
        const roomScene = new RoomEnvironment(renderer);
        envMap = pmremGenerator.fromScene(roomScene, 0.04).texture;
        scene.environment = envMap;
        pmremGenerator.dispose();
      } catch { /* Environment map generation failed — fall back to lights only */ }
    }

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.rotateSpeed = 0.45;
    controls.zoomSpeed = 0.8;
    controls.minDistance = 30;
    controls.maxDistance = 600;

    // Clinical PBR lighting rig (5-light setup)
    // Hemisphere light for broad ambient fill
    const hemi = new THREE.HemisphereLight('#E0E8FF', '#1A1A2E', 0.25);
    scene.add(hemi);

    // Key light (warm white, upper-right)
    const key = new THREE.DirectionalLight('#FFF4E6', 0.9);
    key.position.set(80, 130, 150);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 10;
    key.shadow.camera.far = 500;
    key.shadow.camera.left = -100;
    key.shadow.camera.right = 100;
    key.shadow.camera.top = 100;
    key.shadow.camera.bottom = -100;
    key.shadow.radius = 4;
    key.shadow.blurSamples = 8;
    scene.add(key);

    // Fill light (cool, left side)
    const fill = new THREE.DirectionalLight('#8098CC', 0.35);
    fill.position.set(-90, -30, -80);
    scene.add(fill);

    // Rim light (blue backlight for edge definition)
    const rim = new THREE.DirectionalLight('#4466AA', 0.4);
    rim.position.set(0, 100, -180);
    scene.add(rim);

    // Bottom fill (warm, prevents black shadows under chin)
    const bottom = new THREE.DirectionalLight('#553322', 0.15);
    bottom.position.set(0, -80, 60);
    scene.add(bottom);

    // Subtle point light at nose tip area for nasal highlight
    const accent = new THREE.PointLight('#FFFFFF', 0.15, 200, 2);
    accent.position.set(0, 5, 100);
    scene.add(accent);

    // Groups
    layerGroup = new THREE.Group();
    landmarkGroup = new THREE.Group();
    streamlineGroup = new THREE.Group();
    incisionGroup = new THREE.Group();
    measureGroup = new THREE.Group();
    scene.add(layerGroup, landmarkGroup, streamlineGroup, incisionGroup, measureGroup);

    // Subtle ground plane for spatial grounding
    const groundGeo = new THREE.PlaneGeometry(600, 600);
    const groundMat = new THREE.MeshPhysicalMaterial({
      color: '#0A0A0F',
      roughness: 0.9,
      metalness: 0.0,
      transparent: true,
      opacity: 0.4,
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -80;
    ground.receiveShadow = true;
    ground.name = '_ground';
    scene.add(ground);

    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    loaded = true;
    animate();
  }

  // ── Build Layers ───────────────────────────────────────────
  function buildLayers() {
    if (!THREE || !layerGroup || !layerData) return;

    // Dispose existing
    Object.values(layerMeshes).forEach(m => {
      m.geometry?.dispose();
      if (Array.isArray(m.material)) m.material.forEach(mt => mt.dispose());
      else m.material?.dispose();
    });
    while (layerGroup.children.length > 0) layerGroup.remove(layerGroup.children[0]);
    layerMeshes = {};

    let totalVerts = 0, totalTris = 0;

    const clippingPlanes = [];
    if (clipConfig.enabled && clipPlane) clippingPlanes.push(clipPlane);

    for (const [name, data] of Object.entries(layerData)) {
      if (!data.positions || data.positions.length === 0) continue;

      const preset = getPreset(name);
      const config = layerConfig[name] || { visible: true, opacity: preset.defaultOpacity, wireframe: false };

      // Build geometry
      const posArray = new Float32Array(data.positions.length * 3);
      for (let i = 0; i < data.positions.length; i++) {
        posArray[i * 3] = data.positions[i][0];
        posArray[i * 3 + 1] = data.positions[i][1];
        posArray[i * 3 + 2] = data.positions[i][2];
      }

      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));

      if (data.indices && data.indices.length > 0) {
        const idxArray = new Uint32Array(data.indices.length * 3);
        for (let i = 0; i < data.indices.length; i++) {
          idxArray[i * 3] = data.indices[i][0];
          idxArray[i * 3 + 1] = data.indices[i][1];
          idxArray[i * 3 + 2] = data.indices[i][2];
        }
        geo.setIndex(new THREE.BufferAttribute(idxArray, 1));
      }

      // Apply scalar field colors if present and targeting this layer
      if (scalarField && (!scalarField.layer || scalarField.layer === name)) {
        const colorArray = new Float32Array(data.positions.length * 3);
        const range = scalarField.max - scalarField.min || 1;
        for (let i = 0; i < data.positions.length; i++) {
          const val = scalarField.values?.[i] ?? 0;
          const t = Math.max(0, Math.min(1, (val - scalarField.min) / range));
          const rgb = scalarToRGB(t, scalarField.colormap || 'jet');
          colorArray[i * 3] = rgb[0];
          colorArray[i * 3 + 1] = rgb[1];
          colorArray[i * 3 + 2] = rgb[2];
        }
        geo.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));
      }

      geo.computeVertexNormals();
      geo.computeBoundingSphere();

      const hasVertexColors = geo.attributes.color != null;
      const isHighlighted = highlightLayer === name;

      const mat = new THREE.MeshPhysicalMaterial({
        color: hasVertexColors ? '#FFFFFF' : new THREE.Color(preset.color),
        vertexColors: hasVertexColors,
        roughness: preset.roughness ?? 0.5,
        metalness: preset.metalness ?? 0.0,
        clearcoat: preset.clearcoat ?? 0.0,
        clearcoatRoughness: preset.clearcoatRoughness ?? 0.5,
        emissive: new THREE.Color(preset.emissive || '#000000'),
        emissiveIntensity: isHighlighted ? 0.15 : 0.0,
        ior: preset.ior ?? 1.5,
        thickness: preset.thickness ?? 0,
        transmission: preset.transmission ?? 0,
        sheen: preset.sheen ?? 0,
        sheenRoughness: preset.sheenRoughness ?? 0,
        sheenColor: preset.sheenColor ? new THREE.Color(preset.sheenColor) : new THREE.Color('#000000'),
        side: THREE.DoubleSide,
        transparent: true,
        opacity: config.opacity,
        depthWrite: config.opacity >= 0.9,
        clippingPlanes,
        envMap: envMap,
        envMapIntensity: 0.4,
      });

      if (isHighlighted) {
        mat.emissive = new THREE.Color('#1A2A4A');
        mat.emissiveIntensity = 0.2;
      }

      const mesh = new THREE.Mesh(geo, mat);
      mesh.name = name;
      mesh.visible = config.visible;
      mesh.renderOrder = layerRenderOrder(name);
      mesh.castShadow = true;
      mesh.receiveShadow = true;

      layerGroup.add(mesh);
      layerMeshes[name] = mesh;

      // Optional wireframe overlay
      if (config.wireframe) {
        const wireMat = new THREE.MeshBasicMaterial({
          color: '#2A3040',
          wireframe: true,
          transparent: true,
          opacity: 0.15,
          clippingPlanes,
        });
        const wire = new THREE.Mesh(geo.clone(), wireMat);
        wire.name = `${name}_wire`;
        wire.visible = config.visible;
        wire.renderOrder = mesh.renderOrder + 0.1;
        layerGroup.add(wire);
      }

      totalVerts += data.positions.length;
      totalTris += (data.indices?.length ?? 0);
    }

    stats = {
      totalVertices: totalVerts,
      totalTriangles: totalTris,
      layers: Object.keys(layerData).length,
      landmarks: landmarkData?.landmarks?.length ?? 0,
    };

    // Auto-center camera on first build
    if (!layerGroup.userData.centered) {
      centerCamera();
      layerGroup.userData.centered = true;
    }

    onLayersLoaded(stats);
  }

  function layerRenderOrder(name) {
    const order = { bone: 0, cartilage: 1, muscle: 2, fascia: 3, soft_tissue: 4, skin: 5 };
    for (const [k, v] of Object.entries(order)) {
      if (name.toLowerCase().includes(k)) return v;
    }
    return 3;
  }

  // ── Scalar field color mapping ─────────────────────────────
  function scalarToRGB(t, cmap) {
    const maps = {
      jet: [[0,'#00007F'],[0.15,'#0000FF'],[0.3,'#00BFFF'],[0.5,'#40FF00'],[0.75,'#FFFF00'],[1,'#FF0000']],
      stress: [[0,'#0000FF'],[0.2,'#00AAFF'],[0.4,'#00FF55'],[0.6,'#FFFF00'],[0.8,'#FF8800'],[1,'#FF0000']],
      viridis: [[0,'#440154'],[0.25,'#3B528B'],[0.5,'#21918C'],[0.75,'#5EC962'],[1,'#FDE725']],
      coolwarm: [[0,'#3B4CC0'],[0.5,'#F7F7F7'],[1,'#B40426']],
    };
    const stops = maps[cmap] || maps.jet;
    for (let i = 0; i < stops.length - 1; i++) {
      if (t >= stops[i][0] && t <= stops[i + 1][0]) {
        const lt = (t - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
        return lerpRGB(stops[i][1], stops[i + 1][1], lt);
      }
    }
    return hexToRGB(stops[stops.length - 1][1]);
  }

  function hexToRGB(hex) {
    return [parseInt(hex.slice(1,3),16)/255, parseInt(hex.slice(3,5),16)/255, parseInt(hex.slice(5,7),16)/255];
  }

  function lerpRGB(a, b, t) {
    const ar = hexToRGB(a), br = hexToRGB(b);
    return [ar[0]+(br[0]-ar[0])*t, ar[1]+(br[1]-ar[1])*t, ar[2]+(br[2]-ar[2])*t];
  }

  // ── Streamlines ────────────────────────────────────────────
  function buildStreamlines() {
    if (!THREE || !streamlineGroup) return;
    while (streamlineGroup.children.length > 0) {
      const c = streamlineGroup.children[0];
      c.geometry?.dispose(); c.material?.dispose();
      streamlineGroup.remove(c);
    }
    if (!streamlines?.lines) return;

    for (const line of streamlines.lines) {
      if (!line.points || line.points.length < 2) continue;

      const points = line.points.map(p => new THREE.Vector3(p[0], p[1], p[2]));
      const curve = new THREE.CatmullRomCurve3(points);

      const tubeGeo = new THREE.TubeGeometry(curve, Math.max(points.length * 4, 20), line.radius || 0.3, 6, false);

      const colors = new Float32Array(tubeGeo.attributes.position.count * 3);
      const vMin = streamlines.velocity_min ?? 0;
      const vMax = streamlines.velocity_max ?? 1;
      const range = vMax - vMin || 1;

      for (let i = 0; i < tubeGeo.attributes.position.count; i++) {
        const paramT = i / tubeGeo.attributes.position.count;
        const velIdx = Math.floor(paramT * (line.velocities?.length - 1 || 0));
        const vel = line.velocities?.[velIdx] ?? vMin;
        const t = Math.max(0, Math.min(1, (vel - vMin) / range));
        const rgb = scalarToRGB(t, 'jet');
        colors[i * 3] = rgb[0];
        colors[i * 3 + 1] = rgb[1];
        colors[i * 3 + 2] = rgb[2];
      }
      tubeGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

      const tubeMat = new THREE.MeshBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: streamlines.opacity ?? 0.7,
        side: THREE.DoubleSide,
      });

      const tube = new THREE.Mesh(tubeGeo, tubeMat);
      tube.renderOrder = 10;
      streamlineGroup.add(tube);
    }
  }

  // ── Incision Paths ─────────────────────────────────────────
  function buildIncisions() {
    if (!THREE || !incisionGroup) return;
    while (incisionGroup.children.length > 0) {
      const c = incisionGroup.children[0];
      c.geometry?.dispose(); c.material?.dispose();
      incisionGroup.remove(c);
    }
    if (!incisionPaths) return;

    for (const path of incisionPaths) {
      if (!path.points || path.points.length < 2) continue;

      const points = path.points.map(p => new THREE.Vector3(p[0], p[1], p[2]));

      const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
      const lineMat = new THREE.LineBasicMaterial({
        color: path.color || '#EF4444',
        linewidth: 2,
      });
      incisionGroup.add(new THREE.Line(lineGeo, lineMat));

      if (path.depth && points.length >= 2) {
        for (let i = 0; i < points.length - 1; i++) {
          const a = points[i];
          const b = points[i + 1];
          const dir = new THREE.Vector3().subVectors(b, a).normalize();
          const down = new THREE.Vector3(0, -1, 0);
          const normal = new THREE.Vector3().crossVectors(dir, down).normalize();

          const ribbonGeo = new THREE.PlaneGeometry(a.distanceTo(b), path.depth);
          ribbonGeo.lookAt(normal);
          ribbonGeo.translate(
            (a.x + b.x) / 2,
            (a.y + b.y) / 2 - path.depth / 2,
            (a.z + b.z) / 2
          );

          const ribbonMat = new THREE.MeshBasicMaterial({
            color: '#EF444440',
            transparent: true,
            opacity: 0.2,
            side: THREE.DoubleSide,
            depthWrite: false,
          });
          incisionGroup.add(new THREE.Mesh(ribbonGeo, ribbonMat));
        }
      }
    }
  }

  // ── Landmarks ──────────────────────────────────────────────
  function buildLandmarks() {
    if (!THREE || !landmarkGroup || !landmarkData?.landmarks) return;
    while (landmarkGroup.children.length > 0) {
      const c = landmarkGroup.children[0];
      c.geometry?.dispose(); c.material?.dispose();
      landmarkGroup.remove(c);
    }
    if (!showLandmarks) return;

    const sGeo = new THREE.SphereGeometry(1.0, 12, 12);
    for (const lm of landmarkData.landmarks) {
      if (!lm.position || lm.position.length < 3) continue;
      const color = lm.confidence >= 0.8 ? '#10B981' : lm.confidence >= 0.5 ? '#F59E0B' : '#EF4444';
      const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9 });
      const sphere = new THREE.Mesh(sGeo, mat);
      sphere.position.set(lm.position[0], lm.position[1], lm.position[2]);
      sphere.name = lm.type;
      sphere.userData = lm;
      landmarkGroup.add(sphere);
    }
  }

  // ── Clip Plane ─────────────────────────────────────────────
  function updateClipPlane() {
    if (!THREE) return;
    const normals = { x: [1,0,0], y: [0,1,0], z: [0,0,1] };
    const n = normals[clipConfig.axis] || normals.y;
    clipPlane = new THREE.Plane(new THREE.Vector3(n[0], n[1], n[2]), -clipConfig.position);
  }

  // ── Interaction ────────────────────────────────────────────
  function handleCanvasClick(e) {
    if (!raycaster || !camera || !canvas) return;
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);

    const meshes = Object.values(layerMeshes).filter(m => m.visible);
    const intersects = raycaster.intersectObjects(meshes);

    if (intersects.length === 0) return;
    const hit = intersects[0];

    if (interactionMode === 'measure') {
      handleMeasureClick(hit.point.clone());
    } else if (interactionMode === 'incision') {
      handleIncisionClick(hit.point.clone(), hit.face?.normal?.clone());
    } else if (interactionMode === 'graft') {
      onGraftPlace({
        position: [hit.point.x, hit.point.y, hit.point.z],
        normal: hit.face?.normal ? [hit.face.normal.x, hit.face.normal.y, hit.face.normal.z] : [0,1,0],
        layer: hit.object.name,
      });
    }
  }

  function handleMeasureClick(point) {
    measurePoints.push(point);
    const dotGeo = new THREE.SphereGeometry(0.6, 8, 8);
    const dotMat = new THREE.MeshBasicMaterial({ color: '#F59E0B' });
    const dot = new THREE.Mesh(dotGeo, dotMat);
    dot.position.copy(point);
    measureGroup.add(dot);

    if (measurePoints.length === 2) {
      const lineGeo = new THREE.BufferGeometry().setFromPoints(measurePoints);
      const lineMat = new THREE.LineBasicMaterial({ color: '#F59E0B' });
      measureGroup.add(new THREE.Line(lineGeo, lineMat));
      measureDistance = measurePoints[0].distanceTo(measurePoints[1]);
      onMeasurement({ distance: measureDistance, points: measurePoints.map(p => [p.x, p.y, p.z]) });
    } else if (measurePoints.length > 2) {
      clearMeasurement();
    }
  }

  function handleIncisionClick(point, normal) {
    incisionDrawing.push([point.x, point.y, point.z]);
    if (incisionDrawing.length >= 2) {
      const pts = incisionDrawing.map(p => new THREE.Vector3(p[0], p[1], p[2]));
      const old = incisionGroup.getObjectByName('_incision_preview');
      if (old) { old.geometry?.dispose(); old.material?.dispose(); incisionGroup.remove(old); }

      const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
      const lineMat = new THREE.LineBasicMaterial({ color: '#EF4444', linewidth: 2 });
      const line = new THREE.Line(lineGeo, lineMat);
      line.name = '_incision_preview';
      incisionGroup.add(line);
    }
    onIncisionPoint({ point: [point.x, point.y, point.z], pathLength: incisionDrawing.length });
  }

  export function clearMeasurement() {
    measurePoints = [];
    measureDistance = null;
    while (measureGroup.children.length > 0) {
      const c = measureGroup.children[0]; c.geometry?.dispose(); c.material?.dispose(); measureGroup.remove(c);
    }
  }

  export function commitIncision() {
    if (incisionDrawing.length < 2) return;
    onIncisionCommit({ points: [...incisionDrawing] });
    incisionDrawing = [];
  }

  export function clearIncision() {
    incisionDrawing = [];
    const old = incisionGroup?.getObjectByName('_incision_preview');
    if (old) { old.geometry?.dispose(); old.material?.dispose(); incisionGroup.remove(old); }
  }

  export function centerCamera() {
    if (!camera || !controls) return;
    const box = new THREE.Box3();
    layerGroup.children.forEach(m => {
      if (m.geometry?.boundingSphere) box.expandByObject(m);
    });
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    camera.position.set(center.x, center.y, center.z + size * 0.8);
    controls.target.copy(center);
    controls.update();
  }

  export function resetView() {
    if (layerGroup) layerGroup.userData.centered = false;
    centerCamera();
  }

  /** Move camera to a named anatomical preset */
  export function setCameraPreset(preset) {
    if (!camera || !controls || !THREE) return;
    const box = new THREE.Box3();
    layerGroup.children.forEach(m => {
      if (m.geometry?.boundingSphere) box.expandByObject(m);
    });
    const center = box.isEmpty() ? new THREE.Vector3() : box.getCenter(new THREE.Vector3());
    const size = box.isEmpty() ? 200 : box.getSize(new THREE.Vector3()).length();
    const dist = size * 0.8;

    const presets = {
      anterior:      { pos: [center.x, center.y, center.z + dist],  up: [0, 1, 0] },
      posterior:     { pos: [center.x, center.y, center.z - dist],  up: [0, 1, 0] },
      'lateral-right': { pos: [center.x + dist, center.y, center.z], up: [0, 1, 0] },
      'lateral-left':  { pos: [center.x - dist, center.y, center.z], up: [0, 1, 0] },
      superior:      { pos: [center.x, center.y + dist, center.z],  up: [0, 0, -1] },
      inferior:      { pos: [center.x, center.y - dist, center.z],  up: [0, 0, 1] },
      'oblique-right': { pos: [center.x + dist * 0.7, center.y + dist * 0.3, center.z + dist * 0.7], up: [0, 1, 0] },
      'oblique-left':  { pos: [center.x - dist * 0.7, center.y + dist * 0.3, center.z + dist * 0.7], up: [0, 1, 0] },
    };

    const p = presets[preset];
    if (!p) return;

    camera.position.set(p.pos[0], p.pos[1], p.pos[2]);
    camera.up.set(p.up[0], p.up[1], p.up[2]);
    controls.target.copy(center);
    controls.update();
  }

  // ── Label Projection ───────────────────────────────────────
  function updateLabels() {
    if (!showLabels || !landmarkData?.landmarks || !camera || !container) {
      labelPositions = [];
      return;
    }
    const w = container.clientWidth, h = container.clientHeight;
    const newPos = [];
    for (const lm of landmarkData.landmarks) {
      if (!lm.position) continue;
      const vec = new THREE.Vector3(lm.position[0], lm.position[1], lm.position[2]).project(camera);
      const x = (vec.x * 0.5 + 0.5) * w;
      const y = (-vec.y * 0.5 + 0.5) * h;
      if (vec.z < 1 && x > -20 && x < w + 20 && y > -20 && y < h + 20) {
        newPos.push({ type: lm.type, confidence: lm.confidence, x, y });
      }
    }
    labelPositions = newPos;
  }

  // ── Animation ──────────────────────────────────────────────
  function animate() {
    animFrameId = requestAnimationFrame(animate);
    controls?.update();
    renderer?.render(scene, camera);
    if (showLabels) updateLabels();
  }

  function handleResize() {
    if (!container || !renderer || !camera) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }

  // ── Reactivity ─────────────────────────────────────────────
  $: if (loaded && layerData) buildLayers();
  $: if (loaded && landmarkData) buildLandmarks();
  $: if (loaded && streamlines) buildStreamlines();
  $: if (loaded && incisionPaths) buildIncisions();
  $: if (loaded) { updateClipPlane(); if (layerData) buildLayers(); }
  $: if (loaded && landmarkGroup) landmarkGroup.visible = showLandmarks;
  $: if (loaded && controls) controls.enabled = interactionMode === 'orbit';
  $: if (loaded && highlightLayer !== undefined && layerData) buildLayers();

  // Update individual layer visibility/opacity/wireframe without full rebuild
  $: if (loaded && layerConfig && THREE) {
    for (const [name, config] of Object.entries(layerConfig)) {
      const mesh = layerMeshes[name];
      if (mesh) {
        mesh.visible = config.visible;
        if (mesh.material) {
          mesh.material.opacity = config.opacity;
          mesh.material.transparent = config.opacity < 1.0;
          mesh.material.depthWrite = config.opacity >= 0.9;
          mesh.material.needsUpdate = true;
        }
      }
      // Wireframe: create on-demand if toggled on, hide/remove if toggled off
      let wire = layerGroup?.children.find(c => c.name === `${name}_wire`);
      if (config.wireframe && config.visible && mesh && !wire) {
        // Create wireframe overlay on demand
        const clippingPlanes = [];
        if (clipConfig.enabled && clipPlane) clippingPlanes.push(clipPlane);
        const wireMat = new THREE.MeshBasicMaterial({
          color: '#2A3040',
          wireframe: true,
          transparent: true,
          opacity: 0.15,
          clippingPlanes,
        });
        wire = new THREE.Mesh(mesh.geometry.clone(), wireMat);
        wire.name = `${name}_wire`;
        wire.visible = true;
        wire.renderOrder = mesh.renderOrder + 0.1;
        layerGroup.add(wire);
      } else if (wire) {
        wire.visible = config.visible && config.wireframe;
      }
    }
  }
</script>

<div class="ltv-container" use:viewerSetup>
  <canvas on:click={handleCanvasClick}></canvas>

  <!-- Landmark labels -->
  {#if showLabels && showLandmarks && labelPositions.length > 0}
    <div class="label-overlay">
      {#each labelPositions as lbl}
        <div class="label-tag" style="left: {lbl.x + 8}px; top: {lbl.y - 6}px;"
          class:high={lbl.confidence >= 0.8} class:med={lbl.confidence >= 0.5 && lbl.confidence < 0.8} class:low={lbl.confidence < 0.5}>
          {lbl.type.replace(/_/g, ' ')}
        </div>
      {/each}
    </div>
  {/if}

  {#if !loaded && !loadError}
    <div class="ltv-overlay"><div class="sov-spinner"></div><span>Initializing rendering engine...</span></div>
  {/if}

  {#if loadError}
    <div class="ltv-overlay error"><span>⚠ {loadError}</span></div>
  {/if}

  {#if !layerData && loaded}
    <div class="ltv-overlay"><span class="ltv-hint">No tissue layer data</span></div>
  {/if}

  <!-- HUD -->
  {#if layerData && loaded}
    <div class="ltv-hud">
      <span class="hud-i"><b>{stats.totalVertices.toLocaleString()}</b> verts</span>
      <span class="hud-s">|</span>
      <span class="hud-i"><b>{stats.totalTriangles.toLocaleString()}</b> tris</span>
      <span class="hud-s">|</span>
      <span class="hud-i"><b>{stats.layers}</b> layers</span>
      {#if stats.landmarks > 0}
        <span class="hud-s">|</span>
        <span class="hud-i"><b>{stats.landmarks}</b> lm</span>
      {/if}
      {#if streamlines?.lines?.length > 0}
        <span class="hud-s">|</span>
        <span class="hud-i"><b>{streamlines.lines.length}</b> streamlines</span>
      {/if}
    </div>

    {#if measureDistance != null}
      <div class="measure-badge">
        📏 <span class="measure-val">{measureDistance.toFixed(2)}</span> <span class="measure-unit">mm</span>
        <button class="measure-x" on:click={clearMeasurement}>✕</button>
      </div>
    {/if}

    {#if interactionMode !== 'orbit'}
      <div class="mode-badge" class:incision={interactionMode === 'incision'} class:graft={interactionMode === 'graft'}>
        <span class="mode-dot"></span>
        {interactionMode === 'measure' ? 'MEASURE' : interactionMode === 'incision' ? 'INCISION' : 'GRAFT PLACEMENT'}
        {#if interactionMode === 'incision' && incisionDrawing.length > 0}
          — {incisionDrawing.length} pts
        {/if}
      </div>
    {/if}
  {/if}
</div>

<svelte:window on:keydown={(e) => {
  if (e.key === 'r' || e.key === 'R') resetView();
  if (e.key === 'Escape') { clearMeasurement(); clearIncision(); interactionMode = 'orbit'; onModeChange('orbit'); }
  if (e.key === 'Enter' && interactionMode === 'incision') commitIncision();
}} />

<style>
  .ltv-container { position: relative; width: 100%; height: 100%; min-height: 500px; background: #08080A; border-radius: 8px; overflow: hidden; }
  canvas { display: block; width: 100%; height: 100%; cursor: crosshair; }
  .ltv-overlay {
    position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 8px; background: rgba(8,8,10,0.85); color: #9CA3AF; font-size: 13px; z-index: 10;
  }
  .ltv-overlay.error { color: #EF4444; }
  .ltv-hint { color: #6B7280; }

  .ltv-hud {
    position: absolute; top: 10px; left: 10px; display: flex; align-items: center; gap: 5px;
    padding: 3px 8px; background: rgba(8,10,15,0.85); border: 1px solid #1F2937; border-radius: 3px;
    font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #6B7280; z-index: 5; backdrop-filter: blur(6px);
  }
  .hud-i b { color: #E8E8EC; }
  .hud-s { color: #1F2937; }

  .label-overlay { position: absolute; inset: 0; pointer-events: none; z-index: 4; overflow: hidden; }
  .label-tag {
    position: absolute; padding: 1px 5px; font-size: 9px; font-family: 'JetBrains Mono', monospace;
    border-radius: 2px; white-space: nowrap; background: rgba(8,10,15,0.8); color: #9CA3AF;
    border-left: 2px solid #6B7280; pointer-events: none;
  }
  .label-tag.high { border-left-color: #10B981; color: #A7F3D0; }
  .label-tag.med  { border-left-color: #F59E0B; color: #FDE68A; }
  .label-tag.low  { border-left-color: #EF4444; color: #FCA5A5; }

  .measure-badge {
    position: absolute; top: 10px; right: 10px; display: flex; align-items: center; gap: 6px;
    padding: 5px 10px; background: rgba(8,10,15,0.9); border: 1px solid #F59E0B40; border-radius: 4px; z-index: 5;
  }
  .measure-val { font-family: 'JetBrains Mono', monospace; font-size: 15px; font-weight: 600; color: #F59E0B; }
  .measure-unit { font-size: 10px; color: #6B7280; }
  .measure-x { background: none; border: none; color: #6B7280; cursor: pointer; font-size: 11px; }

  .mode-badge {
    position: absolute; top: 38px; left: 10px; display: flex; align-items: center; gap: 5px;
    padding: 2px 7px; background: rgba(59,130,246,0.12); border: 1px solid #3B82F630; border-radius: 3px;
    font-size: 9px; font-family: 'JetBrains Mono', monospace; color: #3B82F6; letter-spacing: 0.04em; z-index: 5;
  }
  .mode-badge.incision { background: rgba(239,68,68,0.12); border-color: #EF444430; color: #EF4444; }
  .mode-badge.graft { background: rgba(16,185,129,0.12); border-color: #10B98130; color: #10B981; }
  .mode-dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; animation: blink 1s infinite; }
  @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
</style>
