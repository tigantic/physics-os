/**
 * HyperTensor Facial Plastics — G6 Three.js 3D Viewer
 *
 * Full scene manager: camera, lights, controls, mesh rendering,
 * landmark sprites, region coloring, wireframe/opacity/reset.
 * Also exposes renderPreview(canvas) for Twin Inspect compact view.
 *
 * Three.js loaded from CDN via importmap in index.html.
 * Falls back to Canvas2D wireframe if THREE is unavailable.
 */

"use strict";

const Viewer3D = (() => {
  let _el = null;
  let _scene = null;
  let _camera = null;
  let _renderer = null;
  let _controls = null;
  let _group = null;
  let _animId = null;
  let _canvas = null;
  let _active = false;

  // HUD state
  let _wireframe = false;
  let _opacity = 1.0;
  let _showLandmarks = true;
  let _showRegions = true;

  function init() {
    _el = document.getElementById("mode-viewer3d");
  }

  async function load() {
    if (!_el) return;
    const caseId = Store.get("selectedCase.id");
    if (!caseId) { render(); return; }

    Store.set("viewer3d.loading", true);
    render();

    try {
      const [meshData, landmarks] = await Promise.all([
        API.getMeshData(caseId),
        API.getLandmarks(caseId),
      ]);

      Store.set("viewer3d.mesh", meshData);
      Store.set("viewer3d.landmarks", landmarks.landmarks || landmarks);
    } catch (err) {
      Toast.error("3D data load failed: " + (err.message || err));
    } finally {
      Store.set("viewer3d.loading", false);
      _buildScene();
    }
  }

  function render() {
    if (!_el) return;
    const loading = Store.get("viewer3d.loading");
    const caseId = Store.get("selectedCase.id");

    if (!caseId) {
      _el.innerHTML = `<div class="panel-header"><h2>3D Viewer</h2></div><div class="placeholder">Select a case from the Case Library.</div>`;
      return;
    }

    _el.innerHTML = `
      <div class="viewer3d-wrap">
        <canvas id="viewer3d-canvas"></canvas>
        ${loading ? '<div class="placeholder" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;z-index:2">Loading 3D data...</div>' : ""}
        <div class="viewer-hud viewer-hud-tl" id="viewer-hud-controls">
          <div class="hud-row">
            <label style="font-size:11px;color:var(--text-muted)"><input type="checkbox" id="hud-wireframe" ${_wireframe ? "checked" : ""}> Wireframe</label>
            <label style="font-size:11px;color:var(--text-muted)"><input type="checkbox" id="hud-landmarks" ${_showLandmarks ? "checked" : ""}> Landmarks</label>
            <label style="font-size:11px;color:var(--text-muted)"><input type="checkbox" id="hud-regions" ${_showRegions ? "checked" : ""}> Regions</label>
          </div>
          <div class="hud-row">
            <label style="font-size:11px;color:var(--text-muted)">Opacity</label>
            <input type="range" min="0.1" max="1" step="0.05" value="${_opacity}" id="hud-opacity" style="width:100px">
            <button class="btn btn-ghost btn-sm" id="hud-reset">Reset</button>
          </div>
        </div>
        <div class="viewer-info" id="viewer-info"></div>
      </div>
    `;

    _canvas = document.getElementById("viewer3d-canvas");
    if (!loading) _buildScene();
    _bindHUD();
  }

  function activate() {
    _active = true;
    if (_canvas && !_animId) _animate();
    _resize();
  }

  function deactivate() {
    _active = false;
    if (_animId) { cancelAnimationFrame(_animId); _animId = null; }
  }

  /* ── Scene construction ────────────────────────────────── */

  function _buildScene() {
    if (!_canvas) return;
    if (typeof THREE === "undefined") { _fallbackCanvas2D(); return; }

    const rect = _canvas.parentElement.getBoundingClientRect();
    const w = rect.width || 800;
    const h = rect.height || 600;

    // Renderer
    if (_renderer) _renderer.dispose();
    _renderer = new THREE.WebGLRenderer({ canvas: _canvas, antialias: true, alpha: true });
    _renderer.setSize(w, h);
    _renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    _renderer.setClearColor(0x0e1117, 1);

    // Scene
    _scene = new THREE.Scene();
    _group = new THREE.Group();
    _scene.add(_group);

    // Camera
    _camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 1000);
    _camera.position.set(0, 0, 150);

    // Lights
    const ambient = new THREE.AmbientLight(0xffffff, 0.4);
    _scene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(50, 80, 100);
    _scene.add(dirLight);
    const hemiLight = new THREE.HemisphereLight(0x58a6ff, 0x1a1e24, 0.3);
    _scene.add(hemiLight);

    // OrbitControls (from CDN Three addons)
    if (THREE.OrbitControls) {
      _controls = new THREE.OrbitControls(_camera, _canvas);
      _controls.enableDamping = true;
      _controls.dampingFactor = 0.08;
      _controls.rotateSpeed = 0.6;
      _controls.zoomSpeed = 0.8;
    }

    // Build mesh geometry
    const meshData = Store.get("viewer3d.mesh");
    if (meshData) _addMesh(meshData);

    // Landmarks
    const landmarks = Store.get("viewer3d.landmarks");
    if (landmarks && _showLandmarks) _addLandmarks(landmarks);

    // Start render loop
    _active = true;
    _animate();
    _updateInfo();

    window.removeEventListener("resize", _resize);
    window.addEventListener("resize", _resize);
  }

  function _addMesh(data) {
    if (!_group) return;

    const positions = data.positions || data.vertices || [];
    const indices = data.indices || data.faces || [];
    const regions = data.region_ids || [];
    const regionColors = data.region_colors || {};

    if (positions.length === 0) return;

    const geometry = new THREE.BufferGeometry();
    const posArr = new Float32Array(positions.flat ? positions.flat() : positions);
    geometry.setAttribute("position", new THREE.BufferAttribute(posArr, 3));

    if (indices.length > 0) {
      const idxArr = new Uint32Array(indices.flat ? indices.flat() : indices);
      geometry.setIndex(new THREE.BufferAttribute(idxArr, 1));
    }

    // Per-vertex colors by region
    if (regions.length > 0 && _showRegions) {
      const colors = new Float32Array(posArr.length);
      const vertCount = posArr.length / 3;
      for (let i = 0; i < vertCount; i++) {
        const rid = regions[i] !== undefined ? regions[i] : 0;
        const hex = regionColors[rid] || regionColors[String(rid)] || "#58a6ff";
        const c = _hexToRGB(hex);
        colors[i * 3] = c[0];
        colors[i * 3 + 1] = c[1];
        colors[i * 3 + 2] = c[2];
      }
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    }

    geometry.computeVertexNormals();

    // Center geometry
    geometry.computeBoundingBox();
    const center = new THREE.Vector3();
    geometry.boundingBox.getCenter(center);
    geometry.translate(-center.x, -center.y, -center.z);

    // Fit camera
    const size = new THREE.Vector3();
    geometry.boundingBox.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    _camera.position.set(0, 0, maxDim * 1.8);
    if (_controls) _controls.target.set(0, 0, 0);

    const material = new THREE.MeshPhongMaterial({
      vertexColors: regions.length > 0 && _showRegions,
      color: regions.length === 0 || !_showRegions ? 0x58a6ff : 0xffffff,
      wireframe: _wireframe,
      transparent: _opacity < 1,
      opacity: _opacity,
      side: THREE.DoubleSide,
      flatShading: false,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = "facial_mesh";
    _group.add(mesh);

    // Wireframe overlay (subtle)
    if (!_wireframe) {
      const wfMat = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.03 });
      const wfMesh = new THREE.Mesh(geometry, wfMat);
      wfMesh.name = "wireframe_overlay";
      _group.add(wfMesh);
    }
  }

  function _addLandmarks(landmarks) {
    if (!_group || !landmarks || landmarks.length === 0) return;

    const arr = Array.isArray(landmarks) ? landmarks : (landmarks.points || []);
    const geometry = new THREE.BufferGeometry();
    const positions = [];
    const lmNames = [];

    arr.forEach(lm => {
      const coords = lm.coords || lm.position || [lm.x, lm.y, lm.z];
      if (coords && coords.length >= 3) {
        positions.push(coords[0], coords[1], coords[2]);
        lmNames.push(lm.name || lm.id || "");
      }
    });

    if (positions.length === 0) return;

    const posArr = new Float32Array(positions);
    geometry.setAttribute("position", new THREE.BufferAttribute(posArr, 3));

    // Shift by same offset as mesh
    const meshObj = _group.getObjectByName("facial_mesh");
    if (meshObj && meshObj.geometry && meshObj.geometry.boundingBox) {
      const center = new THREE.Vector3();
      // The mesh geometry was already centered, landmarks too
    }

    const material = new THREE.PointsMaterial({
      color: 0xffa657,
      size: 3,
      sizeAttenuation: true,
    });

    const points = new THREE.Points(geometry, material);
    points.name = "landmarks";
    _group.add(points);
  }

  /* ── HUD controls ──────────────────────────────────────── */

  function _bindHUD() {
    const wf = document.getElementById("hud-wireframe");
    if (wf) wf.addEventListener("change", () => { _wireframe = wf.checked; _applyMeshSettings(); });

    const op = document.getElementById("hud-opacity");
    if (op) op.addEventListener("input", () => { _opacity = parseFloat(op.value); _applyMeshSettings(); });

    const lm = document.getElementById("hud-landmarks");
    if (lm) lm.addEventListener("change", () => { _showLandmarks = lm.checked; _toggleLandmarks(); });

    const rg = document.getElementById("hud-regions");
    if (rg) rg.addEventListener("change", () => { _showRegions = rg.checked; _rebuildMesh(); });

    const reset = document.getElementById("hud-reset");
    if (reset) reset.addEventListener("click", () => {
      _wireframe = false;
      _opacity = 1.0;
      _showLandmarks = true;
      _showRegions = true;
      if (_controls) _controls.reset();
      _buildScene();
      render();
    });
  }

  function _applyMeshSettings() {
    if (!_group) return;
    _group.traverse(child => {
      if (child.isMesh && child.name === "facial_mesh") {
        child.material.wireframe = _wireframe;
        child.material.opacity = _opacity;
        child.material.transparent = _opacity < 1;
        child.material.needsUpdate = true;
      }
    });
  }

  function _toggleLandmarks() {
    if (!_group) return;
    const pts = _group.getObjectByName("landmarks");
    if (pts) pts.visible = _showLandmarks;
  }

  function _rebuildMesh() {
    if (!_group) return;
    // Remove old mesh objects and rebuild
    const toRemove = [];
    _group.traverse(child => {
      if (child.isMesh || child.isPoints) toRemove.push(child);
    });
    toRemove.forEach(obj => { _group.remove(obj); if (obj.geometry) obj.geometry.dispose(); if (obj.material) obj.material.dispose(); });

    const meshData = Store.get("viewer3d.mesh");
    if (meshData) _addMesh(meshData);

    const landmarks = Store.get("viewer3d.landmarks");
    if (landmarks && _showLandmarks) _addLandmarks(landmarks);
  }

  /* ── Render loop ───────────────────────────────────────── */

  function _animate() {
    if (!_active) return;
    _animId = requestAnimationFrame(_animate);
    if (_controls) _controls.update();
    if (_renderer && _scene && _camera) _renderer.render(_scene, _camera);
  }

  function _resize() {
    if (!_canvas || !_renderer || !_camera) return;
    const parent = _canvas.parentElement;
    if (!parent) return;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    _renderer.setSize(w, h);
    _camera.aspect = w / h;
    _camera.updateProjectionMatrix();
  }

  function _updateInfo() {
    const info = document.getElementById("viewer-info");
    if (!info) return;
    const mesh = Store.get("viewer3d.mesh") || {};
    const lm = Store.get("viewer3d.landmarks") || [];
    const verts = mesh.positions ? (Array.isArray(mesh.positions[0]) ? mesh.positions.length : mesh.positions.length / 3) : 0;
    const faces = mesh.indices ? (Array.isArray(mesh.indices[0]) ? mesh.indices.length : mesh.indices.length / 3) : 0;
    const lmCount = Array.isArray(lm) ? lm.length : (lm.points ? lm.points.length : 0);
    info.innerHTML = `V: ${verts} | F: ${faces} | LM: ${lmCount}`;
  }

  /* ── Canvas2D fallback (no Three.js) ───────────────────── */

  function _fallbackCanvas2D() {
    if (!_canvas) return;
    const ctx = _canvas.getContext("2d");
    if (!ctx) return;

    const parent = _canvas.parentElement;
    _canvas.width = parent.clientWidth || 800;
    _canvas.height = parent.clientHeight || 600;

    ctx.fillStyle = "#0e1117";
    ctx.fillRect(0, 0, _canvas.width, _canvas.height);

    const meshData = Store.get("viewer3d.mesh");
    if (!meshData || !meshData.positions) {
      ctx.fillStyle = "rgba(255,255,255,0.4)";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No mesh data available", _canvas.width / 2, _canvas.height / 2);
      return;
    }

    // Simple orthographic projection of mesh edges
    const positions = meshData.positions;
    const indices = meshData.indices || [];
    const flat = Array.isArray(positions[0]);

    let xs = [], ys = [];
    const verts = flat ? positions : [];
    if (!flat) {
      for (let i = 0; i < positions.length; i += 3) {
        verts.push([positions[i], positions[i + 1], positions[i + 2]]);
      }
    }
    verts.forEach(v => { xs.push(v[0]); ys.push(v[1]); });

    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const ymin = Math.min(...ys), ymax = Math.max(...ys);
    const xrange = xmax - xmin || 1;
    const yrange = ymax - ymin || 1;
    const pad = 40;
    const gw = _canvas.width - pad * 2;
    const gh = _canvas.height - pad * 2;
    const scale = Math.min(gw / xrange, gh / yrange);
    const cx = _canvas.width / 2;
    const cy = _canvas.height / 2;
    const mx = (xmin + xmax) / 2;
    const my = (ymin + ymax) / 2;
    const tx = v => cx + (v[0] - mx) * scale;
    const ty = v => cy - (v[1] - my) * scale;

    // Draw edges
    ctx.strokeStyle = "rgba(88, 166, 255, 0.15)";
    ctx.lineWidth = 0.5;
    const faceArr = Array.isArray(indices[0]) ? indices : [];
    if (!Array.isArray(indices[0])) {
      for (let i = 0; i < indices.length; i += 3) {
        faceArr.push([indices[i], indices[i + 1], indices[i + 2]]);
      }
    }
    faceArr.forEach(f => {
      if (f.length < 3) return;
      const a = verts[f[0]], b = verts[f[1]], c = verts[f[2]];
      if (!a || !b || !c) return;
      ctx.beginPath();
      ctx.moveTo(tx(a), ty(a));
      ctx.lineTo(tx(b), ty(b));
      ctx.lineTo(tx(c), ty(c));
      ctx.closePath();
      ctx.stroke();
    });

    // Draw landmarks
    const landmarks = Store.get("viewer3d.landmarks") || [];
    const lmArr = Array.isArray(landmarks) ? landmarks : (landmarks.points || []);
    ctx.fillStyle = "#ffa657";
    lmArr.forEach(lm => {
      const coords = lm.coords || lm.position || [lm.x, lm.y, lm.z];
      if (!coords || coords.length < 2) return;
      const x = cx + (coords[0] - mx) * scale;
      const y = cy - (coords[1] - my) * scale;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.font = "11px monospace";
    ctx.textAlign = "left";
    ctx.fillText(`Canvas2D fallback | V: ${verts.length} | F: ${faceArr.length} | LM: ${lmArr.length}`, 10, _canvas.height - 10);
  }

  /* ── Preview renderer (for Twin Inspect) ───────────────── */

  function renderPreview(canvasEl) {
    if (!canvasEl) return;

    if (typeof THREE !== "undefined") {
      // Render a single frame to the provided canvas
      const w = canvasEl.clientWidth || 320;
      const h = canvasEl.clientHeight || 240;
      canvasEl.width = w;
      canvasEl.height = h;

      const renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true, alpha: true });
      renderer.setSize(w, h);
      renderer.setClearColor(0x0e1117, 1);

      const scene = new THREE.Scene();
      scene.add(new THREE.AmbientLight(0xffffff, 0.5));
      const dl = new THREE.DirectionalLight(0xffffff, 0.7);
      dl.position.set(30, 50, 80);
      scene.add(dl);

      const cam = new THREE.PerspectiveCamera(50, w / h, 0.1, 1000);

      const meshData = Store.get("viewer3d.mesh") || Store.get("selectedCase.twin.mesh");
      if (meshData && meshData.positions) {
        const posArr = new Float32Array(meshData.positions.flat ? meshData.positions.flat() : meshData.positions);
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
        if (meshData.indices) {
          const idxArr = new Uint32Array(meshData.indices.flat ? meshData.indices.flat() : meshData.indices);
          geo.setIndex(new THREE.BufferAttribute(idxArr, 1));
        }
        geo.computeVertexNormals();
        geo.computeBoundingBox();
        const center = new THREE.Vector3();
        geo.boundingBox.getCenter(center);
        geo.translate(-center.x, -center.y, -center.z);
        const size = new THREE.Vector3();
        geo.boundingBox.getSize(size);
        cam.position.set(0, 0, Math.max(size.x, size.y, size.z) * 1.8);

        const mat = new THREE.MeshPhongMaterial({ color: 0x58a6ff, side: THREE.DoubleSide });
        scene.add(new THREE.Mesh(geo, mat));
      }

      renderer.render(scene, cam);
      renderer.dispose();
      return;
    }

    // Canvas2D fallback for preview
    const ctx = canvasEl.getContext("2d");
    if (!ctx) return;
    canvasEl.width = canvasEl.clientWidth || 320;
    canvasEl.height = canvasEl.clientHeight || 240;
    ctx.fillStyle = "#0e1117";
    ctx.fillRect(0, 0, canvasEl.width, canvasEl.height);
    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("3D Preview", canvasEl.width / 2, canvasEl.height / 2);
  }

  /* ── Helpers ───────────────────────────────────────────── */

  function _hexToRGB(hex) {
    const h = hex.replace("#", "");
    return [
      parseInt(h.substring(0, 2), 16) / 255,
      parseInt(h.substring(2, 4), 16) / 255,
      parseInt(h.substring(4, 6), 16) / 255,
    ];
  }

  return { init, load, render, activate, deactivate, renderPreview };
})();
