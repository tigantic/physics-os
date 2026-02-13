<script>
  /** @type {DicomSeriesData | null} */
  export let dicomData = null;

  /** @type {'axial' | 'coronal' | 'sagittal'} */
  export let plane = 'axial';

  /** @type {number} slice index */
  export let sliceIndex = 0;

  /** @type {number} window center (HU) */
  export let windowCenter = 400;

  /** @type {number} window width (HU) */
  export let windowWidth = 2000;

  // ── Callback props (replaces createEventDispatcher) ────────
  /** @type {(detail: { center: number, width: number }) => void} */
  export let onWindowChange = () => {};
  /** @type {(index: number) => void} */
  export let onSliceChange = () => {};
  /** @type {() => void} */
  export let onUploadDicom = () => {};

  let canvas;
  let ctx;

  $: hasData = dicomData != null;
  $: sliceCount = getSliceCount(dicomData, plane);
  $: if (canvas && dicomData) renderSlice();

  // ── Svelte action for canvas initialization ────────────────
  /** @param {HTMLCanvasElement} node */
  function canvasSetup(node) {
    canvas = node;
    ctx = node.getContext('2d');
    if (dicomData) renderSlice();
    else renderEmpty();
    return { destroy() { canvas = null; ctx = null; } };
  }

  const PRESETS = [
    { name: 'Bone',        center: 400,  width: 2000 },
    { name: 'Soft Tissue', center: 40,   width: 400 },
    { name: 'Lung',        center: -600, width: 1600 },
    { name: 'Brain',       center: 40,   width: 80 },
    { name: 'Nasal',       center: 200,  width: 1200 },
  ];

  function getSliceCount(data, p) {
    if (!data?.dimensions) return 0;
    if (p === 'axial') return data.dimensions[2] ?? 0;
    if (p === 'coronal') return data.dimensions[1] ?? 0;
    if (p === 'sagittal') return data.dimensions[0] ?? 0;
    return 0;
  }

  function renderSlice() {
    if (!canvas || !dicomData) return;
    ctx = canvas.getContext('2d');
    if (!ctx) return;

    let sliceData;
    let width, height;

    if (dicomData.slices) {
      const slice = dicomData.slices[plane]?.[sliceIndex];
      if (!slice) { renderEmpty(); return; }
      width = slice.width;
      height = slice.height;
      sliceData = slice.pixels;
    } else if (dicomData.volume) {
      const dims = dicomData.dimensions;
      if (plane === 'axial') {
        width = dims[0]; height = dims[1];
        sliceData = extractAxial(dicomData.volume, dims, sliceIndex);
      } else if (plane === 'coronal') {
        width = dims[0]; height = dims[2];
        sliceData = extractCoronal(dicomData.volume, dims, sliceIndex);
      } else {
        width = dims[1]; height = dims[2];
        sliceData = extractSagittal(dicomData.volume, dims, sliceIndex);
      }
    } else {
      renderEmpty();
      return;
    }

    canvas.width = width;
    canvas.height = height;
    const imgData = ctx.createImageData(width, height);

    const lo = windowCenter - windowWidth / 2;
    const hi = windowCenter + windowWidth / 2;
    const range = hi - lo || 1;

    for (let i = 0; i < sliceData.length; i++) {
      const hu = sliceData[i];
      let val = Math.round(((hu - lo) / range) * 255);
      val = Math.max(0, Math.min(255, val));
      imgData.data[i * 4] = val;
      imgData.data[i * 4 + 1] = val;
      imgData.data[i * 4 + 2] = val;
      imgData.data[i * 4 + 3] = 255;
    }

    ctx.putImageData(imgData, 0, 0);
  }

  function extractAxial(vol, dims, z) {
    const [w, h] = dims;
    const slice = new Int16Array(w * h);
    const offset = z * w * h;
    for (let i = 0; i < w * h; i++) slice[i] = vol[offset + i] ?? 0;
    return slice;
  }

  function extractCoronal(vol, dims, y) {
    const [w, h, d] = dims;
    const slice = new Int16Array(w * d);
    for (let z = 0; z < d; z++) {
      for (let x = 0; x < w; x++) {
        slice[z * w + x] = vol[z * w * h + y * w + x] ?? 0;
      }
    }
    return slice;
  }

  function extractSagittal(vol, dims, x) {
    const [w, h, d] = dims;
    const slice = new Int16Array(h * d);
    for (let z = 0; z < d; z++) {
      for (let y = 0; y < h; y++) {
        slice[z * h + y] = vol[z * w * h + y * w + x] ?? 0;
      }
    }
    return slice;
  }

  function renderEmpty() {
    if (!canvas) return;
    canvas.width = 256;
    canvas.height = 256;
    ctx = canvas.getContext('2d');
    ctx.fillStyle = '#08080A';
    ctx.fillRect(0, 0, 256, 256);
    ctx.fillStyle = '#4B5563';
    ctx.font = '12px "JetBrains Mono"';
    ctx.textAlign = 'center';
    ctx.fillText('No DICOM data', 128, 128);
  }

  function applyPreset(p) {
    windowCenter = p.center;
    windowWidth = p.width;
    onWindowChange({ center: windowCenter, width: windowWidth });
  }

  function handleScroll(e) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 1 : -1;
    sliceIndex = Math.max(0, Math.min(sliceCount - 1, sliceIndex + delta));
    onSliceChange(sliceIndex);
  }
</script>

<div class="dcm">
  <div class="dcm-header">
    <span class="dcm-title">Imaging</span>
    <div class="dcm-planes">
      {#each [['axial','A'], ['coronal','C'], ['sagittal','S']] as [p, lbl]}
        <button class="dcm-plane-btn" class:active={plane === p}
          on:click={() => { plane = p; sliceIndex = Math.floor(getSliceCount(dicomData, p) / 2); }}>
          {lbl}
        </button>
      {/each}
    </div>
  </div>

  <div class="dcm-viewport" on:wheel={handleScroll}>
    <canvas use:canvasSetup></canvas>
    {#if hasData}
      <div class="dcm-info-tl">
        <span>{plane.toUpperCase()}</span>
        <span>Slice {sliceIndex + 1}/{sliceCount}</span>
      </div>
      <div class="dcm-info-tr">
        <span>W:{windowWidth} C:{windowCenter}</span>
      </div>
    {/if}
  </div>

  {#if hasData}
    <div class="dcm-controls">
      <div class="dcm-control">
        <span class="dcm-ctrl-label">Slice</span>
        <input type="range" class="dcm-slider" min="0" max={sliceCount - 1} step="1"
          bind:value={sliceIndex}
          on:input={() => onSliceChange(sliceIndex)} />
        <span class="dcm-ctrl-val">{sliceIndex + 1}</span>
      </div>
      <div class="dcm-control">
        <span class="dcm-ctrl-label">Center</span>
        <input type="range" class="dcm-slider" min="-1000" max="2000" step="10"
          bind:value={windowCenter}
          on:input={() => onWindowChange({ center: windowCenter, width: windowWidth })} />
        <span class="dcm-ctrl-val">{windowCenter}</span>
      </div>
      <div class="dcm-control">
        <span class="dcm-ctrl-label">Width</span>
        <input type="range" class="dcm-slider" min="1" max="4000" step="10"
          bind:value={windowWidth}
          on:input={() => onWindowChange({ center: windowCenter, width: windowWidth })} />
        <span class="dcm-ctrl-val">{windowWidth}</span>
      </div>
    </div>

    <div class="dcm-presets">
      {#each PRESETS as p}
        <button class="dcm-preset" class:active={windowCenter === p.center && windowWidth === p.width}
          on:click={() => applyPreset(p)}>
          {p.name}
        </button>
      {/each}
    </div>

    {#if dicomData.metadata}
      <div class="dcm-meta">
        {#if dicomData.metadata.modality}<div class="dcm-meta-row"><span>Modality</span><span>{dicomData.metadata.modality}</span></div>{/if}
        {#if dicomData.metadata.slice_thickness}<div class="dcm-meta-row"><span>Thickness</span><span>{dicomData.metadata.slice_thickness}mm</span></div>{/if}
        {#if dicomData.metadata.pixel_spacing}<div class="dcm-meta-row"><span>Spacing</span><span>{dicomData.metadata.pixel_spacing}mm</span></div>{/if}
        {#if dicomData.metadata.dimensions}<div class="dcm-meta-row"><span>Matrix</span><span>{dicomData.metadata.dimensions.join('×')}</span></div>{/if}
      </div>
    {/if}
  {:else}
    <div class="dcm-empty">
      <span>No imaging data</span>
      <button class="sov-btn sov-btn-secondary sov-btn-sm" on:click={onUploadDicom}>
        Upload DICOM
      </button>
    </div>
  {/if}
</div>

<style>
  .dcm { display: flex; flex-direction: column; gap: 6px; }
  .dcm-header { display: flex; justify-content: space-between; align-items: center; }
  .dcm-title { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--sov-text-tertiary); }
  .dcm-planes { display: flex; gap: 2px; }
  .dcm-plane-btn {
    width: 24px; height: 22px; font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700;
    background: var(--sov-bg-elevated); border: 1px solid var(--sov-border); border-radius: 3px;
    color: var(--sov-text-muted); cursor: pointer; display: flex; align-items: center; justify-content: center;
  }
  .dcm-plane-btn.active { background: var(--sov-accent-dim); color: var(--sov-accent); border-color: var(--sov-accent)40; }

  .dcm-viewport {
    position: relative; background: #000; border-radius: 4px; overflow: hidden;
    aspect-ratio: 1; max-height: 300px;
  }
  .dcm-viewport canvas { width: 100%; height: 100%; object-fit: contain; display: block; image-rendering: pixelated; }
  .dcm-info-tl, .dcm-info-tr {
    position: absolute; padding: 3px 6px; font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: #9CA3AF; background: rgba(0,0,0,0.6); border-radius: 2px;
  }
  .dcm-info-tl { top: 4px; left: 4px; display: flex; flex-direction: column; gap: 1px; }
  .dcm-info-tr { top: 4px; right: 4px; }

  .dcm-controls { display: flex; flex-direction: column; gap: 4px; }
  .dcm-control { display: flex; align-items: center; gap: 6px; }
  .dcm-ctrl-label { font-size: 10px; color: var(--sov-text-muted); min-width: 40px; }
  .dcm-slider { flex: 1; height: 3px; -webkit-appearance: none; background: var(--sov-bg-elevated); border-radius: 2px; }
  .dcm-slider::-webkit-slider-thumb { -webkit-appearance: none; width: 10px; height: 10px; background: var(--sov-accent); border-radius: 50%; cursor: pointer; }
  .dcm-ctrl-val { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--sov-text-muted); min-width: 32px; text-align: right; }

  .dcm-presets { display: flex; gap: 3px; flex-wrap: wrap; }
  .dcm-preset {
    padding: 2px 6px; font-size: 9px; background: var(--sov-bg-elevated); border: 1px solid var(--sov-border);
    border-radius: 3px; color: var(--sov-text-muted); cursor: pointer; font-family: inherit;
  }
  .dcm-preset.active, .dcm-preset:hover { color: var(--sov-accent); border-color: var(--sov-accent)40; }

  .dcm-meta { display: flex; flex-direction: column; gap: 2px; padding-top: 4px; border-top: 1px solid var(--sov-border-subtle); }
  .dcm-meta-row { display: flex; justify-content: space-between; font-size: 10px; color: var(--sov-text-tertiary); }
  .dcm-meta-row span:last-child { font-family: 'JetBrains Mono', monospace; color: var(--sov-text-secondary); }

  .dcm-empty { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 20px 0; font-size: 12px; color: var(--sov-text-muted); }
</style>
