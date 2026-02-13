<script>
  /** @type {Record<string, LayerConfig>} */
  export let layerConfig = {};

  /** @type {string[]} ordered layer names */
  export let layerNames = [];

  /** @type {string | null} */
  export let highlightLayer = null;

  /** @type {(config: Record<string, any>) => void} */
  export let onChange = () => {};

  const LAYER_META = {
    // Specific named layers (checked first via longer keys)
    cartilage_upper:   { icon: '◆',  color: '#A8CCE0', label: 'Upper Lateral Cart.' },
    cartilage_lower:   { icon: '◆',  color: '#90C0D8', label: 'Lower Lateral Cart.' },
    cartilage_septal:  { icon: '◆',  color: '#B8D4E3', label: 'Septal Cartilage' },
    cartilage_columella: { icon: '◆', color: '#C0D8EC', label: 'Columella Cart.' },
    skin_dorsal:       { icon: '◎',  color: '#E8B89D', label: 'Dorsal Skin' },
    skin_tip:          { icon: '◎',  color: '#DCA888', label: 'Tip Skin' },
    skin_alar:         { icon: '◎',  color: '#D4A080', label: 'Alar Skin' },
    soft_tissue:       { icon: '◐',  color: '#D4888C', label: 'Soft Tissue' },
    // Generic fallbacks
    bone:              { icon: '🦴', color: '#F5F0E8', label: 'Bone' },
    cartilage:         { icon: '◆',  color: '#B8D4E3', label: 'Cartilage' },
    muscle:            { icon: '💪', color: '#CC4444', label: 'Muscle / SMAS' },
    skin:              { icon: '◎',  color: '#E8B89D', label: 'Skin' },
    fascia:            { icon: '≋',  color: '#E8D888', label: 'Fascia' },
  };

  function getMeta(name) {
    // Check exact match first, then substring match, longest keys first
    const lower = name.toLowerCase();
    if (LAYER_META[lower]) return LAYER_META[lower];
    const sorted = Object.entries(LAYER_META).sort((a, b) => b[0].length - a[0].length);
    for (const [k, v] of sorted) {
      if (lower.includes(k)) return v;
    }
    return { icon: '◻', color: '#6B7280', label: name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) };
  }

  function toggleVisible(name) {
    const cfg = layerConfig[name] || { visible: true, opacity: 0.8, wireframe: false };
    layerConfig[name] = { ...cfg, visible: !cfg.visible };
    layerConfig = { ...layerConfig };
    onChange(layerConfig);
  }

  function setOpacity(name, val) {
    const cfg = layerConfig[name] || { visible: true, opacity: 0.8, wireframe: false };
    layerConfig[name] = { ...cfg, opacity: parseFloat(val) };
    layerConfig = { ...layerConfig };
    onChange(layerConfig);
  }

  function toggleWireframe(name) {
    const cfg = layerConfig[name] || { visible: true, opacity: 0.8, wireframe: false };
    layerConfig[name] = { ...cfg, wireframe: !cfg.wireframe };
    layerConfig = { ...layerConfig };
    onChange(layerConfig);
  }

  function soloLayer(name) {
    const newConfig = {};
    for (const n of layerNames) {
      const cfg = layerConfig[n] || { visible: true, opacity: 0.8, wireframe: false };
      newConfig[n] = { ...cfg, visible: n === name };
    }
    layerConfig = newConfig;
    onChange(layerConfig);
  }

  function showAll() {
    const newConfig = {};
    for (const n of layerNames) {
      newConfig[n] = { visible: true, opacity: layerConfig[n]?.opacity ?? 0.8, wireframe: layerConfig[n]?.wireframe ?? false };
    }
    layerConfig = newConfig;
    onChange(layerConfig);
  }

  function xrayMode() {
    const newConfig = {};
    for (const n of layerNames) {
      const isBone = n.toLowerCase().includes('bone');
      newConfig[n] = {
        visible: true,
        opacity: isBone ? 1.0 : 0.15,
        wireframe: !isBone,
      };
    }
    layerConfig = newConfig;
    onChange(layerConfig);
  }

  function skinOnly() {
    const newConfig = {};
    for (const n of layerNames) {
      const isSkin = n.toLowerCase().includes('skin');
      newConfig[n] = { visible: true, opacity: isSkin ? 1.0 : 0.0, wireframe: false };
    }
    layerConfig = newConfig;
    onChange(layerConfig);
  }
</script>

<div class="lcp">
  <div class="lcp-header">
    <span class="lcp-title">Tissue Layers</span>
    <span class="sov-badge sov-badge-default">{layerNames.length}</span>
  </div>

  <div class="lcp-presets">
    <button class="lcp-preset" on:click={showAll} title="Show all layers">All</button>
    <button class="lcp-preset" on:click={xrayMode} title="X-ray: bone opaque, rest wireframe">X-Ray</button>
    <button class="lcp-preset" on:click={skinOnly} title="Surface only">Surface</button>
  </div>

  <div class="lcp-layers">
    {#each layerNames as name}
      {@const meta = getMeta(name)}
      {@const cfg = layerConfig[name] || { visible: true, opacity: 0.8, wireframe: false }}
      <div class="lcp-layer" class:highlighted={highlightLayer === name} class:hidden={!cfg.visible}>
        <div class="lcp-layer-head">
          <button class="lcp-vis-btn" class:off={!cfg.visible} on:click={() => toggleVisible(name)} title="Toggle visibility">
            <span class="lcp-dot" style="background: {meta.color}; opacity: {cfg.visible ? 1 : 0.2};"></span>
          </button>
          <span class="lcp-icon">{meta.icon}</span>
          <span class="lcp-name">{meta.label}</span>
          <button class="lcp-solo" on:click={() => soloLayer(name)} title="Solo">S</button>
          <button class="lcp-wire" class:active={cfg.wireframe} on:click={() => toggleWireframe(name)} title="Wireframe">△</button>
        </div>
        {#if cfg.visible}
          <div class="lcp-opacity-row">
            <input type="range" class="lcp-slider" min="0" max="1" step="0.05"
              value={cfg.opacity}
              on:input={(e) => setOpacity(name, e.target.value)} />
            <span class="lcp-opacity-val">{(cfg.opacity * 100).toFixed(0)}%</span>
          </div>
        {/if}
      </div>
    {/each}
  </div>
</div>

<style>
  .lcp { display: flex; flex-direction: column; gap: 0; }
  .lcp-header { display: flex; align-items: center; gap: 6px; padding: 8px 0 6px; }
  .lcp-title { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--sov-text-tertiary); }

  .lcp-presets { display: flex; gap: 3px; padding-bottom: 8px; border-bottom: 1px solid var(--sov-border-subtle, #151820); margin-bottom: 4px; }
  .lcp-preset {
    padding: 3px 8px; font-size: 10px; font-family: inherit;
    background: var(--sov-bg-elevated); border: 1px solid var(--sov-border); border-radius: 3px;
    color: var(--sov-text-muted); cursor: pointer;
  }
  .lcp-preset:hover { color: var(--sov-accent); border-color: var(--sov-accent); }

  .lcp-layers { display: flex; flex-direction: column; gap: 2px; }

  .lcp-layer {
    padding: 6px 4px; border-radius: 4px; transition: all 80ms ease-out;
    border: 1px solid transparent;
  }
  .lcp-layer:hover { background: var(--sov-bg-hover); }
  .lcp-layer.highlighted { border-color: var(--sov-accent)40; background: var(--sov-accent-glow); }
  .lcp-layer.hidden { opacity: 0.5; }

  .lcp-layer-head { display: flex; align-items: center; gap: 6px; }

  .lcp-vis-btn { background: none; border: none; cursor: pointer; padding: 2px; display: flex; }
  .lcp-dot { width: 10px; height: 10px; border-radius: 3px; border: 1px solid rgba(255,255,255,0.1); transition: opacity 80ms; }

  .lcp-icon { font-size: 12px; width: 18px; text-align: center; }
  .lcp-name { font-size: 12px; color: var(--sov-text-primary); flex: 1; }

  .lcp-solo, .lcp-wire {
    width: 18px; height: 18px; font-size: 9px; font-weight: 700;
    background: transparent; border: 1px solid var(--sov-border); border-radius: 2px;
    color: var(--sov-text-muted); cursor: pointer; display: flex; align-items: center; justify-content: center;
  }
  .lcp-solo:hover, .lcp-wire:hover { color: var(--sov-accent); border-color: var(--sov-accent); }
  .lcp-wire.active { color: var(--sov-accent); background: var(--sov-accent-dim); border-color: var(--sov-accent)40; }

  .lcp-opacity-row { display: flex; align-items: center; gap: 6px; padding: 4px 0 0 28px; }
  .lcp-slider {
    flex: 1; height: 3px; -webkit-appearance: none; appearance: none;
    background: var(--sov-bg-elevated); border-radius: 2px; outline: none;
  }
  .lcp-slider::-webkit-slider-thumb {
    -webkit-appearance: none; width: 10px; height: 10px;
    background: var(--sov-accent); border-radius: 50%; cursor: pointer;
    border: 1.5px solid var(--sov-bg-card);
  }
  .lcp-opacity-val { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--sov-text-muted); min-width: 28px; text-align: right; }
</style>
