<script>
  /**
   * Scientific colorbar for scalar field visualization.
   * Supports jet, viridis, plasma, inferno, coolwarm, and stress colormaps.
   */

  /** @type {'vertical' | 'horizontal'} */
  export let orientation = 'vertical';

  /** @type {string} */
  export let label = 'Stress';

  /** @type {string} */
  export let unit = 'MPa';

  /** @type {number} */
  export let min = 0;

  /** @type {number} */
  export let max = 1;

  /** @type {'jet' | 'viridis' | 'plasma' | 'inferno' | 'coolwarm' | 'stress'} */
  export let colormap = 'jet';

  /** @type {number} number of tick marks */
  export let ticks = 6;

  /** @type {number} decimal places */
  export let precision = 2;

  /** @type {boolean} */
  export let compact = false;

  const COLORMAPS = {
    jet: [
      { pos: 0,    color: '#00007F' },
      { pos: 0.15, color: '#0000FF' },
      { pos: 0.3,  color: '#00BFFF' },
      { pos: 0.45, color: '#00FF80' },
      { pos: 0.5,  color: '#40FF00' },
      { pos: 0.6,  color: '#BFFF00' },
      { pos: 0.75, color: '#FFFF00' },
      { pos: 0.85, color: '#FF8000' },
      { pos: 1,    color: '#FF0000' },
    ],
    viridis: [
      { pos: 0,    color: '#440154' },
      { pos: 0.25, color: '#3B528B' },
      { pos: 0.5,  color: '#21918C' },
      { pos: 0.75, color: '#5EC962' },
      { pos: 1,    color: '#FDE725' },
    ],
    plasma: [
      { pos: 0,    color: '#0D0887' },
      { pos: 0.25, color: '#7E03A8' },
      { pos: 0.5,  color: '#CC4778' },
      { pos: 0.75, color: '#F89441' },
      { pos: 1,    color: '#F0F921' },
    ],
    inferno: [
      { pos: 0,    color: '#000004' },
      { pos: 0.25, color: '#420A68' },
      { pos: 0.5,  color: '#932667' },
      { pos: 0.75, color: '#DD513A' },
      { pos: 1,    color: '#FCA50A' },
    ],
    coolwarm: [
      { pos: 0,    color: '#3B4CC0' },
      { pos: 0.25, color: '#7B9FF9' },
      { pos: 0.5,  color: '#F7F7F7' },
      { pos: 0.75, color: '#F4987A' },
      { pos: 1,    color: '#B40426' },
    ],
    stress: [
      { pos: 0,    color: '#0000FF' },
      { pos: 0.2,  color: '#00AAFF' },
      { pos: 0.4,  color: '#00FF55' },
      { pos: 0.6,  color: '#FFFF00' },
      { pos: 0.8,  color: '#FF8800' },
      { pos: 1,    color: '#FF0000' },
    ],
  };

  function gradientCSS(map) {
    const stops = COLORMAPS[map] || COLORMAPS.jet;
    const dir = orientation === 'vertical' ? 'to top' : 'to right';
    return `linear-gradient(${dir}, ${stops.map(s => `${s.color} ${s.pos * 100}%`).join(', ')})`;
  }

  function tickValues() {
    const vals = [];
    for (let i = 0; i < ticks; i++) {
      const t = i / (ticks - 1);
      vals.push(min + t * (max - min));
    }
    return orientation === 'vertical' ? vals.reverse() : vals;
  }

  /** Get the CSS color at a normalized value [0,1] */
  export function getColor(t) {
    const stops = COLORMAPS[colormap] || COLORMAPS.jet;
    t = Math.max(0, Math.min(1, t));
    for (let i = 0; i < stops.length - 1; i++) {
      if (t >= stops[i].pos && t <= stops[i + 1].pos) {
        const localT = (t - stops[i].pos) / (stops[i + 1].pos - stops[i].pos);
        return lerpColor(stops[i].color, stops[i + 1].color, localT);
      }
    }
    return stops[stops.length - 1].color;
  }

  /** Get RGB [0-1, 0-1, 0-1] at normalized value — for Three.js */
  export function getRGB(t) {
    const hex = getColor(t);
    return [
      parseInt(hex.slice(1, 3), 16) / 255,
      parseInt(hex.slice(3, 5), 16) / 255,
      parseInt(hex.slice(5, 7), 16) / 255,
    ];
  }

  function lerpColor(a, b, t) {
    const ar = parseInt(a.slice(1, 3), 16), ag = parseInt(a.slice(3, 5), 16), ab = parseInt(a.slice(5, 7), 16);
    const br = parseInt(b.slice(1, 3), 16), bg = parseInt(b.slice(3, 5), 16), bb = parseInt(b.slice(5, 7), 16);
    const r = Math.round(ar + (br - ar) * t);
    const g = Math.round(ag + (bg - ag) * t);
    const bl = Math.round(ab + (bb - ab) * t);
    return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${bl.toString(16).padStart(2,'0')}`;
  }

  $: gradient = gradientCSS(colormap);
  $: tickVals = tickValues();
</script>

<div class="cb" class:horizontal={orientation === 'horizontal'} class:compact>
  <div class="cb-title">
    <span class="cb-label">{label}</span>
    {#if unit}
      <span class="cb-unit">[{unit}]</span>
    {/if}
  </div>

  <div class="cb-body">
    {#if orientation === 'vertical'}
      <div class="cb-ticks-v">
        {#each tickVals as val}
          <span class="cb-tick">{val.toFixed(precision)}</span>
        {/each}
      </div>
      <div class="cb-bar-v" style="background: {gradient};"></div>
    {:else}
      <div class="cb-bar-h" style="background: {gradient};"></div>
      <div class="cb-ticks-h">
        {#each tickVals as val}
          <span class="cb-tick">{val.toFixed(precision)}</span>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .cb {
    display: flex;
    flex-direction: column;
    gap: 4px;
    pointer-events: none;
    user-select: none;
  }

  .cb-title {
    display: flex;
    align-items: baseline;
    gap: 4px;
  }

  .cb-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    color: var(--sov-text-secondary, #9CA3AF);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .cb-unit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: var(--sov-text-muted, #4B5563);
  }

  .cb-body { display: flex; }

  /* Vertical */
  .cb:not(.horizontal) .cb-body {
    flex-direction: row;
    gap: 4px;
  }

  .cb-bar-v {
    width: 14px;
    height: 160px;
    border-radius: 2px;
    border: 1px solid rgba(255,255,255,0.08);
    flex-shrink: 0;
  }

  .compact .cb-bar-v { height: 120px; width: 10px; }

  .cb-ticks-v {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 0 0 0 0;
  }

  /* Horizontal */
  .horizontal .cb-body {
    flex-direction: column;
    gap: 3px;
  }

  .cb-bar-h {
    height: 10px;
    width: 160px;
    border-radius: 2px;
    border: 1px solid rgba(255,255,255,0.08);
  }

  .compact .cb-bar-h { width: 120px; height: 8px; }

  .cb-ticks-h {
    display: flex;
    justify-content: space-between;
  }

  .cb-tick {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px;
    color: var(--sov-text-muted, #4B5563);
    line-height: 1;
  }
</style>
