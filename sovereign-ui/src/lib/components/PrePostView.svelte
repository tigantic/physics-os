<script>
  /** @type {HTMLElement} */
  let splitContainer;

  /** @type {number} split position 0-100 */
  export let splitPosition = 50;

  /** @type {string} */
  export let preLabel = 'PRE-OP';

  /** @type {string} */
  export let postLabel = 'POST-OP PLAN';

  /** @type {boolean} */
  export let locked = false;

  /** @type {'overlay' | 'sidebyside' | 'slider'} */
  export let mode = 'slider';

  // ── Callback props (replaces createEventDispatcher) ────────
  /** @type {(position: number) => void} */
  export let onPositionChange = () => {};
  /** @type {(mode: string) => void} */
  export let onModeChange = () => {};

  let dragging = false;

  function handleMouseDown(e) {
    if (locked) return;
    dragging = true;
    e.preventDefault();
  }

  function handleMouseMove(e) {
    if (!dragging || !splitContainer) return;
    const rect = splitContainer.getBoundingClientRect();
    splitPosition = Math.max(5, Math.min(95, ((e.clientX - rect.left) / rect.width) * 100));
    onPositionChange(splitPosition);
  }

  function handleMouseUp() { dragging = false; }

  function handleTouchMove(e) {
    if (!dragging || !splitContainer) return;
    const rect = splitContainer.getBoundingClientRect();
    const touch = e.touches[0];
    splitPosition = Math.max(5, Math.min(95, ((touch.clientX - rect.left) / rect.width) * 100));
    onPositionChange(splitPosition);
  }
</script>

<svelte:window on:mousemove={handleMouseMove} on:mouseup={handleMouseUp} on:touchmove={handleTouchMove} on:touchend={handleMouseUp} />

<div class="ppv">
  <!-- Mode selector -->
  <div class="ppv-modes">
    {#each [['slider','⬌ Slider'], ['sidebyside','◫ Side by Side'], ['overlay','◉ Overlay']] as [m, lbl]}
      <button class="ppv-mode-btn" class:active={mode === m}
        on:click={() => { mode = m; onModeChange(m); }}>
        {lbl}
      </button>
    {/each}
  </div>

  {#if mode === 'slider'}
    <div class="ppv-split" bind:this={splitContainer}>
      <!-- Pre-op (full width, clipped by splitPosition) -->
      <div class="ppv-pane ppv-pre" style="clip-path: inset(0 {100 - splitPosition}% 0 0);">
        <div class="ppv-label ppv-label-left">{preLabel}</div>
        <slot name="pre" />
      </div>

      <!-- Post-op (full width, clipped inverse) -->
      <div class="ppv-pane ppv-post" style="clip-path: inset(0 0 0 {splitPosition}%);">
        <div class="ppv-label ppv-label-right">{postLabel}</div>
        <slot name="post" />
      </div>

      <!-- Divider -->
      <div class="ppv-divider" style="left: {splitPosition}%;"
        on:mousedown={handleMouseDown} on:touchstart={handleMouseDown}>
        <div class="ppv-divider-line"></div>
        <div class="ppv-divider-handle">
          <span class="ppv-handle-arrow">◀</span>
          <span class="ppv-handle-arrow">▶</span>
        </div>
      </div>
    </div>

  {:else if mode === 'sidebyside'}
    <div class="ppv-sbs">
      <div class="ppv-sbs-pane">
        <div class="ppv-label ppv-label-left">{preLabel}</div>
        <slot name="pre" />
      </div>
      <div class="ppv-sbs-divider"></div>
      <div class="ppv-sbs-pane">
        <div class="ppv-label ppv-label-right">{postLabel}</div>
        <slot name="post" />
      </div>
    </div>

  {:else}
    <!-- Overlay: post on top with opacity slider -->
    <div class="ppv-overlay-wrap">
      <div class="ppv-overlay-base">
        <div class="ppv-label ppv-label-left">{preLabel}</div>
        <slot name="pre" />
      </div>
      <div class="ppv-overlay-top" style="opacity: {splitPosition / 100};">
        <div class="ppv-label ppv-label-right">{postLabel} ({splitPosition.toFixed(0)}%)</div>
        <slot name="post" />
      </div>
    </div>
    <div class="ppv-overlay-slider">
      <span class="ppv-slider-label">{preLabel}</span>
      <input type="range" class="ppv-slider" min="0" max="100" step="1"
        bind:value={splitPosition} on:input={() => onPositionChange(splitPosition)} />
      <span class="ppv-slider-label">{postLabel}</span>
    </div>
  {/if}
</div>

<style>
  .ppv { display: flex; flex-direction: column; gap: 6px; }

  .ppv-modes { display: flex; gap: 3px; }
  .ppv-mode-btn {
    padding: 3px 8px; font-size: 10px; font-family: inherit;
    background: var(--sov-bg-elevated); border: 1px solid var(--sov-border); border-radius: 3px;
    color: var(--sov-text-muted); cursor: pointer;
  }
  .ppv-mode-btn.active { background: var(--sov-accent-dim); color: var(--sov-accent); border-color: var(--sov-accent)40; }

  /* Slider mode */
  .ppv-split { position: relative; width: 100%; height: 100%; min-height: 400px; overflow: hidden; border-radius: 6px; }
  .ppv-pane { position: absolute; inset: 0; }
  .ppv-pane :global(> *) { width: 100%; height: 100%; }

  .ppv-divider {
    position: absolute; top: 0; bottom: 0; width: 4px; transform: translateX(-50%);
    cursor: ew-resize; z-index: 10;
  }
  .ppv-divider-line { position: absolute; top: 0; bottom: 0; left: 50%; width: 2px; background: var(--sov-accent); transform: translateX(-50%); }
  .ppv-divider-handle {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    display: flex; align-items: center; gap: 2px;
    padding: 4px 6px; background: var(--sov-accent); border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
  }
  .ppv-handle-arrow { font-size: 8px; color: white; }

  /* Labels */
  .ppv-label {
    position: absolute; top: 10px; z-index: 5;
    padding: 3px 10px; font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700;
    letter-spacing: 0.06em; color: white; background: rgba(8,8,10,0.7); border-radius: 3px;
    backdrop-filter: blur(4px);
  }
  .ppv-label-left { left: 10px; }
  .ppv-label-right { right: 10px; }

  /* Side by side */
  .ppv-sbs { display: flex; gap: 0; min-height: 400px; border-radius: 6px; overflow: hidden; }
  .ppv-sbs-pane { flex: 1; position: relative; overflow: hidden; }
  .ppv-sbs-pane :global(> *:not(.ppv-label)) { width: 100%; height: 100%; }
  .ppv-sbs-divider { width: 2px; background: var(--sov-border); flex-shrink: 0; }

  /* Overlay */
  .ppv-overlay-wrap { position: relative; min-height: 400px; border-radius: 6px; overflow: hidden; }
  .ppv-overlay-base { position: absolute; inset: 0; }
  .ppv-overlay-base :global(> *:not(.ppv-label)) { width: 100%; height: 100%; }
  .ppv-overlay-top { position: absolute; inset: 0; pointer-events: none; }
  .ppv-overlay-top :global(> *:not(.ppv-label)) { width: 100%; height: 100%; }
  .ppv-overlay-slider { display: flex; align-items: center; gap: 8px; padding: 4px 0; }
  .ppv-slider-label { font-size: 10px; font-weight: 600; color: var(--sov-text-muted); white-space: nowrap; }
  .ppv-slider { flex: 1; height: 3px; -webkit-appearance: none; background: var(--sov-bg-elevated); border-radius: 2px; }
  .ppv-slider::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; background: var(--sov-accent); border-radius: 50%; cursor: pointer; }
</style>
