<script>
  /**
   * Dynamic form generated entirely from operator param_defs.
   * Every field type, min, max, unit, enum comes from the API.
   * Zero hardcoded form fields.
   *
   * @type {Record<string, import('$lib/api-client').ParamDef>}
   */
  export let paramDefs = {};

  /** Current parameter values */
  export let values = {};

  /** Read-only mode */
  export let readonly = false;

  /** Compact layout */
  export let compact = false;

  /** @type {(detail: { key: string, value: any, allValues: Record<string, any> }) => void} */
  export let onChange = () => {};

  // Initialize values from defaults
  $: {
    for (const [key, def] of Object.entries(paramDefs)) {
      if (values[key] === undefined && def.default !== null && def.default !== undefined) {
        values[key] = def.default;
      }
    }
  }

  function handleChange(key, val) {
    values[key] = val;
    values = values; // trigger reactivity
    onChange({ key, value: val, allValues: { ...values } });
  }

  function formatUnit(u) {
    if (!u || u === 'none' || u === 'unitless') return '';
    return u;
  }

  function formatLabel(name) {
    if (!name) return '';
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  $: entries = Object.entries(paramDefs);
</script>

{#if entries.length === 0}
  <div class="pe-empty">No parameters</div>
{:else}
  <div class="pe-grid" class:compact>
    {#each entries as [key, def]}
      <div class="pe-field">
        <label class="pe-label" for="param-{key}">
          <span class="pe-label-text">{formatLabel(def.name || key)}</span>
          {#if formatUnit(def.unit)}
            <span class="pe-unit">{formatUnit(def.unit)}</span>
          {/if}
        </label>

        {#if def.enum_values && def.enum_values.length > 0}
          <!-- Enum → select -->
          <select
            id="param-{key}"
            class="sov-select pe-select"
            value={values[key] ?? def.default ?? def.enum_values[0]}
            disabled={readonly}
            on:change={(e) => handleChange(key, e.target.value)}
          >
            {#each def.enum_values as opt}
              <option value={opt}>{formatLabel(opt)}</option>
            {/each}
          </select>

        {:else if def.param_type === 'bool' || def.param_type === 'boolean'}
          <!-- Boolean → toggle -->
          <button
            class="pe-toggle"
            class:active={values[key] ?? def.default}
            disabled={readonly}
            on:click={() => handleChange(key, !(values[key] ?? def.default))}
          >
            <span class="pe-toggle-track">
              <span class="pe-toggle-thumb"></span>
            </span>
            <span class="pe-toggle-label">
              {(values[key] ?? def.default) ? 'Yes' : 'No'}
            </span>
          </button>

        {:else if def.param_type === 'int' || def.param_type === 'integer'}
          <!-- Integer → number input -->
          <div class="pe-number-wrap">
            <input
              id="param-{key}"
              class="sov-input pe-input"
              type="number"
              step="1"
              min={def.min_value}
              max={def.max_value}
              value={values[key] ?? def.default ?? 0}
              disabled={readonly}
              on:input={(e) => handleChange(key, parseInt(e.target.value, 10))}
            />
            {#if def.min_value != null && def.max_value != null}
              <span class="pe-range">{def.min_value}–{def.max_value}</span>
            {/if}
          </div>

        {:else if def.param_type === 'float' || def.param_type === 'number'}
          <!-- Float → number input + range slider -->
          <div class="pe-number-wrap">
            <input
              id="param-{key}"
              class="sov-input pe-input"
              type="number"
              step="0.1"
              min={def.min_value}
              max={def.max_value}
              value={values[key] ?? def.default ?? 0}
              disabled={readonly}
              on:input={(e) => handleChange(key, parseFloat(e.target.value))}
            />
            {#if def.min_value != null && def.max_value != null}
              <input
                type="range"
                class="pe-slider"
                min={def.min_value}
                max={def.max_value}
                step={(def.max_value - def.min_value) / 100}
                value={values[key] ?? def.default ?? def.min_value}
                disabled={readonly}
                on:input={(e) => handleChange(key, parseFloat(e.target.value))}
              />
              <span class="pe-range">{def.min_value}–{def.max_value}</span>
            {/if}
          </div>

        {:else if def.param_type === 'string' || def.param_type === 'str'}
          <!-- String → text input -->
          <input
            id="param-{key}"
            class="sov-input pe-input"
            type="text"
            value={values[key] ?? def.default ?? ''}
            disabled={readonly}
            on:input={(e) => handleChange(key, e.target.value)}
          />

        {:else if def.param_type === 'array' || def.param_type === 'list'}
          <!-- Array → JSON textarea -->
          <textarea
            id="param-{key}"
            class="sov-input pe-input pe-json"
            rows="3"
            disabled={readonly}
            on:input={(e) => {
              try { handleChange(key, JSON.parse(e.target.value)); }
              catch { /* keep current value until valid JSON entered */ }
            }}
          >{JSON.stringify(values[key] ?? def.default ?? [], null, 2)}</textarea>

        {:else if def.param_type === 'object' || def.param_type === 'dict'}
          <!-- Object → JSON textarea -->
          <textarea
            id="param-{key}"
            class="sov-input pe-input pe-json"
            rows="4"
            disabled={readonly}
            on:input={(e) => {
              try { handleChange(key, JSON.parse(e.target.value)); }
              catch { /* keep current value until valid JSON entered */ }
            }}
          >{JSON.stringify(values[key] ?? def.default ?? {}, null, 2)}</textarea>

        {:else}
          <!-- Fallback: text input -->
          <input
            id="param-{key}"
            class="sov-input pe-input"
            type="text"
            value={values[key] ?? def.default ?? ''}
            disabled={readonly}
            on:input={(e) => handleChange(key, e.target.value)}
          />
        {/if}

        {#if def.description && !compact}
          <span class="pe-desc">{def.description}</span>
        {/if}
      </div>
    {/each}
  </div>
{/if}

<style>
  .pe-grid {
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .pe-grid.compact {
    gap: 8px;
  }

  .pe-field {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .pe-label {
    display: flex;
    align-items: baseline;
    gap: 6px;
  }

  .pe-label-text {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--sov-text-tertiary, #6B7280);
  }

  .pe-unit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--sov-text-muted, #4B5563);
    background: var(--sov-bg-elevated, #161920);
    padding: 0 4px;
    border-radius: 2px;
  }

  .pe-input {
    height: 30px !important;
    font-size: 12px !important;
  }

  .pe-select {
    height: 30px !important;
    font-size: 12px !important;
    width: 100%;
  }

  .pe-number-wrap {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .pe-number-wrap .pe-input {
    width: 100%;
  }

  .pe-slider {
    width: 100%;
    height: 4px;
    -webkit-appearance: none;
    appearance: none;
    background: var(--sov-bg-elevated, #161920);
    border-radius: 2px;
    outline: none;
    cursor: pointer;
  }

  .pe-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px;
    height: 14px;
    background: var(--sov-accent, #3B82F6);
    border-radius: 50%;
    cursor: pointer;
    border: 2px solid var(--sov-bg-card, #111318);
  }

  .pe-slider::-moz-range-thumb {
    width: 14px;
    height: 14px;
    background: var(--sov-accent, #3B82F6);
    border-radius: 50%;
    cursor: pointer;
    border: 2px solid var(--sov-bg-card, #111318);
  }

  .pe-slider:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .pe-range {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: var(--sov-text-muted, #4B5563);
    text-align: right;
  }

  .pe-desc {
    font-size: 10px;
    color: var(--sov-text-muted, #4B5563);
    line-height: 1.3;
  }

  .pe-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    background: none;
    border: none;
    cursor: pointer;
    font-family: inherit;
    padding: 4px 0;
  }

  .pe-toggle:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .pe-toggle-track {
    width: 32px;
    height: 18px;
    background: var(--sov-bg-elevated, #161920);
    border: 1px solid var(--sov-border, #1F2937);
    border-radius: 9px;
    position: relative;
    transition: all 150ms ease-out;
  }

  .pe-toggle.active .pe-toggle-track {
    background: var(--sov-accent-dim, #1E3A5F);
    border-color: var(--sov-accent, #3B82F6);
  }

  .pe-toggle-thumb {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 12px;
    height: 12px;
    background: var(--sov-text-muted, #4B5563);
    border-radius: 50%;
    transition: all 150ms ease-out;
  }

  .pe-toggle.active .pe-toggle-thumb {
    left: 16px;
    background: var(--sov-accent, #3B82F6);
  }

  .pe-toggle-label {
    font-size: 12px;
    color: var(--sov-text-secondary, #9CA3AF);
  }

  .pe-empty {
    font-size: 12px;
    color: var(--sov-text-muted, #4B5563);
    padding: 8px 0;
  }

  .pe-json {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    resize: vertical;
    min-height: 48px;
    white-space: pre;
    overflow-wrap: normal;
  }
</style>
