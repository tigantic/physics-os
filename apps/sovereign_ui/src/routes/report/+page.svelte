<script>
  import {
    casesStore,
    activePlan,
    reportStore,
    generateReport,
  } from '$lib/stores';

  // ── State ──────────────────────────────────────────────────
  let selectedCaseId = '';
  let reportFormat = 'markdown';
  let includeImages = false;
  let includeMeasurements = true;
  let includeTimeline = true;
  let createError = '';

  // ── Set defaults (SSR off — safe at top level) ────────────
  if ($casesStore.data?.cases?.[0]) {
    selectedCaseId = $casesStore.data.cases[0].case_id;
  }

  $: plan = $activePlan;
  $: cases = $casesStore.data?.cases ?? [];
  $: hasPlan = plan && plan.n_steps > 0;
  $: report = $reportStore.data;

  async function handleGenerate() {
    if (!selectedCaseId) return;
    createError = '';
    try {
      await generateReport(selectedCaseId, reportFormat, {
        includeImages,
        includeMeasurements,
        includeTimeline,
      });
    } catch (err) {
      createError = err instanceof Error ? err.message : String(err);
    }
  }

  /** Sanitize HTML to prevent XSS — strips script tags and event handlers */
  function sanitizeHtml(html) {
    if (!html) return '';
    return html
      .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
      .replace(/on\w+\s*=\s*["'][^"']*["']/gi, '')
      .replace(/on\w+\s*=\s*\S+/gi, '');
  }

  function downloadReport() {
    if (!report?.content) return;
    const ext = reportFormat === 'markdown' ? 'md' : reportFormat === 'html' ? 'html' : 'json';
    const mime = reportFormat === 'html' ? 'text/html' : reportFormat === 'json' ? 'application/json' : 'text/markdown';
    const blob = new Blob([report.content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `report_${selectedCaseId.substring(0, 8)}.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function formatName(n) {
    if (!n) return '';
    return n.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }
</script>

<div class="sov-page-header">
  <h1 class="sov-page-title">Report Generation</h1>
  <p class="sov-page-subtitle">Clinical documentation and surgical plan reports</p>
</div>

{#if !hasPlan}
  <div class="sov-error-banner" style="border-color: var(--sov-warning); background: #F59E0B08; color: var(--sov-warning);">
    <span>⚠</span>
    <span>No active plan. Go to <a href="/plan" style="color: var(--sov-accent);">Plan Editor</a> to build one first.</span>
  </div>
{/if}

<div class="report-layout">
  <!-- Config -->
  <div class="sov-card">
    <div class="sov-card-header">
      <span class="sov-card-title">Report Configuration</span>
    </div>
    <div class="sov-card-body">
      <div class="config-grid">
        <div>
          <label class="sov-label" for="report-case">Case</label>
          <select class="sov-select" style="width: 100%;" id="report-case" bind:value={selectedCaseId}>
            <option value="">Select case...</option>
            {#each cases as c}
              <option value={c.case_id}>
                {c.case_id.substring(0, 12)}... — {formatName(c.procedure_type)}
              </option>
            {/each}
          </select>
        </div>

        <div>
          <label class="sov-label" for="report-format">Format</label>
          <select class="sov-select" style="width: 100%;" id="report-format" bind:value={reportFormat}>
            <option value="markdown">Markdown — readable text with headers</option>
            <option value="html">HTML — styled for browser preview / print</option>
            <option value="json">JSON — structured data for integration</option>
          </select>
        </div>

        <div class="config-toggles">
          <label class="config-toggle">
            <input type="checkbox" bind:checked={includeMeasurements} />
            <span>Measurements</span>
          </label>
          <label class="config-toggle">
            <input type="checkbox" bind:checked={includeTimeline} />
            <span>Timeline</span>
          </label>
          <label class="config-toggle">
            <input type="checkbox" bind:checked={includeImages} />
            <span>Images</span>
          </label>
        </div>
      </div>

      {#if createError}
        <div class="sov-error-banner" style="margin-top: 8px;">
          <span>⚠</span><span>{createError}</span>
        </div>
      {/if}

      {#if plan}
        <div class="plan-summary">
          <span class="sov-label">Active Plan</span>
          <div style="display: flex; align-items: center; gap: 8px; margin-top: 4px;">
            <span style="font-size: 13px; font-weight: 500; color: var(--sov-text-primary);">
              {plan.name || 'Untitled'}
            </span>
            <span class="sov-badge sov-badge-accent">{plan.n_steps} steps</span>
            <span class="sov-badge sov-badge-default">{formatName(plan.procedure)}</span>
            {#if plan.content_hash}
              <span class="font-data" style="font-size: 10px; color: var(--sov-text-muted);">
                #{plan.content_hash.substring(0, 8)}
              </span>
            {/if}
          </div>
        </div>
      {/if}

      <div style="margin-top: 16px; text-align: right;">
        <button class="sov-btn sov-btn-primary"
          disabled={!selectedCaseId || !hasPlan || $reportStore.loading}
          on:click={handleGenerate}>
          {$reportStore.loading ? 'Generating...' : '▤ Generate Report'}
        </button>
      </div>
    </div>
  </div>

  <!-- Preview -->
  <div class="sov-card report-preview-card">
    <div class="sov-card-header">
      <span class="sov-card-title">Preview</span>
      {#if report}
        <button class="sov-btn sov-btn-secondary sov-btn-sm" on:click={downloadReport}>
          ↓ Download
        </button>
      {/if}
    </div>
    <div class="sov-card-body">
      {#if $reportStore.loading}
        <div class="sov-loading" style="min-height: 200px;">
          <div class="sov-spinner"></div>
          <span>Generating report...</span>
        </div>
      {:else if $reportStore.error}
        <div class="sov-error-banner">
          <span>⚠</span><span>{$reportStore.error}</span>
        </div>
      {:else if report}
        <div class="report-meta">
          {#each Object.entries(report).filter(([k]) => k !== 'content') as [key, val]}
            <span class="report-meta-item">
              <span class="report-meta-label">{formatName(key)}</span>
              <span class="report-meta-val font-data">
                {typeof val === 'object' ? JSON.stringify(val).substring(0, 30) : String(val).substring(0, 40)}
              </span>
            </span>
          {/each}
        </div>

        <div class="report-content">
          {#if reportFormat === 'html'}
            <div class="report-html-frame">
              {@html sanitizeHtml(report.content)}
            </div>
          {:else if reportFormat === 'json'}
            <pre class="report-pre font-data">{JSON.stringify(JSON.parse(report.content || '{}'), null, 2)}</pre>
          {:else}
            <pre class="report-pre">{report.content}</pre>
          {/if}
        </div>
      {:else}
        <div class="sov-empty" style="padding: 32px;">
          <div class="sov-empty-title">No Report Generated</div>
          <p style="font-size: 12px;">Configure options and generate.</p>
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .report-layout {
    display: grid;
    grid-template-columns: 360px 1fr;
    gap: 16px;
    align-items: start;
  }

  .config-grid {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .config-toggles {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding-top: 4px;
  }

  .config-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--sov-text-secondary, #9CA3AF);
    cursor: pointer;
  }

  .config-toggle input[type="checkbox"] {
    accent-color: var(--sov-accent, #3B82F6);
  }

  .plan-summary {
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid var(--sov-border-subtle, #151820);
  }

  .report-preview-card {
    max-height: 700px;
    display: flex;
    flex-direction: column;
  }

  .report-preview-card .sov-card-body {
    overflow-y: auto;
    flex: 1;
  }

  .report-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    padding-bottom: 10px;
    margin-bottom: 10px;
    border-bottom: 1px solid var(--sov-border-subtle, #151820);
  }

  .report-meta-item {
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .report-meta-label {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--sov-text-muted, #4B5563);
  }

  .report-meta-val {
    font-size: 11px;
    color: var(--sov-text-secondary, #9CA3AF);
  }

  .report-content {
    background: var(--sov-bg-root, #08080A);
    border: 1px solid var(--sov-border-subtle, #151820);
    border-radius: 4px;
    overflow: auto;
  }

  .report-pre {
    padding: 14px;
    font-size: 12px;
    line-height: 1.5;
    color: var(--sov-text-secondary, #9CA3AF);
    white-space: pre-wrap;
    word-break: break-word;
    margin: 0;
  }

  .report-html-frame {
    padding: 14px;
    font-size: 13px;
    color: var(--sov-text-primary, #E8E8EC);
    line-height: 1.6;
  }

  .report-html-frame :global(h1),
  .report-html-frame :global(h2),
  .report-html-frame :global(h3) {
    color: var(--sov-text-primary, #E8E8EC);
    margin: 16px 0 8px;
  }

  .report-html-frame :global(table) {
    border-collapse: collapse;
    width: 100%;
    margin: 8px 0;
  }

  .report-html-frame :global(th),
  .report-html-frame :global(td) {
    border: 1px solid var(--sov-border, #1F2937);
    padding: 6px 8px;
    font-size: 12px;
  }

  @media (max-width: 900px) {
    .report-layout { grid-template-columns: 1fr; }
  }
</style>
