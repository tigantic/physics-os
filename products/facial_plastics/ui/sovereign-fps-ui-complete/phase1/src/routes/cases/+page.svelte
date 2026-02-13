<script>
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import {
    casesStore,
    contractStore,
    loadCases,
    createNewCase,
    removeCase,
    runCuration,
  } from '$lib/stores';

  // ── Filter state ───────────────────────────────────────────
  let filterProcedure = '';
  let filterQuality = '';
  let currentPage = 0;
  const pageSize = 20;

  // ── Modal state ────────────────────────────────────────────
  let showCreateModal = false;
  let showDeleteModal = false;
  let deleteTargetId = '';
  let deleteTargetName = '';

  // ── Create form ────────────────────────────────────────────
  let newCase = {
    patient_age: 35,
    patient_sex: 'female',
    procedure: 'rhinoplasty',
    notes: '',
  };

  // ── Curation state ─────────────────────────────────────────
  let curateLoading = false;
  let curateResult = null;

  // ── Derived from contract ──────────────────────────────────
  $: procedures = $contractStore.data?.procedures ?? [];

  // ── Load cases when filters change ─────────────────────────
  $: {
    const opts = {
      limit: pageSize,
      offset: currentPage * pageSize,
    };
    if (filterProcedure) opts.procedure = filterProcedure;
    if (filterQuality) opts.quality = filterQuality;
    loadCases(opts);
  }

  // ── Actions ────────────────────────────────────────────────
  async function handleCreate() {
    await createNewCase(newCase);
    showCreateModal = false;
    newCase = { patient_age: 35, patient_sex: 'female', procedure: 'rhinoplasty', notes: '' };
  }

  function confirmDelete(caseId) {
    deleteTargetId = caseId;
    deleteTargetName = caseId.substring(0, 12) + '...';
    showDeleteModal = true;
  }

  async function handleDelete() {
    await removeCase(deleteTargetId);
    showDeleteModal = false;
    deleteTargetId = '';
  }

  async function handleCurate() {
    curateLoading = true;
    try {
      curateResult = await runCuration();
    } catch (err) {
      curateResult = { error: err.message };
    }
    curateLoading = false;
  }

  function navigateToCase(caseId) {
    goto(`/cases/${caseId}`);
  }

  // ── Pagination ─────────────────────────────────────────────
  $: totalCases = $casesStore.data?.total ?? 0;
  $: totalPages = Math.ceil(totalCases / pageSize);
  $: cases = $casesStore.data?.cases ?? [];

  function prevPage() {
    if (currentPage > 0) currentPage--;
  }

  function nextPage() {
    if (currentPage < totalPages - 1) currentPage++;
  }

  // ── Helpers ────────────────────────────────────────────────
  function procedureLabel(val) {
    if (!val) return '—';
    return val.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  function truncateId(id) {
    if (!id) return '';
    return id.length > 16 ? id.substring(0, 8) + '…' + id.substring(id.length - 4) : id;
  }
</script>

<!-- Page Header -->
<div class="sov-page-header">
  <h1 class="sov-page-title">Case Library</h1>
  <p class="sov-page-subtitle">Patient cases and digital twin management</p>
</div>

<!-- Stats Row -->
<div class="sov-stat-row">
  <div class="sov-stat">
    <span class="sov-stat-value accent">{totalCases}</span>
    <span class="sov-stat-label">Total Cases</span>
  </div>
  <div class="sov-stat">
    <span class="sov-stat-value">{$contractStore.data?.operators?.count ?? '—'}</span>
    <span class="sov-stat-label">Operators</span>
  </div>
  <div class="sov-stat">
    <span class="sov-stat-value">{Object.values($contractStore.data?.templates?.templates ?? {}).flat().length || '—'}</span>
    <span class="sov-stat-label">Templates</span>
  </div>
  <div class="sov-stat">
    <span class="sov-stat-value">{procedures.length || '—'}</span>
    <span class="sov-stat-label">Procedures</span>
  </div>
</div>

<!-- Error Banner -->
{#if $casesStore.error}
  <div class="sov-error-banner">
    <span>⚠</span>
    <span>{$casesStore.error}</span>
  </div>
{/if}

<!-- Curate Result -->
{#if curateResult}
  <div class="sov-error-banner" style="border-color: var(--sov-accent)30; background: var(--sov-accent)08; color: var(--sov-accent);">
    <span>✓</span>
    <span>Curation complete — {JSON.stringify(curateResult).substring(0, 120)}</span>
    <button class="sov-btn sov-btn-ghost sov-btn-sm" style="margin-left: auto;"
      on:click={() => curateResult = null}>✕</button>
  </div>
{/if}

<!-- Toolbar: Filters + Actions -->
<div class="sov-toolbar">
  <select class="sov-select" style="width: 180px;"
    bind:value={filterProcedure}>
    <option value="">All Procedures</option>
    {#each procedures as proc}
      <option value={proc}>{procedureLabel(proc)}</option>
    {/each}
  </select>

  <select class="sov-select" style="width: 160px;"
    bind:value={filterQuality}>
    <option value="">All Quality</option>
    <option value="clinical">Clinical</option>
    <option value="research">Research</option>
    <option value="training">Training</option>
    <option value="synthetic">Synthetic</option>
  </select>

  <div class="sov-toolbar-spacer"></div>

  <button class="sov-btn sov-btn-secondary"
    on:click={handleCurate}
    disabled={curateLoading}>
    {curateLoading ? 'Curating...' : '⟳ Curate'}
  </button>

  <button class="sov-btn sov-btn-primary"
    on:click={() => showCreateModal = true}>
    + New Case
  </button>
</div>

<!-- Case Table -->
<div class="sov-card">
  {#if $casesStore.loading}
    <div class="sov-loading">
      <div class="sov-spinner"></div>
      <span>Loading cases...</span>
    </div>

  {:else if cases.length === 0}
    <div class="sov-empty">
      <div class="sov-empty-title">No cases found</div>
      <p>Create a new case or adjust filters to see results.</p>
    </div>

  {:else}
    <div class="sov-table-wrap">
      <table class="sov-table">
        <thead>
          <tr>
            <th>Case ID</th>
            <th>Procedure</th>
            <th>Quality</th>
            <th>Status</th>
            <th style="width: 80px;"></th>
          </tr>
        </thead>
        <tbody>
          {#each cases as c}
            <tr>
              <td>
                <span class="cell-id" on:click={() => navigateToCase(c.case_id)}
                  on:keydown={(e) => e.key === 'Enter' && navigateToCase(c.case_id)}
                  role="button" tabindex="0">
                  {truncateId(c.case_id)}
                </span>
              </td>
              <td>
                <span class="sov-badge sov-badge-accent">
                  {procedureLabel(c.procedure_type)}
                </span>
              </td>
              <td>
                <span class="sov-badge sov-badge-default">
                  {c.quality_level ?? '—'}
                </span>
              </td>
              <td>
                <span class="sov-badge sov-badge-success">Ready</span>
              </td>
              <td style="text-align: right;">
                <button class="sov-btn sov-btn-ghost sov-btn-sm"
                  on:click={() => navigateToCase(c.case_id)}
                  title="Open case">
                  →
                </button>
                <button class="sov-btn sov-btn-danger sov-btn-sm"
                  on:click={() => confirmDelete(c.case_id)}
                  title="Delete case">
                  ✕
                </button>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>

    <!-- Pagination -->
    <div class="sov-pagination">
      <span class="sov-pagination-info">
        {currentPage * pageSize + 1}–{Math.min((currentPage + 1) * pageSize, totalCases)}
        of {totalCases}
      </span>
      <div class="sov-pagination-controls">
        <button class="sov-btn sov-btn-ghost sov-btn-sm"
          disabled={currentPage === 0}
          on:click={prevPage}>
          ‹ Prev
        </button>
        <span class="font-data" style="padding: 0 8px; font-size: 12px; color: var(--sov-text-tertiary);">
          {currentPage + 1} / {totalPages}
        </span>
        <button class="sov-btn sov-btn-ghost sov-btn-sm"
          disabled={currentPage >= totalPages - 1}
          on:click={nextPage}>
          Next ›
        </button>
      </div>
    </div>
  {/if}
</div>

<!-- Create Case Modal -->
{#if showCreateModal}
  <div class="sov-modal-overlay" on:click|self={() => showCreateModal = false}
    role="dialog" aria-modal="true">
    <div class="sov-modal">
      <div class="sov-modal-header">
        <span class="sov-modal-title">New Case</span>
        <button class="sov-btn sov-btn-ghost sov-btn-sm"
          on:click={() => showCreateModal = false}>✕</button>
      </div>
      <div class="sov-modal-body">
        <div>
          <label class="sov-label">Procedure</label>
          <select class="sov-select" style="width: 100%;"
            bind:value={newCase.procedure}>
            {#each procedures as proc}
              <option value={proc}>{procedureLabel(proc)}</option>
            {/each}
          </select>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
          <div>
            <label class="sov-label">Patient Age</label>
            <input class="sov-input" type="number" min="0" max="120"
              bind:value={newCase.patient_age} />
          </div>
          <div>
            <label class="sov-label">Patient Sex</label>
            <select class="sov-select" style="width: 100%;"
              bind:value={newCase.patient_sex}>
              <option value="female">Female</option>
              <option value="male">Male</option>
              <option value="unknown">Unknown</option>
            </select>
          </div>
        </div>
        <div>
          <label class="sov-label">Notes</label>
          <input class="sov-input" type="text"
            placeholder="Optional notes..."
            bind:value={newCase.notes} />
        </div>
      </div>
      <div class="sov-modal-footer">
        <button class="sov-btn sov-btn-secondary"
          on:click={() => showCreateModal = false}>Cancel</button>
        <button class="sov-btn sov-btn-primary"
          on:click={handleCreate}>Create Case</button>
      </div>
    </div>
  </div>
{/if}

<!-- Delete Confirmation Modal -->
{#if showDeleteModal}
  <div class="sov-modal-overlay" on:click|self={() => showDeleteModal = false}
    role="dialog" aria-modal="true">
    <div class="sov-modal">
      <div class="sov-modal-header">
        <span class="sov-modal-title">Delete Case</span>
        <button class="sov-btn sov-btn-ghost sov-btn-sm"
          on:click={() => showDeleteModal = false}>✕</button>
      </div>
      <div class="sov-modal-body">
        <p style="color: var(--sov-text-secondary); font-size: 13px;">
          Are you sure you want to delete case
          <code class="font-data text-accent">{deleteTargetName}</code>?
          This action is permanent and recorded in the audit trail.
        </p>
      </div>
      <div class="sov-modal-footer">
        <button class="sov-btn sov-btn-secondary"
          on:click={() => showDeleteModal = false}>Cancel</button>
        <button class="sov-btn sov-btn-danger"
          on:click={handleDelete}>Delete</button>
      </div>
    </div>
  </div>
{/if}
