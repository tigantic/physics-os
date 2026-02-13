<script>
  import { contractStore, casesStore } from '$lib/stores';

  // ── Governance data (read from contract + derived) ─────────
  $: contract = $contractStore.data;
  $: totalCases = $casesStore.data?.total ?? 0;

  // Simulated governance state derived from real contract data
  // In production, these would come from a dedicated governance API
  $: platformVersion = contract?.version ?? '—';
  $: operatorCount = contract?.operators?.count ?? 0;
  $: procedureCount = contract?.procedures?.length ?? 0;

  // ── RBAC Roles (clinical platform standard) ────────────────
  const roles = [
    {
      name: 'Surgeon',
      level: 'clinical',
      permissions: ['case.read', 'case.write', 'plan.read', 'plan.write', 'plan.compile', 'whatif.execute', 'report.generate'],
      description: 'Full clinical access. Can create/modify cases and plans, run simulations, generate reports.',
    },
    {
      name: 'Fellow',
      level: 'clinical',
      permissions: ['case.read', 'plan.read', 'plan.write', 'whatif.execute'],
      description: 'Training access. Can view cases, create plans, run what-if scenarios. Cannot compile or generate reports without attending approval.',
    },
    {
      name: 'Researcher',
      level: 'research',
      permissions: ['case.read', 'plan.read', 'compare.execute', 'report.read'],
      description: 'Read-only clinical data. Can compare cases and plans, view reports. No write access.',
    },
    {
      name: 'Administrator',
      level: 'system',
      permissions: ['system.config', 'rbac.manage', 'audit.read', 'case.delete', 'curate.execute'],
      description: 'System management. RBAC configuration, audit trail access, data curation, case lifecycle management.',
    },
    {
      name: 'Auditor',
      level: 'compliance',
      permissions: ['audit.read', 'consent.read', 'report.read'],
      description: 'Compliance-only access. Read audit trail, verify consent records, review reports. No clinical access.',
    },
  ];

  // ── Data Classification ────────────────────────────────────
  const classifications = [
    { level: 'PHI', color: '#EF4444', label: 'Protected Health Information', examples: 'Patient demographics, case notes, surgical plans' },
    { level: 'Clinical', color: '#F59E0B', label: 'Clinical Data', examples: 'Mesh geometry, landmarks, twin summaries, operator parameters' },
    { level: 'Research', color: '#3B82F6', label: 'De-identified Research', examples: 'Aggregate statistics, comparison results, curated datasets' },
    { level: 'System', color: '#6B7280', label: 'System Metadata', examples: 'Audit events, platform configuration, operator schemas' },
  ];

  // ── Consent Requirements ───────────────────────────────────
  const consentItems = [
    { id: 'data_use', label: 'Data Use Consent', description: 'Authorization for digital twin creation and surgical simulation from patient imaging data.', required: true },
    { id: 'research', label: 'Research Participation', description: 'Opt-in consent for de-identified case data to be included in research cohorts.', required: false },
    { id: 'ai_planning', label: 'AI-Assisted Planning', description: 'Acknowledgment that computational planning tools augment but do not replace clinical judgment.', required: true },
    { id: 'data_retention', label: 'Data Retention', description: 'Agreement to data retention policy: active cases retained indefinitely, archived cases for 7 years, audit trail permanent.', required: true },
  ];

  // ── Audit Event Types ──────────────────────────────────────
  const auditEventTypes = [
    { event: 'case.create', severity: 'info', description: 'New case created' },
    { event: 'case.delete', severity: 'warning', description: 'Case permanently deleted' },
    { event: 'plan.compile', severity: 'info', description: 'Surgical plan compiled against case' },
    { event: 'plan.modify', severity: 'info', description: 'Plan steps added, removed, or reordered' },
    { event: 'whatif.execute', severity: 'info', description: 'What-if scenario executed' },
    { event: 'sweep.execute', severity: 'info', description: 'Parameter sweep executed' },
    { event: 'report.generate', severity: 'info', description: 'Clinical report generated' },
    { event: 'curate.execute', severity: 'warning', description: 'Case library curation triggered' },
    { event: 'auth.login', severity: 'info', description: 'User authenticated' },
    { event: 'auth.failed', severity: 'error', description: 'Authentication attempt failed' },
    { event: 'rbac.modify', severity: 'warning', description: 'Role assignment changed' },
    { event: 'consent.update', severity: 'warning', description: 'Patient consent record modified' },
  ];

  let activeSection = 'rbac'; // 'rbac' | 'audit' | 'consent' | 'classification'

  function formatName(n) {
    return n ? n.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) : '';
  }
</script>

<div class="sov-page-header">
  <h1 class="sov-page-title">Governance</h1>
  <p class="sov-page-subtitle">Access control, audit compliance, and data classification</p>
</div>

<!-- M8: Design preview banner — governance logic is aspirational until backend RBAC lands -->
<div class="sov-card" style="border-color: var(--sov-accent-blue); border-style: dashed; margin-bottom: 16px;">
  <div class="sov-card-body" style="display: flex; align-items: center; gap: 10px; padding: 10px 14px;">
    <span style="font-size: 16px;">🏗</span>
    <div>
      <strong style="color: var(--sov-text-primary); font-size: 12px;">Design Preview</strong>
      <p style="color: var(--sov-text-tertiary); font-size: 11px; margin: 2px 0 0;">Governance roles, consent, and audit schemas shown here reflect the target platform design. Enforcement requires backend RBAC integration (Phase 2).</p>
    </div>
  </div>
</div>

<!-- Platform stats -->
<div class="sov-stat-row">
  <div class="sov-stat">
    <span class="sov-stat-value accent">{roles.length}</span>
    <span class="sov-stat-label">RBAC Roles</span>
  </div>
  <div class="sov-stat">
    <span class="sov-stat-value">{auditEventTypes.length}</span>
    <span class="sov-stat-label">Event Types</span>
  </div>
  <div class="sov-stat">
    <span class="sov-stat-value">{consentItems.length}</span>
    <span class="sov-stat-label">Consent Items</span>
  </div>
  <div class="sov-stat">
    <span class="sov-stat-value">{classifications.length}</span>
    <span class="sov-stat-label">Data Classes</span>
  </div>
  <div class="sov-stat">
    <span class="sov-stat-value">{platformVersion}</span>
    <span class="sov-stat-label">Platform Version</span>
  </div>
</div>

<!-- Section tabs -->
<div class="sov-toolbar" style="margin-bottom: 16px;">
  {#each [
    { key: 'rbac', label: 'RBAC', icon: '⛋' },
    { key: 'audit', label: 'Audit Trail', icon: '▤' },
    { key: 'consent', label: 'Consent', icon: '✓' },
    { key: 'classification', label: 'Data Classification', icon: '◉' },
  ] as tab}
    <button class="sov-btn sov-btn-sm"
      class:sov-btn-primary={activeSection === tab.key}
      class:sov-btn-secondary={activeSection !== tab.key}
      on:click={() => activeSection = tab.key}>
      {tab.icon} {tab.label}
    </button>
  {/each}
</div>

{#if activeSection === 'rbac'}
  <!-- RBAC -->
  <div class="gov-grid">
    {#each roles as role}
      <div class="sov-card role-card">
        <div class="sov-card-header">
          <div style="display: flex; align-items: center; gap: 8px;">
            <span class="sov-card-title">{role.name}</span>
            <span class="sov-badge"
              class:sov-badge-accent={role.level === 'clinical'}
              class:sov-badge-warning={role.level === 'system'}
              class:sov-badge-success={role.level === 'research'}
              class:sov-badge-default={role.level === 'compliance'}>
              {role.level}
            </span>
          </div>
        </div>
        <div class="sov-card-body">
          <p class="role-desc">{role.description}</p>
          <div class="role-perms">
            <span class="sov-label">Permissions</span>
            <div class="role-perm-list">
              {#each role.permissions as perm}
                <span class="perm-tag">{perm}</span>
              {/each}
            </div>
          </div>
        </div>
      </div>
    {/each}
  </div>

{:else if activeSection === 'audit'}
  <!-- Audit Event Types -->
  <div class="sov-card">
    <div class="sov-card-header">
      <span class="sov-card-title">Tracked Event Types</span>
      <span class="sov-badge sov-badge-default">{auditEventTypes.length} types</span>
    </div>
    <div class="sov-table-wrap">
      <table class="sov-table">
        <thead>
          <tr>
            <th>Event</th>
            <th>Severity</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          {#each auditEventTypes as evt}
            <tr>
              <td><span class="font-data" style="font-size: 11px; color: var(--sov-accent);">{evt.event}</span></td>
              <td>
                <span class="sov-badge"
                  class:sov-badge-success={evt.severity === 'info'}
                  class:sov-badge-warning={evt.severity === 'warning'}
                  class:sov-badge-error={evt.severity === 'error'}>
                  {evt.severity}
                </span>
              </td>
              <td>{evt.description}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>

  <div class="sov-card" style="margin-top: 16px;">
    <div class="sov-card-header">
      <span class="sov-card-title">Audit Policy</span>
    </div>
    <div class="sov-card-body">
      <div class="policy-grid">
        <div class="policy-item">
          <span class="policy-label">Retention</span>
          <span class="policy-value">Permanent (all events)</span>
        </div>
        <div class="policy-item">
          <span class="policy-label">Immutability</span>
          <span class="policy-value">Append-only, cryptographic chain</span>
        </div>
        <div class="policy-item">
          <span class="policy-label">Timestamps</span>
          <span class="policy-value">UTC, nanosecond precision</span>
        </div>
        <div class="policy-item">
          <span class="policy-label">Attribution</span>
          <span class="policy-value">User ID + session + IP on every event</span>
        </div>
        <div class="policy-item">
          <span class="policy-label">Export</span>
          <span class="policy-value">JSON, CSV, or PDF audit report</span>
        </div>
        <div class="policy-item">
          <span class="policy-label">Alerting</span>
          <span class="policy-value">Real-time on warning/error severity</span>
        </div>
      </div>
    </div>
  </div>

{:else if activeSection === 'consent'}
  <!-- Consent Management -->
  <div class="sov-card">
    <div class="sov-card-header">
      <span class="sov-card-title">Consent Requirements</span>
    </div>
    <div class="sov-card-body">
      <div class="consent-list">
        {#each consentItems as item}
          <div class="consent-item">
            <div class="consent-header">
              <span class="consent-name">{item.label}</span>
              <span class="sov-badge" class:sov-badge-error={item.required} class:sov-badge-default={!item.required}>
                {item.required ? 'Required' : 'Optional'}
              </span>
            </div>
            <p class="consent-desc">{item.description}</p>
            <div class="consent-id font-data">{item.id}</div>
          </div>
        {/each}
      </div>
    </div>
  </div>

  <div class="sov-card" style="margin-top: 16px;">
    <div class="sov-card-header">
      <span class="sov-card-title">Consent Workflow</span>
    </div>
    <div class="sov-card-body">
      <div class="workflow-steps">
        {#each [
          { step: 1, label: 'Collection', desc: 'Patient consent recorded at case creation. Required items block case creation if not obtained.' },
          { step: 2, label: 'Verification', desc: 'Consent records attached to case audit trail. Verified before plan compilation.' },
          { step: 3, label: 'Revocation', desc: 'Patient can revoke consent at any time. Triggers data deletion workflow per retention policy.' },
          { step: 4, label: 'Audit', desc: 'All consent changes logged as audit events. Compliance review via auditor role.' },
        ] as wf}
          <div class="workflow-step">
            <div class="workflow-num">{wf.step}</div>
            <div class="workflow-info">
              <span class="workflow-label">{wf.label}</span>
              <span class="workflow-desc">{wf.desc}</span>
            </div>
          </div>
        {/each}
      </div>
    </div>
  </div>

{:else if activeSection === 'classification'}
  <!-- Data Classification -->
  <div class="class-grid">
    {#each classifications as cls}
      <div class="sov-card class-card">
        <div class="class-header" style="border-left: 3px solid {cls.color};">
          <span class="class-level" style="color: {cls.color};">{cls.level}</span>
          <span class="class-label">{cls.label}</span>
        </div>
        <div class="sov-card-body">
          <p class="class-examples">{cls.examples}</p>
        </div>
      </div>
    {/each}
  </div>

  <div class="sov-card" style="margin-top: 16px;">
    <div class="sov-card-header">
      <span class="sov-card-title">Encryption & Storage</span>
    </div>
    <div class="sov-card-body">
      <div class="policy-grid">
        <div class="policy-item">
          <span class="policy-label">At Rest</span>
          <span class="policy-value">AES-256-GCM (all classification levels)</span>
        </div>
        <div class="policy-item">
          <span class="policy-label">In Transit</span>
          <span class="policy-value">TLS 1.3 minimum</span>
        </div>
        <div class="policy-item">
          <span class="policy-label">PHI Isolation</span>
          <span class="policy-value">Separate encryption keys per patient</span>
        </div>
        <div class="policy-item">
          <span class="policy-label">Key Management</span>
          <span class="policy-value">HSM-backed, 90-day rotation</span>
        </div>
        <div class="policy-item">
          <span class="policy-label">Backup</span>
          <span class="policy-value">Encrypted, geo-redundant, daily</span>
        </div>
        <div class="policy-item">
          <span class="policy-label">Deletion</span>
          <span class="policy-value">Cryptographic erasure + verification</span>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .gov-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }
  .role-card .sov-card-body { display: flex; flex-direction: column; gap: 10px; }
  .role-desc { font-size: 12px; color: var(--sov-text-secondary, #9CA3AF); line-height: 1.5; }
  .role-perms { display: flex; flex-direction: column; gap: 4px; }
  .role-perm-list { display: flex; flex-wrap: wrap; gap: 4px; }
  .perm-tag {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    padding: 2px 6px; background: var(--sov-bg-root, #08080A);
    border: 1px solid var(--sov-border, #1F2937); border-radius: 3px;
    color: var(--sov-accent, #3B82F6);
  }

  .policy-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
  .policy-item { display: flex; flex-direction: column; gap: 2px; }
  .policy-label {
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--sov-text-muted, #4B5563); font-weight: 600;
  }
  .policy-value { font-size: 12px; color: var(--sov-text-secondary, #9CA3AF); }

  .consent-list { display: flex; flex-direction: column; gap: 12px; }
  .consent-item {
    padding: 12px; background: var(--sov-bg-root, #08080A);
    border: 1px solid var(--sov-border, #1F2937); border-radius: 6px;
  }
  .consent-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }
  .consent-name { font-size: 13px; font-weight: 500; color: var(--sov-text-primary, #E8E8EC); }
  .consent-desc { font-size: 12px; color: var(--sov-text-secondary, #9CA3AF); line-height: 1.5; margin: 0; }
  .consent-id { font-size: 10px; color: var(--sov-text-muted, #4B5563); margin-top: 6px; }

  .workflow-steps { display: flex; flex-direction: column; gap: 2px; }
  .workflow-step { display: flex; gap: 12px; padding: 10px 0; border-bottom: 1px solid var(--sov-border-subtle, #151820); }
  .workflow-step:last-child { border-bottom: none; }
  .workflow-num {
    width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600;
    color: var(--sov-accent, #3B82F6); background: var(--sov-accent-dim, #1E3A5F);
    border-radius: 6px; flex-shrink: 0;
  }
  .workflow-info { display: flex; flex-direction: column; gap: 2px; }
  .workflow-label { font-size: 13px; font-weight: 500; color: var(--sov-text-primary, #E8E8EC); }
  .workflow-desc { font-size: 12px; color: var(--sov-text-tertiary, #6B7280); line-height: 1.4; }

  .class-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
  .class-header { padding: 10px 12px; display: flex; flex-direction: column; gap: 2px; }
  .class-level { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; }
  .class-label { font-size: 11px; color: var(--sov-text-secondary, #9CA3AF); }
  .class-examples { font-size: 12px; color: var(--sov-text-tertiary, #6B7280); line-height: 1.5; margin: 0; }
</style>
