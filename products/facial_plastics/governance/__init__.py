"""Governance, audit, and compliance infrastructure.

Submodules:
  audit    – Immutable audit trail and event log
  consent  – Informed consent workflow management
  access   – Role-based access control

Reports:
  reports  – PDF/HTML report generation

Post-operative loop:
  postop/outcome_ingest   – Import actual surgical outcomes
  postop/alignment        – Registration of pre-op prediction to post-op reality
  postop/calibration      – Model parameter calibration from outcomes
  postop/validation       – Prediction accuracy statistics
"""

from .audit import AuditLog, AuditEvent
from .consent import ConsentManager, ConsentRecord
from .access import AccessControl, Role, Permission
from .tenant import (
    CrossTenantAccessError,
    ResourceQuota,
    TenantConfig,
    TenantContext,
    TenantError,
    TenantManager,
    TenantNotFoundError,
    TenantQuotaExceededError,
    TenantRecord,
    TenantStatus,
    TenantSuspendedError,
    TenantTier,
)

__all__ = [
    "AuditLog",
    "AuditEvent",
    "ConsentManager",
    "ConsentRecord",
    "AccessControl",
    "Role",
    "Permission",
    # Tenant
    "CrossTenantAccessError",
    "ResourceQuota",
    "TenantConfig",
    "TenantContext",
    "TenantError",
    "TenantManager",
    "TenantNotFoundError",
    "TenantQuotaExceededError",
    "TenantRecord",
    "TenantStatus",
    "TenantSuspendedError",
    "TenantTier",
]
