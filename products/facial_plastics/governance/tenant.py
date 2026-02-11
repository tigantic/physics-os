"""Multi-tenant isolation layer for the facial plastics platform.

Provides per-tenant data isolation, configuration scoping, and
resource quotas so that a single deployment can safely serve
multiple clinical institutions without data leakage.

Architecture:
  - Each tenant has an isolated ``data_root`` on disk (case bundles,
    audit logs, model weights).
  - ``TenantContext`` is a thread-local / async-context-safe handle
    that all downstream modules consult for the active tenant.
  - ``TenantManager`` is the CRUD layer that provisions, suspends,
    and destroys tenant namespaces.
  - Guards enforce that cross-tenant data access is impossible
    unless the caller has explicit ``CROSS_TENANT_READ`` permission.

No external dependencies — stdlib + dataclasses only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Set

logger = logging.getLogger(__name__)


# ── Enums and constants ──────────────────────────────────────────

class TenantStatus(Enum):
    """Lifecycle states for a tenant."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING_DELETION = "pending_deletion"
    DELETED = "deleted"


class TenantTier(Enum):
    """Service-tier differentiation."""
    FREE = "free"
    STANDARD = "standard"
    ENTERPRISE = "enterprise"


# Default resource quotas per tier
_DEFAULT_QUOTAS: Dict[TenantTier, "ResourceQuota"] = {}  # populated after class def


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class ResourceQuota:
    """Resource limits for a tenant."""
    max_cases: int = 100
    max_storage_mb: int = 10_000
    max_concurrent_sims: int = 4
    max_users: int = 20
    max_plans_per_case: int = 50

    def to_dict(self) -> Dict[str, int]:
        return {
            "max_cases": self.max_cases,
            "max_storage_mb": self.max_storage_mb,
            "max_concurrent_sims": self.max_concurrent_sims,
            "max_users": self.max_users,
            "max_plans_per_case": self.max_plans_per_case,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ResourceQuota:
        return cls(
            max_cases=int(d.get("max_cases", 100)),
            max_storage_mb=int(d.get("max_storage_mb", 10_000)),
            max_concurrent_sims=int(d.get("max_concurrent_sims", 4)),
            max_users=int(d.get("max_users", 20)),
            max_plans_per_case=int(d.get("max_plans_per_case", 50)),
        )


# Populate default quotas
_DEFAULT_QUOTAS[TenantTier.FREE] = ResourceQuota(
    max_cases=10, max_storage_mb=500, max_concurrent_sims=1,
    max_users=3, max_plans_per_case=10,
)
_DEFAULT_QUOTAS[TenantTier.STANDARD] = ResourceQuota(
    max_cases=100, max_storage_mb=10_000, max_concurrent_sims=4,
    max_users=20, max_plans_per_case=50,
)
_DEFAULT_QUOTAS[TenantTier.ENTERPRISE] = ResourceQuota(
    max_cases=10_000, max_storage_mb=500_000, max_concurrent_sims=32,
    max_users=500, max_plans_per_case=200,
)


@dataclass
class TenantConfig:
    """Per-tenant configuration overrides.

    These overlay the global ``PlatformConfig`` for everything
    within the tenant boundary.
    """
    solver_timeout_seconds: float = 300.0
    max_mesh_elements: int = 500_000
    enable_cfd: bool = True
    enable_fsi: bool = True
    enable_aging: bool = True
    custom_tissue_library: Optional[Dict[str, Any]] = None
    default_population_seed: int = 42
    report_branding: Optional[Dict[str, str]] = None
    """Keys: logo_url, institution_name, disclaimer."""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "solver_timeout_seconds": self.solver_timeout_seconds,
            "max_mesh_elements": self.max_mesh_elements,
            "enable_cfd": self.enable_cfd,
            "enable_fsi": self.enable_fsi,
            "enable_aging": self.enable_aging,
            "default_population_seed": self.default_population_seed,
        }
        if self.custom_tissue_library is not None:
            d["custom_tissue_library"] = self.custom_tissue_library
        if self.report_branding is not None:
            d["report_branding"] = self.report_branding
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TenantConfig:
        return cls(
            solver_timeout_seconds=float(d.get("solver_timeout_seconds", 300.0)),
            max_mesh_elements=int(d.get("max_mesh_elements", 500_000)),
            enable_cfd=bool(d.get("enable_cfd", True)),
            enable_fsi=bool(d.get("enable_fsi", True)),
            enable_aging=bool(d.get("enable_aging", True)),
            custom_tissue_library=d.get("custom_tissue_library"),
            default_population_seed=int(d.get("default_population_seed", 42)),
            report_branding=d.get("report_branding"),
        )


@dataclass
class TenantRecord:
    """Persistent record for a single tenant."""
    tenant_id: str
    display_name: str
    status: TenantStatus = TenantStatus.ACTIVE
    tier: TenantTier = TenantTier.STANDARD
    quota: ResourceQuota = field(default_factory=ResourceQuota)
    config: TenantConfig = field(default_factory=TenantConfig)
    owner_user_id: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    user_ids: Set[str] = field(default_factory=set)
    """Users affiliated with this tenant."""

    case_count: int = 0
    storage_used_mb: float = 0.0

    @property
    def data_directory_name(self) -> str:
        """Deterministic and filesystem-safe directory name."""
        safe = self.tenant_id.replace("/", "_").replace("\\", "_")
        return f"tenant_{safe}"

    def is_operational(self) -> bool:
        return self.status == TenantStatus.ACTIVE

    def check_case_quota(self) -> bool:
        return self.case_count < self.quota.max_cases

    def check_storage_quota(self, additional_mb: float = 0.0) -> bool:
        return (self.storage_used_mb + additional_mb) <= self.quota.max_storage_mb

    def check_user_quota(self) -> bool:
        return len(self.user_ids) < self.quota.max_users

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "display_name": self.display_name,
            "status": self.status.value,
            "tier": self.tier.value,
            "quota": self.quota.to_dict(),
            "config": self.config.to_dict(),
            "owner_user_id": self.owner_user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "user_ids": sorted(self.user_ids),
            "case_count": self.case_count,
            "storage_used_mb": self.storage_used_mb,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TenantRecord:
        return cls(
            tenant_id=d["tenant_id"],
            display_name=d["display_name"],
            status=TenantStatus(d.get("status", "active")),
            tier=TenantTier(d.get("tier", "standard")),
            quota=ResourceQuota.from_dict(d.get("quota", {})),
            config=TenantConfig.from_dict(d.get("config", {})),
            owner_user_id=d.get("owner_user_id", ""),
            created_at=float(d.get("created_at", 0.0)),
            updated_at=float(d.get("updated_at", 0.0)),
            user_ids=set(d.get("user_ids", [])),
            case_count=int(d.get("case_count", 0)),
            storage_used_mb=float(d.get("storage_used_mb", 0.0)),
        )


# ── Thread-local tenant context ──────────────────────────────────

class _TenantLocal(threading.local):
    """Thread-local storage for the active tenant."""
    tenant_id: Optional[str] = None
    tenant_record: Optional[TenantRecord] = None


_thread_ctx = _TenantLocal()


class TenantContext:
    """Access the currently-active tenant from any module.

    Usage::

        with tenant_manager.activate("clinic_123"):
            tid = TenantContext.current_tenant_id()        # "clinic_123"
            root = TenantContext.current_data_root()        # Path(...)
            cfg = TenantContext.current_config()             # TenantConfig

    Raises ``TenantError`` if accessed outside an active context.
    """

    @staticmethod
    def current_tenant_id() -> str:
        tid = _thread_ctx.tenant_id
        if tid is None:
            raise TenantError("No active tenant context")
        return tid

    @staticmethod
    def current_record() -> TenantRecord:
        rec = _thread_ctx.tenant_record
        if rec is None:
            raise TenantError("No active tenant context")
        return rec

    @staticmethod
    def current_config() -> TenantConfig:
        return TenantContext.current_record().config

    @staticmethod
    def is_active() -> bool:
        return _thread_ctx.tenant_id is not None

    @staticmethod
    def _set(record: TenantRecord) -> None:
        _thread_ctx.tenant_id = record.tenant_id
        _thread_ctx.tenant_record = record

    @staticmethod
    def _clear() -> None:
        _thread_ctx.tenant_id = None
        _thread_ctx.tenant_record = None


# ── Errors ───────────────────────────────────────────────────────

class TenantError(Exception):
    """Base exception for tenant operations."""


class TenantNotFoundError(TenantError):
    """Raised when referencing a non-existent tenant."""


class TenantQuotaExceededError(TenantError):
    """Raised when a quota limit is reached."""


class TenantSuspendedError(TenantError):
    """Raised when operating on a suspended tenant."""


class CrossTenantAccessError(TenantError):
    """Raised on unauthorized cross-tenant data access."""


# ── Tenant Manager ───────────────────────────────────────────────

class TenantManager:
    """CRUD and lifecycle manager for tenants.

    Provisions isolated directories, persists tenant metadata to
    JSON, and provides ``activate()`` context manager that sets the
    thread-local ``TenantContext``.

    Args:
        base_path: Root directory under which per-tenant data lives.
    """

    def __init__(self, base_path: Path) -> None:
        self._base = base_path
        self._tenants: Dict[str, TenantRecord] = {}
        self._lock = threading.Lock()
        self._meta_file = base_path / "_tenant_registry.json"

        base_path.mkdir(parents=True, exist_ok=True)
        if self._meta_file.exists():
            self._load()

    # ── CRUD ─────────────────────────────────────────────────────

    def create_tenant(
        self,
        tenant_id: str,
        display_name: str,
        *,
        tier: TenantTier = TenantTier.STANDARD,
        owner_user_id: str = "",
        config: Optional[TenantConfig] = None,
        quota: Optional[ResourceQuota] = None,
    ) -> TenantRecord:
        """Provision a new tenant with isolated data directory."""
        with self._lock:
            if tenant_id in self._tenants:
                raise TenantError(f"Tenant '{tenant_id}' already exists")

            now = time.time()
            record = TenantRecord(
                tenant_id=tenant_id,
                display_name=display_name,
                status=TenantStatus.ACTIVE,
                tier=tier,
                quota=quota or _DEFAULT_QUOTAS.get(tier, ResourceQuota()),
                config=config or TenantConfig(),
                owner_user_id=owner_user_id,
                created_at=now,
                updated_at=now,
                user_ids={owner_user_id} if owner_user_id else set(),
            )

            # Create isolated directory structure
            data_dir = self._tenant_path(record)
            data_dir.mkdir(parents=True, exist_ok=True)
            for sub in ("cases", "audit", "models", "exports", "backups"):
                (data_dir / sub).mkdir(exist_ok=True)

            self._tenants[tenant_id] = record
            self._save()

            logger.info("Tenant created: %s (%s)", tenant_id, display_name)
            return record

    def get_tenant(self, tenant_id: str) -> TenantRecord:
        """Retrieve tenant record or raise."""
        with self._lock:
            rec = self._tenants.get(tenant_id)
            if rec is None:
                raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")
            return rec

    def list_tenants(
        self,
        *,
        status: Optional[TenantStatus] = None,
        tier: Optional[TenantTier] = None,
    ) -> List[TenantRecord]:
        """List tenants with optional filters."""
        with self._lock:
            result = list(self._tenants.values())
            if status is not None:
                result = [r for r in result if r.status == status]
            if tier is not None:
                result = [r for r in result if r.tier == tier]
            return sorted(result, key=lambda r: r.created_at)

    def update_tenant(
        self,
        tenant_id: str,
        *,
        display_name: Optional[str] = None,
        tier: Optional[TenantTier] = None,
        config: Optional[TenantConfig] = None,
        quota: Optional[ResourceQuota] = None,
    ) -> TenantRecord:
        """Update mutable fields on a tenant."""
        with self._lock:
            rec = self._tenants.get(tenant_id)
            if rec is None:
                raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")

            if display_name is not None:
                rec.display_name = display_name
            if tier is not None:
                rec.tier = tier
            if config is not None:
                rec.config = config
            if quota is not None:
                rec.quota = quota
            rec.updated_at = time.time()

            self._save()
            return rec

    def suspend_tenant(self, tenant_id: str, reason: str = "") -> TenantRecord:
        """Suspend a tenant — data preserved, access denied."""
        with self._lock:
            rec = self._tenants.get(tenant_id)
            if rec is None:
                raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")
            rec.status = TenantStatus.SUSPENDED
            rec.updated_at = time.time()
            self._save()
            logger.warning("Tenant suspended: %s — %s", tenant_id, reason)
            return rec

    def reactivate_tenant(self, tenant_id: str) -> TenantRecord:
        """Reactivate a suspended tenant."""
        with self._lock:
            rec = self._tenants.get(tenant_id)
            if rec is None:
                raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")
            if rec.status not in (TenantStatus.SUSPENDED, TenantStatus.PENDING_DELETION):
                raise TenantError(
                    f"Cannot reactivate tenant in state '{rec.status.value}'"
                )
            rec.status = TenantStatus.ACTIVE
            rec.updated_at = time.time()
            self._save()
            logger.info("Tenant reactivated: %s", tenant_id)
            return rec

    def delete_tenant(
        self,
        tenant_id: str,
        *,
        purge_data: bool = False,
    ) -> None:
        """Mark tenant for deletion, optionally purge data immediately."""
        with self._lock:
            rec = self._tenants.get(tenant_id)
            if rec is None:
                raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")

            if purge_data:
                data_dir = self._tenant_path(rec)
                if data_dir.exists():
                    shutil.rmtree(data_dir)
                rec.status = TenantStatus.DELETED
                del self._tenants[tenant_id]
                logger.warning("Tenant purged: %s", tenant_id)
            else:
                rec.status = TenantStatus.PENDING_DELETION
                rec.updated_at = time.time()
                logger.info("Tenant marked for deletion: %s", tenant_id)

            self._save()

    # ── User management ──────────────────────────────────────────

    def add_user_to_tenant(self, tenant_id: str, user_id: str) -> None:
        """Associate a user with a tenant."""
        with self._lock:
            rec = self._tenants.get(tenant_id)
            if rec is None:
                raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")
            if not rec.check_user_quota():
                raise TenantQuotaExceededError(
                    f"Tenant '{tenant_id}' user quota exceeded "
                    f"({len(rec.user_ids)}/{rec.quota.max_users})"
                )
            rec.user_ids.add(user_id)
            rec.updated_at = time.time()
            self._save()

    def remove_user_from_tenant(self, tenant_id: str, user_id: str) -> None:
        """Remove a user from a tenant."""
        with self._lock:
            rec = self._tenants.get(tenant_id)
            if rec is None:
                raise TenantNotFoundError(f"Tenant '{tenant_id}' not found")
            rec.user_ids.discard(user_id)
            rec.updated_at = time.time()
            self._save()

    def user_belongs_to_tenant(self, user_id: str, tenant_id: str) -> bool:
        """Check if a user is affiliated with a tenant."""
        with self._lock:
            rec = self._tenants.get(tenant_id)
            if rec is None:
                return False
            return user_id in rec.user_ids

    def get_user_tenants(self, user_id: str) -> List[TenantRecord]:
        """Return all tenants a user belongs to."""
        with self._lock:
            return [
                rec for rec in self._tenants.values()
                if user_id in rec.user_ids
            ]

    # ── Context activation ───────────────────────────────────────

    @contextmanager
    def activate(self, tenant_id: str) -> Generator[TenantRecord, None, None]:
        """Set the active tenant for the current thread.

        Usage::

            with manager.activate("clinic_123") as tenant:
                # All modules see tenant_id = "clinic_123"
                ...

        Raises:
            TenantNotFoundError: If tenant does not exist.
            TenantSuspendedError: If tenant is not in ACTIVE state.
        """
        rec = self.get_tenant(tenant_id)
        if not rec.is_operational():
            raise TenantSuspendedError(
                f"Tenant '{tenant_id}' is {rec.status.value}, not operational"
            )
        previous_id = _thread_ctx.tenant_id
        previous_rec = _thread_ctx.tenant_record
        TenantContext._set(rec)
        try:
            yield rec
        finally:
            _thread_ctx.tenant_id = previous_id
            _thread_ctx.tenant_record = previous_rec

    # ── Data path helpers ────────────────────────────────────────

    def tenant_data_root(self, tenant_id: str) -> Path:
        """Absolute path to a tenant's isolated data directory."""
        rec = self.get_tenant(tenant_id)
        return self._tenant_path(rec)

    def tenant_cases_dir(self, tenant_id: str) -> Path:
        return self.tenant_data_root(tenant_id) / "cases"

    def tenant_audit_dir(self, tenant_id: str) -> Path:
        return self.tenant_data_root(tenant_id) / "audit"

    def tenant_exports_dir(self, tenant_id: str) -> Path:
        return self.tenant_data_root(tenant_id) / "exports"

    # ── Quota enforcement ────────────────────────────────────────

    def check_case_creation(self, tenant_id: str) -> None:
        """Raise if the tenant cannot create another case."""
        rec = self.get_tenant(tenant_id)
        if not rec.is_operational():
            raise TenantSuspendedError(
                f"Tenant '{tenant_id}' is {rec.status.value}"
            )
        if not rec.check_case_quota():
            raise TenantQuotaExceededError(
                f"Tenant '{tenant_id}' case quota exceeded "
                f"({rec.case_count}/{rec.quota.max_cases})"
            )

    def record_case_created(self, tenant_id: str) -> None:
        """Increment case count after successful creation."""
        with self._lock:
            rec = self._tenants.get(tenant_id)
            if rec is not None:
                rec.case_count += 1
                rec.updated_at = time.time()
                self._save()

    def record_case_deleted(self, tenant_id: str) -> None:
        """Decrement case count after deletion."""
        with self._lock:
            rec = self._tenants.get(tenant_id)
            if rec is not None:
                rec.case_count = max(0, rec.case_count - 1)
                rec.updated_at = time.time()
                self._save()

    def update_storage_usage(self, tenant_id: str) -> float:
        """Scan tenant directory and update storage_used_mb."""
        rec = self.get_tenant(tenant_id)
        data_dir = self._tenant_path(rec)
        total_bytes = 0
        if data_dir.exists():
            for f in data_dir.rglob("*"):
                if f.is_file():
                    total_bytes += f.stat().st_size
        mb = total_bytes / (1024 * 1024)
        with self._lock:
            rec.storage_used_mb = round(mb, 6)
            rec.updated_at = time.time()
            self._save()
        return mb

    # ── Cross-tenant guard ───────────────────────────────────────

    @staticmethod
    def assert_same_tenant(expected_tenant_id: str) -> None:
        """Verify the current context matches the expected tenant.

        Call this in any data-access path to prevent cross-tenant leakage.
        """
        if not TenantContext.is_active():
            raise TenantError("No active tenant context — cannot verify isolation")
        current = TenantContext.current_tenant_id()
        if current != expected_tenant_id:
            raise CrossTenantAccessError(
                f"Cross-tenant access denied: active='{current}', "
                f"requested='{expected_tenant_id}'"
            )

    # ── Persistence ──────────────────────────────────────────────

    def _tenant_path(self, rec: TenantRecord) -> Path:
        return self._base / rec.data_directory_name

    def _save(self) -> None:
        data = {
            "version": 1,
            "tenants": {
                tid: rec.to_dict()
                for tid, rec in self._tenants.items()
            },
        }
        tmp = self._meta_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(self._meta_file)

    def _load(self) -> None:
        try:
            raw = json.loads(self._meta_file.read_text())
            for tid, d in raw.get("tenants", {}).items():
                self._tenants[tid] = TenantRecord.from_dict(d)
            logger.info("Loaded %d tenants from %s", len(self._tenants), self._meta_file)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error("Failed to load tenant registry: %s", exc)
