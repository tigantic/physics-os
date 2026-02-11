"""Tests for multi-tenant isolation infrastructure."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Generator, List

import pytest

from products.facial_plastics.governance.tenant import (
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


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    d = Path(tempfile.mkdtemp(prefix="fp_tenant_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def manager(tmp_dir: Path) -> TenantManager:
    return TenantManager(base_path=tmp_dir)


# ── ResourceQuota tests ──────────────────────────────────────────

class TestResourceQuota:
    def test_defaults(self) -> None:
        q = ResourceQuota()
        assert q.max_cases == 100
        assert q.max_storage_mb == 10_000
        assert q.max_concurrent_sims == 4
        assert q.max_users == 20
        assert q.max_plans_per_case == 50

    def test_round_trip(self) -> None:
        q = ResourceQuota(max_cases=50, max_storage_mb=5000)
        d = q.to_dict()
        q2 = ResourceQuota.from_dict(d)
        assert q2.max_cases == 50
        assert q2.max_storage_mb == 5000


# ── TenantConfig tests ──────────────────────────────────────────

class TestTenantConfig:
    def test_defaults(self) -> None:
        c = TenantConfig()
        assert c.solver_timeout_seconds == 300.0
        assert c.enable_cfd is True
        assert c.report_branding is None

    def test_round_trip(self) -> None:
        c = TenantConfig(
            solver_timeout_seconds=120.0,
            enable_fsi=False,
            report_branding={"institution_name": "Test Clinic"},
        )
        d = c.to_dict()
        c2 = TenantConfig.from_dict(d)
        assert c2.solver_timeout_seconds == 120.0
        assert c2.enable_fsi is False
        assert c2.report_branding is not None
        assert c2.report_branding["institution_name"] == "Test Clinic"


# ── TenantRecord tests ──────────────────────────────────────────

class TestTenantRecord:
    def test_data_directory_name(self) -> None:
        r = TenantRecord(tenant_id="clinic/abc", display_name="Clinic ABC")
        assert "/" not in r.data_directory_name
        assert r.data_directory_name == "tenant_clinic_abc"

    def test_is_operational(self) -> None:
        r = TenantRecord(tenant_id="t1", display_name="T1", status=TenantStatus.ACTIVE)
        assert r.is_operational() is True
        r.status = TenantStatus.SUSPENDED
        assert r.is_operational() is False

    def test_check_case_quota(self) -> None:
        r = TenantRecord(
            tenant_id="t1", display_name="T1",
            quota=ResourceQuota(max_cases=5),
            case_count=4,
        )
        assert r.check_case_quota() is True
        r.case_count = 5
        assert r.check_case_quota() is False

    def test_check_storage_quota(self) -> None:
        r = TenantRecord(
            tenant_id="t1", display_name="T1",
            quota=ResourceQuota(max_storage_mb=100),
            storage_used_mb=90.0,
        )
        assert r.check_storage_quota(additional_mb=10.0) is True
        assert r.check_storage_quota(additional_mb=11.0) is False

    def test_check_user_quota(self) -> None:
        r = TenantRecord(
            tenant_id="t1", display_name="T1",
            quota=ResourceQuota(max_users=2),
            user_ids={"u1"},
        )
        assert r.check_user_quota() is True
        r.user_ids.add("u2")
        assert r.check_user_quota() is False

    def test_round_trip(self) -> None:
        r = TenantRecord(
            tenant_id="t1",
            display_name="Test",
            status=TenantStatus.ACTIVE,
            tier=TenantTier.ENTERPRISE,
            quota=ResourceQuota(max_cases=9999),
            config=TenantConfig(enable_cfd=False),
            owner_user_id="admin",
            user_ids={"admin", "doc1"},
            case_count=42,
            storage_used_mb=123.45,
        )
        d = r.to_dict()
        r2 = TenantRecord.from_dict(d)
        assert r2.tenant_id == "t1"
        assert r2.tier == TenantTier.ENTERPRISE
        assert r2.quota.max_cases == 9999
        assert r2.config.enable_cfd is False
        assert r2.case_count == 42
        assert "admin" in r2.user_ids


# ── TenantContext tests ──────────────────────────────────────────

class TestTenantContext:
    def test_no_active_context_raises(self) -> None:
        # Ensure clean state
        TenantContext._clear()
        with pytest.raises(TenantError, match="No active tenant context"):
            TenantContext.current_tenant_id()

    def test_is_active(self) -> None:
        TenantContext._clear()
        assert TenantContext.is_active() is False
        rec = TenantRecord(tenant_id="t1", display_name="T1")
        TenantContext._set(rec)
        assert TenantContext.is_active() is True
        assert TenantContext.current_tenant_id() == "t1"
        TenantContext._clear()
        assert TenantContext.is_active() is False

    def test_context_provides_config(self) -> None:
        rec = TenantRecord(
            tenant_id="t1", display_name="T1",
            config=TenantConfig(solver_timeout_seconds=60.0),
        )
        TenantContext._set(rec)
        cfg = TenantContext.current_config()
        assert cfg.solver_timeout_seconds == 60.0
        TenantContext._clear()


# ── TenantManager CRUD tests ────────────────────────────────────

class TestTenantManagerCRUD:
    def test_create_and_get(self, manager: TenantManager) -> None:
        rec = manager.create_tenant("clinic_1", "Clinic One", owner_user_id="u1")
        assert rec.tenant_id == "clinic_1"
        assert rec.display_name == "Clinic One"
        assert "u1" in rec.user_ids
        assert rec.status == TenantStatus.ACTIVE

        fetched = manager.get_tenant("clinic_1")
        assert fetched.tenant_id == "clinic_1"

    def test_create_duplicate_raises(self, manager: TenantManager) -> None:
        manager.create_tenant("t1", "T1")
        with pytest.raises(TenantError, match="already exists"):
            manager.create_tenant("t1", "T1 again")

    def test_get_nonexistent_raises(self, manager: TenantManager) -> None:
        with pytest.raises(TenantNotFoundError):
            manager.get_tenant("nonexistent")

    def test_creates_data_directories(self, manager: TenantManager, tmp_dir: Path) -> None:
        manager.create_tenant("t1", "T1")
        data_dir = tmp_dir / "tenant_t1"
        assert data_dir.is_dir()
        for sub in ("cases", "audit", "models", "exports", "backups"):
            assert (data_dir / sub).is_dir()

    def test_list_tenants(self, manager: TenantManager) -> None:
        manager.create_tenant("a", "A")
        manager.create_tenant("b", "B", tier=TenantTier.ENTERPRISE)
        manager.create_tenant("c", "C")

        all_tenants = manager.list_tenants()
        assert len(all_tenants) == 3

        enterprise = manager.list_tenants(tier=TenantTier.ENTERPRISE)
        assert len(enterprise) == 1
        assert enterprise[0].tenant_id == "b"

    def test_update_tenant(self, manager: TenantManager) -> None:
        manager.create_tenant("t1", "Old Name")
        updated = manager.update_tenant(
            "t1",
            display_name="New Name",
            tier=TenantTier.ENTERPRISE,
        )
        assert updated.display_name == "New Name"
        assert updated.tier == TenantTier.ENTERPRISE

    def test_update_nonexistent_raises(self, manager: TenantManager) -> None:
        with pytest.raises(TenantNotFoundError):
            manager.update_tenant("nope", display_name="X")


# ── Lifecycle tests ──────────────────────────────────────────────

class TestTenantLifecycle:
    def test_suspend_and_reactivate(self, manager: TenantManager) -> None:
        manager.create_tenant("t1", "T1")
        manager.suspend_tenant("t1", reason="billing")
        rec = manager.get_tenant("t1")
        assert rec.status == TenantStatus.SUSPENDED

        manager.reactivate_tenant("t1")
        rec = manager.get_tenant("t1")
        assert rec.status == TenantStatus.ACTIVE

    def test_reactivate_active_raises(self, manager: TenantManager) -> None:
        manager.create_tenant("t1", "T1")
        with pytest.raises(TenantError, match="Cannot reactivate"):
            manager.reactivate_tenant("t1")

    def test_delete_soft(self, manager: TenantManager) -> None:
        manager.create_tenant("t1", "T1")
        manager.delete_tenant("t1", purge_data=False)
        rec = manager.get_tenant("t1")
        assert rec.status == TenantStatus.PENDING_DELETION

    def test_delete_purge(self, manager: TenantManager, tmp_dir: Path) -> None:
        manager.create_tenant("t1", "T1")
        data_dir = tmp_dir / "tenant_t1"
        assert data_dir.exists()

        manager.delete_tenant("t1", purge_data=True)
        assert not data_dir.exists()
        with pytest.raises(TenantNotFoundError):
            manager.get_tenant("t1")


# ── User management tests ────────────────────────────────────────

class TestTenantUserManagement:
    def test_add_and_remove_user(self, manager: TenantManager) -> None:
        manager.create_tenant("t1", "T1")
        manager.add_user_to_tenant("t1", "doc1")
        assert manager.user_belongs_to_tenant("doc1", "t1")

        manager.remove_user_from_tenant("t1", "doc1")
        assert not manager.user_belongs_to_tenant("doc1", "t1")

    def test_user_quota_enforcement(self, manager: TenantManager) -> None:
        manager.create_tenant(
            "t1", "T1",
            quota=ResourceQuota(max_users=2),
        )
        manager.add_user_to_tenant("t1", "u1")
        manager.add_user_to_tenant("t1", "u2")
        with pytest.raises(TenantQuotaExceededError, match="user quota"):
            manager.add_user_to_tenant("t1", "u3")

    def test_get_user_tenants(self, manager: TenantManager) -> None:
        manager.create_tenant("t1", "T1")
        manager.create_tenant("t2", "T2")
        manager.add_user_to_tenant("t1", "doc1")
        manager.add_user_to_tenant("t2", "doc1")

        tenants = manager.get_user_tenants("doc1")
        assert len(tenants) == 2


# ── Context activation tests ────────────────────────────────────

class TestTenantActivation:
    def test_activate_context(self, manager: TenantManager) -> None:
        manager.create_tenant("t1", "T1")
        with manager.activate("t1") as tenant:
            assert TenantContext.current_tenant_id() == "t1"
            assert TenantContext.current_record().display_name == "T1"
        # Context should be cleared after exit
        assert not TenantContext.is_active()

    def test_activate_suspended_raises(self, manager: TenantManager) -> None:
        manager.create_tenant("t1", "T1")
        manager.suspend_tenant("t1")
        with pytest.raises(TenantSuspendedError):
            with manager.activate("t1"):
                pass

    def test_activate_nonexistent_raises(self, manager: TenantManager) -> None:
        with pytest.raises(TenantNotFoundError):
            with manager.activate("nope"):
                pass

    def test_nested_activation_restores(self, manager: TenantManager) -> None:
        manager.create_tenant("t1", "T1")
        manager.create_tenant("t2", "T2")
        with manager.activate("t1"):
            assert TenantContext.current_tenant_id() == "t1"
            with manager.activate("t2"):
                assert TenantContext.current_tenant_id() == "t2"
            # Should restore t1
            assert TenantContext.current_tenant_id() == "t1"
        assert not TenantContext.is_active()

    def test_activation_thread_isolation(self, manager: TenantManager) -> None:
        """Each thread should have independent tenant context."""
        manager.create_tenant("t1", "T1")
        manager.create_tenant("t2", "T2")

        results: Dict[str, str] = {}

        def worker(tid: str) -> None:
            with manager.activate(tid):
                results[threading.current_thread().name] = TenantContext.current_tenant_id()

        th1 = threading.Thread(target=worker, args=("t1",), name="worker_1")
        th2 = threading.Thread(target=worker, args=("t2",), name="worker_2")
        th1.start()
        th2.start()
        th1.join()
        th2.join()

        assert results["worker_1"] == "t1"
        assert results["worker_2"] == "t2"


# ── Quota enforcement tests ──────────────────────────────────────

class TestQuotaEnforcement:
    def test_case_creation_quota(self, manager: TenantManager) -> None:
        manager.create_tenant(
            "t1", "T1",
            quota=ResourceQuota(max_cases=2),
        )
        # Record 2 cases
        manager.record_case_created("t1")
        manager.record_case_created("t1")

        with pytest.raises(TenantQuotaExceededError, match="case quota"):
            manager.check_case_creation("t1")

    def test_case_deletion_restores_quota(self, manager: TenantManager) -> None:
        manager.create_tenant(
            "t1", "T1",
            quota=ResourceQuota(max_cases=2),
        )
        manager.record_case_created("t1")
        manager.record_case_created("t1")
        manager.record_case_deleted("t1")

        # Should not raise now
        manager.check_case_creation("t1")

    def test_check_case_creation_suspended(self, manager: TenantManager) -> None:
        manager.create_tenant("t1", "T1")
        manager.suspend_tenant("t1")
        with pytest.raises(TenantSuspendedError):
            manager.check_case_creation("t1")

    def test_storage_usage_scan(self, manager: TenantManager, tmp_dir: Path) -> None:
        manager.create_tenant("t1", "T1")
        # Write a small file
        cases_dir = tmp_dir / "tenant_t1" / "cases"
        (cases_dir / "test.json").write_text('{"data": "test"}')

        mb = manager.update_storage_usage("t1")
        assert mb > 0.0
        rec = manager.get_tenant("t1")
        assert rec.storage_used_mb > 0.0


# ── Cross-tenant guard tests ────────────────────────────────────

class TestCrossTenantGuard:
    def test_same_tenant_ok(self, manager: TenantManager) -> None:
        manager.create_tenant("t1", "T1")
        with manager.activate("t1"):
            TenantManager.assert_same_tenant("t1")  # should not raise

    def test_cross_tenant_raises(self, manager: TenantManager) -> None:
        manager.create_tenant("t1", "T1")
        with manager.activate("t1"):
            with pytest.raises(CrossTenantAccessError):
                TenantManager.assert_same_tenant("t2")

    def test_no_context_raises(self) -> None:
        TenantContext._clear()
        with pytest.raises(TenantError, match="No active tenant context"):
            TenantManager.assert_same_tenant("t1")


# ── Path helper tests ────────────────────────────────────────────

class TestTenantPaths:
    def test_data_root(self, manager: TenantManager, tmp_dir: Path) -> None:
        manager.create_tenant("t1", "T1")
        root = manager.tenant_data_root("t1")
        assert root == tmp_dir / "tenant_t1"

    def test_cases_dir(self, manager: TenantManager, tmp_dir: Path) -> None:
        manager.create_tenant("t1", "T1")
        assert manager.tenant_cases_dir("t1") == tmp_dir / "tenant_t1" / "cases"

    def test_audit_dir(self, manager: TenantManager, tmp_dir: Path) -> None:
        manager.create_tenant("t1", "T1")
        assert manager.tenant_audit_dir("t1") == tmp_dir / "tenant_t1" / "audit"

    def test_exports_dir(self, manager: TenantManager, tmp_dir: Path) -> None:
        manager.create_tenant("t1", "T1")
        assert manager.tenant_exports_dir("t1") == tmp_dir / "tenant_t1" / "exports"


# ── Persistence tests ────────────────────────────────────────────

class TestTenantPersistence:
    def test_save_and_reload(self, tmp_dir: Path) -> None:
        """Tenant data should survive manager reinstantiation."""
        m1 = TenantManager(base_path=tmp_dir)
        m1.create_tenant("t1", "Clinic A", tier=TenantTier.ENTERPRISE)
        m1.add_user_to_tenant("t1", "doc1")
        m1.record_case_created("t1")

        # Reload from disk
        m2 = TenantManager(base_path=tmp_dir)
        rec = m2.get_tenant("t1")
        assert rec.display_name == "Clinic A"
        assert rec.tier == TenantTier.ENTERPRISE
        assert "doc1" in rec.user_ids
        assert rec.case_count == 1

    def test_registry_file_format(self, manager: TenantManager, tmp_dir: Path) -> None:
        manager.create_tenant("t1", "T1")
        meta = tmp_dir / "_tenant_registry.json"
        assert meta.exists()
        data = json.loads(meta.read_text())
        assert "version" in data
        assert "tenants" in data
        assert "t1" in data["tenants"]


# ── Tier defaults tests ──────────────────────────────────────────

class TestTierDefaults:
    def test_free_tier_limits(self, manager: TenantManager) -> None:
        rec = manager.create_tenant("t1", "T1", tier=TenantTier.FREE)
        assert rec.quota.max_cases == 10
        assert rec.quota.max_storage_mb == 500
        assert rec.quota.max_concurrent_sims == 1

    def test_standard_tier_limits(self, manager: TenantManager) -> None:
        rec = manager.create_tenant("t1", "T1", tier=TenantTier.STANDARD)
        assert rec.quota.max_cases == 100
        assert rec.quota.max_concurrent_sims == 4

    def test_enterprise_tier_limits(self, manager: TenantManager) -> None:
        rec = manager.create_tenant("t1", "T1", tier=TenantTier.ENTERPRISE)
        assert rec.quota.max_cases == 10_000
        assert rec.quota.max_concurrent_sims == 32
        assert rec.quota.max_users == 500

    def test_custom_quota_overrides_tier(self, manager: TenantManager) -> None:
        custom = ResourceQuota(max_cases=42)
        rec = manager.create_tenant("t1", "T1", tier=TenantTier.FREE, quota=custom)
        assert rec.quota.max_cases == 42  # custom wins
