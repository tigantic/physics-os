"""Tests for governance sub-package: audit, consent, access control."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from products.facial_plastics.governance.audit import (
    AuditEvent,
    AuditLog,
    EventType,
)
from products.facial_plastics.governance.consent import (
    ConsentManager,
    ConsentRecord,
    ConsentScope,
    ConsentStatus,
    STANDARD_DISCLAIMERS,
)
from products.facial_plastics.governance.access import (
    AccessControl,
    AccessDecision,
    Permission,
    Role,
    ROLE_PERMISSIONS,
    UserProfile,
)


# ── Audit Log ────────────────────────────────────────────────────

class TestAuditEvent:
    """Test audit event construction and verification."""

    def test_event_creation(self):
        event = AuditEvent(
            event_type=EventType.CASE_CREATE,
            timestamp_utc=time.time(),
            user_id="user001",
            case_id="case001",
            metadata={"note": "test"},
        )
        assert event.event_type == EventType.CASE_CREATE
        assert event.user_id == "user001"

    def test_event_hash(self):
        event = AuditEvent(
            event_type=EventType.CASE_CREATE,
            timestamp_utc=time.time(),
            user_id="user001",
            case_id="case001",
        )
        h = event.compute_hash()
        assert len(h) == 64  # SHA-256

    def test_event_verify(self):
        event = AuditEvent(
            event_type=EventType.SIM_START,
            timestamp_utc=time.time(),
            user_id="user002",
            case_id="case002",
        )
        event.event_hash = event.compute_hash()
        assert event.verify() is True

    def test_tampered_event_fails_verify(self):
        event = AuditEvent(
            event_type=EventType.SIM_START,
            timestamp_utc=time.time(),
            user_id="user002",
            case_id="case002",
        )
        event.event_hash = event.compute_hash()
        event.metadata = {"tampered": True}
        assert event.verify() is False


class TestAuditLog:
    """Test append-only audit log."""

    def test_record_and_query(self):
        with tempfile.TemporaryDirectory() as td:
            log = AuditLog(Path(td) / "audit.jsonl")
            log.record(
                EventType.CASE_CREATE,
                user_id="user001",
                case_id="case001",
            )
            log.record(
                EventType.PLAN_CREATE,
                user_id="user001",
                case_id="case001",
            )
            events = log.query()
            assert len(events) == 2

    def test_chain_verification(self):
        with tempfile.TemporaryDirectory() as td:
            log = AuditLog(Path(td) / "audit.jsonl")
            for i in range(5):
                log.record(
                    EventType.CASE_CREATE,
                    user_id=f"user{i}",
                    case_id=f"case{i}",
                )
            assert log.verify_chain() is True

    def test_query_filter_by_case(self):
        with tempfile.TemporaryDirectory() as td:
            log = AuditLog(Path(td) / "audit.jsonl")
            log.record(EventType.CASE_CREATE, user_id="u1", case_id="case_A")
            log.record(EventType.CASE_CREATE, user_id="u1", case_id="case_B")
            log.record(EventType.PLAN_CREATE, user_id="u1", case_id="case_A")

            events_a = log.query(case_id="case_A")
            assert len(events_a) == 2

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "audit.jsonl"
            log1 = AuditLog(path)
            log1.record(EventType.CASE_CREATE, user_id="u1", case_id="c1")

            log2 = AuditLog(path)
            events = log2.query()
            assert len(events) >= 1


# ── Consent Manager ──────────────────────────────────────────────

class TestConsentScopes:
    """Test consent scope definitions."""

    def test_standard_disclaimers_populated(self):
        assert len(STANDARD_DISCLAIMERS) >= 5

    def test_scopes_exist(self):
        assert hasattr(ConsentScope, "DATA_COLLECTION")
        assert hasattr(ConsentScope, "SIMULATION_USE")
        assert hasattr(ConsentScope, "OUTCOME_TRACKING")


class TestConsentManager:
    """Test consent workflow."""

    def test_record_consent(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = ConsentManager(Path(td) / "consent.json")
            record = mgr.record_consent(
                "case001", "patient001", ConsentScope.SIMULATION_USE,
            )
            assert isinstance(record, ConsentRecord)
            assert record.case_id == "case001"
            # check_consent expects a frozenset
            ok, missing = mgr.check_consent(
                "case001", frozenset({ConsentScope.SIMULATION_USE}),
            )
            assert ok is True

    def test_revoke_consent(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = ConsentManager(Path(td) / "consent.json")
            record = mgr.record_consent(
                "c1", "p1", ConsentScope.DATA_COLLECTION,
            )
            ok, _ = mgr.check_consent(
                "c1", frozenset({ConsentScope.DATA_COLLECTION}),
            )
            assert ok is True
            mgr.revoke_consent(record.consent_id, "test revocation")
            ok2, missing = mgr.check_consent(
                "c1", frozenset({ConsentScope.DATA_COLLECTION}),
            )
            assert ok2 is False

    def test_can_simulate_requires_consent(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = ConsentManager(Path(td) / "consent.json")
            ok, _ = mgr.can_simulate("case_no_consent")
            assert ok is False
            # can_simulate requires DATA_COLLECTION + SIMULATION_USE + REPORT_GENERATION
            mgr.record_consent(
                "case_consented", "p1", ConsentScope.DATA_COLLECTION,
            )
            mgr.record_consent(
                "case_consented", "p1", ConsentScope.SIMULATION_USE,
            )
            mgr.record_consent(
                "case_consented", "p1", ConsentScope.REPORT_GENERATION,
            )
            ok2, _ = mgr.can_simulate("case_consented")
            assert ok2 is True


# ── Access Control ───────────────────────────────────────────────

class TestRolePermissions:
    """Test role-permission mappings."""

    def test_surgeon_has_plan_permissions(self):
        perms = ROLE_PERMISSIONS[Role.SURGEON]
        assert Permission.PLAN_CREATE in perms
        assert Permission.SIM_RUN in perms

    def test_auditor_limited(self):
        perms = ROLE_PERMISSIONS[Role.AUDITOR]
        assert Permission.PLAN_CREATE not in perms

    def test_administrator_has_manage(self):
        admin_perms = ROLE_PERMISSIONS[Role.ADMINISTRATOR]
        assert Permission.ADMIN_USER_MANAGE in admin_perms


class TestAccessControl:
    """Test access control enforcement."""

    def test_add_user_and_check(self):
        with tempfile.TemporaryDirectory() as td:
            ac = AccessControl(Path(td) / "access.json")
            ac.add_user("user001", "Dr. Smith", Role.SURGEON)
            user = ac.get_user("user001")
            assert user is not None
            assert user.role == Role.SURGEON

    def test_check_access_granted(self):
        with tempfile.TemporaryDirectory() as td:
            ac = AccessControl(Path(td) / "access.json")
            ac.add_user("user001", "Dr. Smith", Role.SURGEON)
            decision = ac.check_access("user001", Permission.PLAN_CREATE)
            assert decision.granted is True

    def test_check_access_denied(self):
        with tempfile.TemporaryDirectory() as td:
            ac = AccessControl(Path(td) / "access.json")
            ac.add_user("user001", "Auditor", Role.AUDITOR)
            decision = ac.check_access("user001", Permission.PLAN_CREATE)
            assert decision.granted is False

    def test_case_specific_access(self):
        with tempfile.TemporaryDirectory() as td:
            ac = AccessControl(Path(td) / "access.json")
            ac.add_user("user001", "Dr. Smith", Role.SURGEON)
            ac.grant_case_access("user001", "case_A")
            decision = ac.check_access(
                "user001", Permission.CASE_READ, case_id="case_A",
            )
            assert decision.granted is True

    def test_deactivated_user_denied(self):
        with tempfile.TemporaryDirectory() as td:
            ac = AccessControl(Path(td) / "access.json")
            ac.add_user("user001", "Dr. Smith", Role.SURGEON)
            ac.deactivate_user("user001")
            decision = ac.check_access("user001", Permission.CASE_READ)
            assert decision.granted is False
