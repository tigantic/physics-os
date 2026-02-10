"""Role-based access control for platform operations.

Defines roles, permissions, and access policies for:
  - Surgeons (full access to their own cases)
  - Residents (read + plan, no final report export)
  - Researchers (anonymized data access only)
  - Administrators (user management, config)
  - Auditors (read-only audit trail access)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Granular permissions."""
    CASE_CREATE = "case.create"
    CASE_READ = "case.read"
    CASE_UPDATE = "case.update"
    CASE_DELETE = "case.delete"
    CASE_EXPORT = "case.export"
    PLAN_CREATE = "plan.create"
    PLAN_UPDATE = "plan.update"
    PLAN_DELETE = "plan.delete"
    SIM_RUN = "simulation.run"
    SIM_READ = "simulation.read"
    REPORT_GENERATE = "report.generate"
    REPORT_EXPORT = "report.export"
    OUTCOME_IMPORT = "outcome.import"
    OUTCOME_READ = "outcome.read"
    AUDIT_READ = "audit.read"
    ADMIN_USER_MANAGE = "admin.user_manage"
    ADMIN_CONFIG = "admin.config"
    RESEARCH_ACCESS = "research.access"


class Role(Enum):
    """Standard platform roles."""
    SURGEON = "surgeon"
    RESIDENT = "resident"
    RESEARCHER = "researcher"
    ADMINISTRATOR = "administrator"
    AUDITOR = "auditor"


# ── Role → permission mapping ────────────────────────────────────

ROLE_PERMISSIONS: Dict[Role, FrozenSet[Permission]] = {
    Role.SURGEON: frozenset({
        Permission.CASE_CREATE,
        Permission.CASE_READ,
        Permission.CASE_UPDATE,
        Permission.CASE_DELETE,
        Permission.CASE_EXPORT,
        Permission.PLAN_CREATE,
        Permission.PLAN_UPDATE,
        Permission.PLAN_DELETE,
        Permission.SIM_RUN,
        Permission.SIM_READ,
        Permission.REPORT_GENERATE,
        Permission.REPORT_EXPORT,
        Permission.OUTCOME_IMPORT,
        Permission.OUTCOME_READ,
    }),
    Role.RESIDENT: frozenset({
        Permission.CASE_READ,
        Permission.PLAN_CREATE,
        Permission.PLAN_UPDATE,
        Permission.SIM_RUN,
        Permission.SIM_READ,
        Permission.REPORT_GENERATE,
        Permission.OUTCOME_READ,
    }),
    Role.RESEARCHER: frozenset({
        Permission.CASE_READ,
        Permission.SIM_READ,
        Permission.OUTCOME_READ,
        Permission.RESEARCH_ACCESS,
    }),
    Role.ADMINISTRATOR: frozenset({
        Permission.CASE_READ,
        Permission.ADMIN_USER_MANAGE,
        Permission.ADMIN_CONFIG,
        Permission.AUDIT_READ,
    }),
    Role.AUDITOR: frozenset({
        Permission.AUDIT_READ,
        Permission.CASE_READ,
        Permission.SIM_READ,
    }),
}


@dataclass
class UserProfile:
    """A platform user."""
    user_id: str
    display_name: str
    role: Role
    email: str = ""
    institution: str = ""
    is_active: bool = True
    created_at: float = 0.0
    last_login: float = 0.0
    extra_permissions: Set[Permission] = field(default_factory=set)
    revoked_permissions: Set[Permission] = field(default_factory=set)
    case_access: Set[str] = field(default_factory=set)  # explicit case IDs

    @property
    def effective_permissions(self) -> FrozenSet[Permission]:
        """Compute effective permissions (role + extras - revoked)."""
        base = set(ROLE_PERMISSIONS.get(self.role, frozenset()))
        base.update(self.extra_permissions)
        base -= self.revoked_permissions
        return frozenset(base)

    def has_permission(self, perm: Permission) -> bool:
        return self.is_active and perm in self.effective_permissions

    def has_case_access(self, case_id: str) -> bool:
        """Check if user can access a specific case.

        Surgeons/admins have access to all cases by default.
        Other roles need explicit case_access grants.
        """
        if self.role in (Role.SURGEON, Role.ADMINISTRATOR):
            return True
        return case_id in self.case_access

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "role": self.role.value,
            "email": self.email,
            "institution": self.institution,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "extra_permissions": [p.value for p in self.extra_permissions],
            "revoked_permissions": [p.value for p in self.revoked_permissions],
            "case_access": list(self.case_access),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> UserProfile:
        return cls(
            user_id=d["user_id"],
            display_name=d["display_name"],
            role=Role(d["role"]),
            email=d.get("email", ""),
            institution=d.get("institution", ""),
            is_active=d.get("is_active", True),
            created_at=d.get("created_at", 0.0),
            last_login=d.get("last_login", 0.0),
            extra_permissions={Permission(p) for p in d.get("extra_permissions", [])},
            revoked_permissions={Permission(p) for p in d.get("revoked_permissions", [])},
            case_access=set(d.get("case_access", [])),
        )


@dataclass
class AccessDecision:
    """Result of an access control check."""
    granted: bool
    user_id: str
    permission: Permission
    case_id: str = ""
    reason: str = ""
    timestamp: float = 0.0


class AccessControl:
    """Role-based access control manager.

    Manages users, roles, and per-case access decisions.
    All access decisions are logged for audit purposes.
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._users: Dict[str, UserProfile] = {}
        self._access_log: List[AccessDecision] = []
        self._storage_path = storage_path

        if storage_path and storage_path.exists():
            self._load()

    def add_user(
        self,
        user_id: str,
        display_name: str,
        role: Role,
        *,
        email: str = "",
        institution: str = "",
    ) -> UserProfile:
        """Register a new user."""
        if user_id in self._users:
            raise ValueError(f"User {user_id} already exists")

        user = UserProfile(
            user_id=user_id,
            display_name=display_name,
            role=role,
            email=email,
            institution=institution,
            created_at=time.time(),
        )
        self._users[user_id] = user
        self._save()
        logger.info("User created: %s (%s)", user_id, role.value)
        return user

    def get_user(self, user_id: str) -> Optional[UserProfile]:
        return self._users.get(user_id)

    def update_role(self, user_id: str, new_role: Role) -> bool:
        """Change a user's role."""
        user = self._users.get(user_id)
        if user is None:
            return False
        user.role = new_role
        self._save()
        logger.info("User %s role changed to %s", user_id, new_role.value)
        return True

    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user (soft delete)."""
        user = self._users.get(user_id)
        if user is None:
            return False
        user.is_active = False
        self._save()
        logger.info("User %s deactivated", user_id)
        return True

    def grant_case_access(self, user_id: str, case_id: str) -> bool:
        """Grant a user access to a specific case."""
        user = self._users.get(user_id)
        if user is None:
            return False
        user.case_access.add(case_id)
        self._save()
        return True

    def revoke_case_access(self, user_id: str, case_id: str) -> bool:
        """Revoke a user's access to a specific case."""
        user = self._users.get(user_id)
        if user is None:
            return False
        user.case_access.discard(case_id)
        self._save()
        return True

    def check_access(
        self,
        user_id: str,
        permission: Permission,
        case_id: str = "",
    ) -> AccessDecision:
        """Check if a user has permission (optionally for a specific case).

        Records the decision for audit purposes.
        """
        decision = AccessDecision(
            granted=False,
            user_id=user_id,
            permission=permission,
            case_id=case_id,
            timestamp=time.time(),
        )

        user = self._users.get(user_id)
        if user is None:
            decision.reason = "user not found"
            self._access_log.append(decision)
            return decision

        if not user.is_active:
            decision.reason = "user deactivated"
            self._access_log.append(decision)
            return decision

        if not user.has_permission(permission):
            decision.reason = f"missing permission: {permission.value}"
            self._access_log.append(decision)
            return decision

        if case_id and not user.has_case_access(case_id):
            decision.reason = f"no case access: {case_id}"
            self._access_log.append(decision)
            return decision

        decision.granted = True
        decision.reason = "authorized"
        self._access_log.append(decision)
        return decision

    def list_users(self, role: Optional[Role] = None) -> List[UserProfile]:
        """List all users, optionally filtered by role."""
        users = list(self._users.values())
        if role is not None:
            users = [u for u in users if u.role == role]
        return users

    def recent_access_decisions(
        self,
        limit: int = 50,
        denied_only: bool = False,
    ) -> List[AccessDecision]:
        """Get recent access decisions."""
        decisions = self._access_log[-limit:]
        if denied_only:
            decisions = [d for d in decisions if not d.granted]
        return list(reversed(decisions))

    def _save(self) -> None:
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = [u.to_dict() for u in self._users.values()]
        with open(self._storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self) -> None:
        if self._storage_path is None or not self._storage_path.exists():
            return
        with open(self._storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            try:
                user = UserProfile.from_dict(d)
                self._users[user.user_id] = user
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping invalid user record: %s", exc)
