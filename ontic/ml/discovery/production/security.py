"""
Security Hardening for Production Systems

Input validation, authentication, request signing, and audit logging.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Type, Union
from enum import Enum
from functools import wraps
import time
import threading
import hashlib
import hmac
import secrets
import re
import json
import base64
from datetime import datetime, timezone
from collections import defaultdict
import ipaddress
import logging

from ..config import get_config

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Input validation error."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(f"{field}: {message}" if field else message)


class ValidationRule:
    """Base validation rule."""
    
    def validate(self, value: Any, field_name: str) -> None:
        """Validate value. Raises ValidationError if invalid."""
        raise NotImplementedError


class RequiredRule(ValidationRule):
    """Require value to be present."""
    
    def validate(self, value: Any, field_name: str) -> None:
        if value is None or value == "":
            raise ValidationError("Field is required", field_name, value)


class TypeRule(ValidationRule):
    """Require value to be specific type."""
    
    def __init__(self, expected_type: Type):
        self.expected_type = expected_type
    
    def validate(self, value: Any, field_name: str) -> None:
        if value is not None and not isinstance(value, self.expected_type):
            raise ValidationError(
                f"Expected {self.expected_type.__name__}, got {type(value).__name__}",
                field_name,
                value,
            )


class RangeRule(ValidationRule):
    """Require numeric value to be in range."""
    
    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, value: Any, field_name: str) -> None:
        if value is None:
            return
        
        if self.min_value is not None and value < self.min_value:
            raise ValidationError(
                f"Value must be >= {self.min_value}",
                field_name,
                value,
            )
        
        if self.max_value is not None and value > self.max_value:
            raise ValidationError(
                f"Value must be <= {self.max_value}",
                field_name,
                value,
            )


class LengthRule(ValidationRule):
    """Require string/list length to be in range."""
    
    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
    
    def validate(self, value: Any, field_name: str) -> None:
        if value is None:
            return
        
        length = len(value)
        
        if self.min_length is not None and length < self.min_length:
            raise ValidationError(
                f"Length must be >= {self.min_length}",
                field_name,
                value,
            )
        
        if self.max_length is not None and length > self.max_length:
            raise ValidationError(
                f"Length must be <= {self.max_length}",
                field_name,
                value,
            )


class PatternRule(ValidationRule):
    """Require string to match regex pattern."""
    
    def __init__(self, pattern: Union[str, Pattern]):
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
    
    def validate(self, value: Any, field_name: str) -> None:
        if value is None:
            return
        
        if not self.pattern.match(str(value)):
            raise ValidationError(
                f"Value does not match pattern: {self.pattern.pattern}",
                field_name,
                value,
            )


class EnumRule(ValidationRule):
    """Require value to be one of allowed values."""
    
    def __init__(self, allowed: Set[Any]):
        self.allowed = allowed
    
    def validate(self, value: Any, field_name: str) -> None:
        if value is None:
            return
        
        if value not in self.allowed:
            raise ValidationError(
                f"Value must be one of: {', '.join(map(str, self.allowed))}",
                field_name,
                value,
            )


class EmailRule(PatternRule):
    """Validate email format."""
    
    EMAIL_PATTERN = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    def __init__(self):
        super().__init__(self.EMAIL_PATTERN)


class URLRule(PatternRule):
    """Validate URL format."""
    
    URL_PATTERN = re.compile(
        r'^https?://[a-zA-Z0-9][-a-zA-Z0-9]*(\.[a-zA-Z0-9][-a-zA-Z0-9]*)+(/.*)?$'
    )
    
    def __init__(self):
        super().__init__(self.URL_PATTERN)


class IPAddressRule(ValidationRule):
    """Validate IP address format."""
    
    def __init__(self, version: Optional[int] = None):
        self.version = version  # 4 or 6, or None for both
    
    def validate(self, value: Any, field_name: str) -> None:
        if value is None:
            return
        
        try:
            addr = ipaddress.ip_address(value)
            if self.version and addr.version != self.version:
                raise ValidationError(
                    f"Expected IPv{self.version} address",
                    field_name,
                    value,
                )
        except ValueError as e:
            raise ValidationError(str(e), field_name, value)


@dataclass
class FieldSpec:
    """Field validation specification."""
    name: str
    rules: List[ValidationRule] = field(default_factory=list)
    
    def validate(self, value: Any) -> None:
        """Validate value against all rules."""
        for rule in self.rules:
            rule.validate(value, self.name)


class InputValidator:
    """
    Input validation engine.
    
    Validates input data against defined rules.
    """
    
    # Dangerous patterns for SQL injection, XSS, etc.
    DANGEROUS_PATTERNS = [
        re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'on\w+\s*=', re.IGNORECASE),  # onclick, onload, etc.
        re.compile(r"['\"]\s*OR\s+['\"0-9]", re.IGNORECASE),  # SQL injection
        re.compile(r';\s*(DROP|DELETE|UPDATE|INSERT|TRUNCATE)', re.IGNORECASE),
        re.compile(r'--\s*$', re.MULTILINE),  # SQL comment
        re.compile(r'\$\{.*\}'),  # Template injection
        re.compile(r'\{\{.*\}\}'),  # Template injection
    ]
    
    def __init__(self):
        """Initialize validator."""
        self._field_specs: Dict[str, List[FieldSpec]] = defaultdict(list)
    
    def add_field(self, schema: str, spec: FieldSpec) -> None:
        """
        Add field validation spec.
        
        Args:
            schema: Schema name (e.g., "create_user")
            spec: Field specification
        """
        self._field_specs[schema].append(spec)
    
    def validate(self, schema: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against schema.
        
        Args:
            schema: Schema name
            data: Data to validate
            
        Returns:
            Validated data
            
        Raises:
            ValidationError: If validation fails
        """
        errors = []
        
        for spec in self._field_specs.get(schema, []):
            value = data.get(spec.name)
            try:
                spec.validate(value)
            except ValidationError as e:
                errors.append(e)
        
        if errors:
            raise ValidationError(
                f"Validation failed: {'; '.join(e.message for e in errors)}"
            )
        
        return data
    
    @classmethod
    def is_safe(cls, value: str) -> bool:
        """Check if string is safe from injection attacks."""
        for pattern in cls.DANGEROUS_PATTERNS:
            if pattern.search(value):
                return False
        return True
    
    @classmethod
    def sanitize(cls, value: str) -> str:
        """
        Sanitize string by escaping dangerous characters.
        
        Args:
            value: Input string
            
        Returns:
            Sanitized string
        """
        # HTML escape
        escapes = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
        }
        
        result = value
        for char, escape in escapes.items():
            result = result.replace(char, escape)
        
        return result
    
    @classmethod
    def validate_json(cls, data: str, max_depth: int = 10) -> Dict[str, Any]:
        """
        Safely parse and validate JSON.
        
        Args:
            data: JSON string
            max_depth: Maximum nesting depth
            
        Returns:
            Parsed JSON
            
        Raises:
            ValidationError: If JSON is invalid or too deeply nested
        """
        def check_depth(obj: Any, depth: int = 0) -> None:
            if depth > max_depth:
                raise ValidationError(f"JSON nesting exceeds max depth of {max_depth}")
            
            if isinstance(obj, dict):
                for v in obj.values():
                    check_depth(v, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    check_depth(item, depth + 1)
        
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {e}")
        
        check_depth(parsed)
        return parsed


def sanitize_input(value: str) -> str:
    """Convenience function to sanitize input."""
    return InputValidator.sanitize(value)


class AuthenticationError(Exception):
    """Authentication failed."""
    pass


class AuthorizationError(Exception):
    """Authorization failed."""
    pass


@dataclass
class APIKey:
    """API key representation."""
    key_id: str
    key_hash: str  # Hashed key value
    name: str
    permissions: Set[str]
    rate_limit: Optional[int] = None  # Requests per minute
    expires_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    last_used_at: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def has_permission(self, permission: str) -> bool:
        """Check if key has permission."""
        if "*" in self.permissions:
            return True
        return permission in self.permissions


class APIKeyAuth:
    """
    API key authentication handler.
    
    Manages API key generation, validation, and storage.
    """
    
    def __init__(self, header_name: str = "X-API-Key"):
        """
        Initialize authenticator.
        
        Args:
            header_name: HTTP header name for API key
        """
        self.header_name = header_name
        self._keys: Dict[str, APIKey] = {}
        self._key_by_hash: Dict[str, str] = {}  # hash -> key_id
        self._lock = threading.Lock()
    
    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash an API key."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def generate_key(
        self,
        name: str,
        permissions: Optional[Set[str]] = None,
        rate_limit: Optional[int] = None,
        expires_in_days: Optional[int] = None,
    ) -> tuple[str, APIKey]:
        """
        Generate a new API key.
        
        Args:
            name: Key name/description
            permissions: Set of allowed permissions
            rate_limit: Rate limit (requests per minute)
            expires_in_days: Expiration in days
            
        Returns:
            Tuple of (raw_key, APIKey)
        """
        key_id = secrets.token_hex(8)
        raw_key = f"tn_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(raw_key)
        
        expires_at = None
        if expires_in_days:
            expires_at = time.time() + (expires_in_days * 86400)
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            permissions=permissions or {"read"},
            rate_limit=rate_limit,
            expires_at=expires_at,
        )
        
        with self._lock:
            self._keys[key_id] = api_key
            self._key_by_hash[key_hash] = key_id
        
        return raw_key, api_key
    
    def validate(self, raw_key: str) -> APIKey:
        """
        Validate an API key.
        
        Args:
            raw_key: Raw API key string
            
        Returns:
            APIKey if valid
            
        Raises:
            AuthenticationError: If key is invalid or expired
        """
        key_hash = self._hash_key(raw_key)
        
        with self._lock:
            key_id = self._key_by_hash.get(key_hash)
            if not key_id:
                raise AuthenticationError("Invalid API key")
            
            api_key = self._keys.get(key_id)
            if not api_key:
                raise AuthenticationError("Invalid API key")
            
            if api_key.is_expired():
                raise AuthenticationError("API key has expired")
            
            # Update last used
            api_key.last_used_at = time.time()
            
            return api_key
    
    def revoke(self, key_id: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            key_id: Key ID to revoke
            
        Returns:
            True if revoked, False if not found
        """
        with self._lock:
            api_key = self._keys.pop(key_id, None)
            if api_key:
                self._key_by_hash.pop(api_key.key_hash, None)
                return True
            return False
    
    def require_auth(
        self,
        permission: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to require API key authentication.
        
        Args:
            permission: Required permission
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract key from kwargs (assumes key is passed as 'api_key')
                raw_key = kwargs.pop('api_key', None)
                if not raw_key:
                    raise AuthenticationError("API key required")
                
                api_key = self.validate(raw_key)
                
                if permission and not api_key.has_permission(permission):
                    raise AuthorizationError(
                        f"Permission '{permission}' required"
                    )
                
                # Add authenticated key to kwargs
                kwargs['authenticated_key'] = api_key
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def list_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (excluding hash)."""
        with self._lock:
            return [
                {
                    "key_id": k.key_id,
                    "name": k.name,
                    "permissions": list(k.permissions),
                    "rate_limit": k.rate_limit,
                    "expires_at": k.expires_at,
                    "created_at": k.created_at,
                    "last_used_at": k.last_used_at,
                    "is_expired": k.is_expired(),
                }
                for k in self._keys.values()
            ]


class RequestSigner:
    """
    HMAC request signing for API requests.
    
    Signs requests to prevent tampering.
    """
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "sha256",
        header_name: str = "X-Signature",
        timestamp_header: str = "X-Timestamp",
        max_age_seconds: Optional[int] = None,
    ):
        """
        Initialize request signer.
        
        Args:
            secret_key: Signing secret (generated if not provided)
            algorithm: HMAC algorithm
            header_name: Signature header name
            timestamp_header: Timestamp header name
            max_age_seconds: Maximum request age (default from config)
        """
        config = get_config()
        
        self.secret_key = secret_key or secrets.token_hex(32)
        self.algorithm = algorithm
        self.header_name = header_name
        self.timestamp_header = timestamp_header
        self.max_age_seconds = max_age_seconds if max_age_seconds is not None else config.security.max_request_age_seconds
    
    def sign(
        self,
        method: str,
        path: str,
        body: Optional[str] = None,
        timestamp: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Sign a request.
        
        Args:
            method: HTTP method
            path: Request path
            body: Request body
            timestamp: Unix timestamp (current if not provided)
            
        Returns:
            Dict with signature headers
        """
        timestamp = timestamp or int(time.time())
        
        # Create signing string
        components = [
            method.upper(),
            path,
            str(timestamp),
            body or "",
        ]
        signing_string = "\n".join(components)
        
        # Generate HMAC
        signature = hmac.new(
            self.secret_key.encode(),
            signing_string.encode(),
            self.algorithm,
        ).hexdigest()
        
        return {
            self.header_name: signature,
            self.timestamp_header: str(timestamp),
        }
    
    def verify(
        self,
        method: str,
        path: str,
        signature: str,
        timestamp: int,
        body: Optional[str] = None,
    ) -> bool:
        """
        Verify request signature.
        
        Args:
            method: HTTP method
            path: Request path
            signature: Provided signature
            timestamp: Request timestamp
            body: Request body
            
        Returns:
            True if signature is valid
            
        Raises:
            AuthenticationError: If signature is invalid or expired
        """
        # Check timestamp age
        age = abs(time.time() - timestamp)
        if age > self.max_age_seconds:
            raise AuthenticationError(
                f"Request expired (age: {age:.0f}s, max: {self.max_age_seconds}s)"
            )
        
        # Generate expected signature
        expected = self.sign(method, path, body, timestamp)
        expected_sig = expected[self.header_name]
        
        # Constant-time comparison
        if not hmac.compare_digest(signature, expected_sig):
            raise AuthenticationError("Invalid request signature")
        
        return True


class AuditEventType(str, Enum):
    """Audit event types."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    ADMIN_ACTION = "admin_action"
    SECURITY_EVENT = "security_event"
    API_CALL = "api_call"


@dataclass
class AuditEvent:
    """Audit log event."""
    event_id: str
    event_type: AuditEventType
    timestamp: str
    actor: str  # User/API key ID
    action: str
    resource: str
    status: str  # success, failure
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "status": self.status,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
            "details": self.details,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """
    Audit logging for compliance and security.
    
    Records security-relevant events for audit trails.
    """
    
    def __init__(
        self,
        service_name: str = "ontic-discovery",
        max_events: int = 10000,
    ):
        """
        Initialize audit logger.
        
        Args:
            service_name: Service name for events
            max_events: Maximum events to retain in memory
        """
        self.service_name = service_name
        self.max_events = max_events
        self._events: List[AuditEvent] = []
        self._lock = threading.Lock()
        self._handlers: List[Callable[[AuditEvent], None]] = []
    
    def add_handler(self, handler: Callable[[AuditEvent], None]) -> None:
        """Add event handler."""
        self._handlers.append(handler)
    
    def log(
        self,
        event_type: AuditEventType,
        actor: str,
        action: str,
        resource: str,
        status: str = "success",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        **details,
    ) -> AuditEvent:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            actor: Who performed the action
            action: What action was performed
            resource: What resource was affected
            status: success or failure
            ip_address: Client IP address
            user_agent: Client user agent
            request_id: Request ID for correlation
            **details: Additional details
            
        Returns:
            Created AuditEvent
        """
        event = AuditEvent(
            event_id=secrets.token_hex(16),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            actor=actor,
            action=action,
            resource=resource,
            status=status,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            details=details,
        )
        
        with self._lock:
            self._events.append(event)
            
            # Trim if over limit
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events:]
        
        # Call handlers
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Audit handler error: {e}")
        
        # Log to standard logger
        logger.info(f"AUDIT: {event.to_json()}")
        
        return event
    
    def log_authentication(
        self,
        actor: str,
        success: bool,
        method: str = "api_key",
        **details,
    ) -> AuditEvent:
        """Log authentication event."""
        return self.log(
            AuditEventType.AUTHENTICATION,
            actor=actor,
            action=f"authenticate_{method}",
            resource="auth",
            status="success" if success else "failure",
            **details,
        )
    
    def log_authorization(
        self,
        actor: str,
        permission: str,
        resource: str,
        granted: bool,
        **details,
    ) -> AuditEvent:
        """Log authorization event."""
        return self.log(
            AuditEventType.AUTHORIZATION,
            actor=actor,
            action=f"check_{permission}",
            resource=resource,
            status="success" if granted else "failure",
            **details,
        )
    
    def log_api_call(
        self,
        actor: str,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
        **details,
    ) -> AuditEvent:
        """Log API call event."""
        return self.log(
            AuditEventType.API_CALL,
            actor=actor,
            action=f"{method.upper()}_{endpoint}",
            resource=endpoint,
            status="success" if status_code < 400 else "failure",
            status_code=status_code,
            latency_ms=latency_ms,
            **details,
        )
    
    def log_security_event(
        self,
        actor: str,
        event_name: str,
        severity: str = "medium",
        **details,
    ) -> AuditEvent:
        """Log security event."""
        return self.log(
            AuditEventType.SECURITY_EVENT,
            actor=actor,
            action=event_name,
            resource="security",
            status="alert",
            severity=severity,
            **details,
        )
    
    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        actor: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """
        Query audit events.
        
        Args:
            event_type: Filter by event type
            actor: Filter by actor
            since: Filter events after this ISO timestamp
            limit: Maximum events to return
            
        Returns:
            List of matching events
        """
        with self._lock:
            events = list(self._events)
        
        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if actor:
            events = [e for e in events if e.actor == actor]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        # Return most recent
        return events[-limit:]
    
    def export(self, format: str = "json") -> str:
        """Export audit log."""
        with self._lock:
            events = list(self._events)
        
        if format == "json":
            return json.dumps(
                [e.to_dict() for e in events],
                default=str,
                indent=2,
            )
        elif format == "jsonl":
            return "\n".join(e.to_json() for e in events)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def clear(self) -> int:
        """Clear audit log and return count of cleared events."""
        with self._lock:
            count = len(self._events)
            self._events.clear()
            return count


# Content security policy helpers

@dataclass
class CSPPolicy:
    """Content Security Policy configuration."""
    default_src: List[str] = field(default_factory=lambda: ["'self'"])
    script_src: List[str] = field(default_factory=lambda: ["'self'"])
    style_src: List[str] = field(default_factory=lambda: ["'self'"])
    img_src: List[str] = field(default_factory=lambda: ["'self'", "data:"])
    connect_src: List[str] = field(default_factory=lambda: ["'self'"])
    font_src: List[str] = field(default_factory=lambda: ["'self'"])
    object_src: List[str] = field(default_factory=lambda: ["'none'"])
    frame_ancestors: List[str] = field(default_factory=lambda: ["'none'"])
    base_uri: List[str] = field(default_factory=lambda: ["'self'"])
    form_action: List[str] = field(default_factory=lambda: ["'self'"])
    
    def to_header(self) -> str:
        """Generate CSP header value."""
        directives = []
        
        if self.default_src:
            directives.append(f"default-src {' '.join(self.default_src)}")
        if self.script_src:
            directives.append(f"script-src {' '.join(self.script_src)}")
        if self.style_src:
            directives.append(f"style-src {' '.join(self.style_src)}")
        if self.img_src:
            directives.append(f"img-src {' '.join(self.img_src)}")
        if self.connect_src:
            directives.append(f"connect-src {' '.join(self.connect_src)}")
        if self.font_src:
            directives.append(f"font-src {' '.join(self.font_src)}")
        if self.object_src:
            directives.append(f"object-src {' '.join(self.object_src)}")
        if self.frame_ancestors:
            directives.append(f"frame-ancestors {' '.join(self.frame_ancestors)}")
        if self.base_uri:
            directives.append(f"base-uri {' '.join(self.base_uri)}")
        if self.form_action:
            directives.append(f"form-action {' '.join(self.form_action)}")
        
        return "; ".join(directives)


def get_security_headers(csp: Optional[CSPPolicy] = None) -> Dict[str, str]:
    """
    Get recommended security headers.
    
    Args:
        csp: Content Security Policy configuration
        
    Returns:
        Dict of security headers
    """
    headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    }
    
    if csp:
        headers["Content-Security-Policy"] = csp.to_header()
    
    return headers
