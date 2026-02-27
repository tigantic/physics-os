"""
Logging Infrastructure for HyperFOAM Intake
===========================================

Structured logging for auditability and debugging.

Per Article VI (Documentation Duty):
    - All significant operations logged
    - Logs are structured (JSON format available)
    - Log levels follow standard convention
    - No sensitive data in logs (sanitized)

CONFIGURATION:
--------------
    LOG_LEVEL: Environment variable (default: INFO)
    LOG_FORMAT: 'json' or 'text' (default: text)
    LOG_FILE: Path to log file (default: None - stdout only)

USAGE:
------
    from staging.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("Processing document", extra={"file": "spec.pdf", "size_kb": 150})
    logger.error("Failed to parse", extra={"error": "timeout"})

LOG LEVELS:
-----------
    DEBUG: Development/troubleshooting only
    INFO: Normal operation events
    WARNING: Recoverable issues, user should be aware
    ERROR: Operation failed, needs attention
    CRITICAL: System-level failure

STRUCTURED FIELDS:
------------------
    All logs include:
    - timestamp: ISO 8601 format
    - level: DEBUG/INFO/WARNING/ERROR/CRITICAL
    - logger: Module name
    - message: Human-readable description
    
    Optional context (via extra={}):
    - case_id: Simulation case ID
    - file: Filename being processed
    - duration_ms: Operation duration
    - user_action: UI action taken
"""

import logging
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional


# =============================================================================
# Configuration from Environment
# =============================================================================

LOG_LEVEL = os.environ.get("HYPERFOAM_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.environ.get("HYPERFOAM_LOG_FORMAT", "text")  # 'text' or 'json'
LOG_FILE = os.environ.get("HYPERFOAM_LOG_FILE", None)


# =============================================================================
# Custom JSON Formatter
# =============================================================================

class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Output format (single line per log):
    {"timestamp": "...", "level": "INFO", "logger": "staging.ingestor", "message": "...", ...}
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add location info for debugging
        if record.levelno >= logging.WARNING:
            log_data["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields (sanitized)
        if hasattr(record, "__dict__"):
            # Standard fields to skip
            skip_fields = {
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                'taskName', 'message',
            }
            
            for key, value in record.__dict__.items():
                if key not in skip_fields and not key.startswith('_'):
                    # Sanitize sensitive fields
                    if any(sensitive in key.lower() for sensitive in ['password', 'token', 'secret', 'key']):
                        value = "[REDACTED]"
                    log_data[key] = value
        
        return json.dumps(log_data, default=str)


class HumanFormatter(logging.Formatter):
    """
    Human-readable formatter for console output.
    
    Format: 2024-01-15 14:30:00 [INFO] staging.ingestor: Processing document
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',
    }
    
    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color and sys.stderr.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as human-readable text."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        
        if self.use_color:
            color = self.COLORS.get(level, '')
            reset = self.COLORS['RESET']
            level_str = f"{color}[{level:8}]{reset}"
        else:
            level_str = f"[{level:8}]"
        
        # Base message
        msg = f"{timestamp} {level_str} {record.name}: {record.getMessage()}"
        
        # Add extra context if present
        extras = []
        skip_fields = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName',
            'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'pathname', 'process', 'processName', 'relativeCreated',
            'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
            'taskName', 'message',
        }
        
        for key, value in record.__dict__.items():
            if key not in skip_fields and not key.startswith('_'):
                extras.append(f"{key}={value}")
        
        if extras:
            msg += f" | {', '.join(extras)}"
        
        # Add exception info
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
        
        return msg


# =============================================================================
# Logger Factory
# =============================================================================

_loggers: Dict[str, logging.Logger] = {}


def get_logger(name: str = "hyperfoam") -> logging.Logger:
    """
    Get or create a logger with proper configuration.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    
    Example:
        logger = get_logger(__name__)
        logger.info("Document parsed", extra={"fields": 8, "duration_ms": 150})
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        
        if LOG_FORMAT == "json":
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(HumanFormatter())
        
        logger.addHandler(console_handler)
        
        # File handler (if configured)
        if LOG_FILE:
            try:
                file_handler = logging.FileHandler(LOG_FILE)
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(StructuredFormatter())  # Always JSON for files
                logger.addHandler(file_handler)
            except (IOError, OSError) as e:
                logger.warning(f"Could not create log file: {e}")
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    _loggers[name] = logger
    return logger


# =============================================================================
# Convenience Functions
# =============================================================================

def log_operation(operation: str, **kwargs) -> None:
    """
    Log an operation with structured context.
    
    Args:
        operation: Operation name (e.g., "parse_document", "submit_job")
        **kwargs: Additional context fields
    
    Example:
        log_operation("parse_document", file="spec.pdf", duration_ms=150, fields=8)
    """
    logger = get_logger("hyperfoam.operations")
    logger.info(operation, extra=kwargs)


def log_user_action(action: str, **kwargs) -> None:
    """
    Log a user action for audit trail.
    
    Args:
        action: Action name (e.g., "upload_file", "confirm_field", "submit")
        **kwargs: Additional context fields
    """
    logger = get_logger("hyperfoam.audit")
    logger.info(action, extra=kwargs)


def log_error(message: str, exception: Optional[Exception] = None, **kwargs) -> None:
    """
    Log an error with context.
    
    Args:
        message: Error description
        exception: Optional exception object
        **kwargs: Additional context fields
    """
    logger = get_logger("hyperfoam.errors")
    logger.error(message, exc_info=exception, extra=kwargs)


# =============================================================================
# Self-Test
# =============================================================================

def _self_test():
    """Run self-test of logging infrastructure."""
    logger = get_logger("hyperfoam.test")
    
    logger.debug("Debug message - verbose details")
    logger.info("Info message - normal operation", extra={"file_name": "test.pdf"})
    logger.warning("Warning message - recoverable issue")
    logger.error("Error message - operation failed")
    
    # Test convenience functions
    log_operation("test_parse", file_name="sample.pdf", duration_ms=50)
    log_user_action("upload", file_name="spec.xlsx", size_kb=250)
    
    print("\n✅ Logging self-test complete!")


if __name__ == "__main__":
    _self_test()
