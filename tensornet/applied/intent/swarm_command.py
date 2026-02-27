"""
Swarm Command Interface
========================

Natural language command and control for autonomous swarm.

Input: "Swarm Alpha, intercept trajectory vector 3-5-0 at Mach 8."
Output: Swarm calculates Phase 3 path and Phase 4 agents execute it.

This is the C2 (Command and Control) layer - where human intent
becomes machine action.
"""

from __future__ import annotations

import os
import re

# Import intent system
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# SWARM COMMAND TYPES
# =============================================================================


class SwarmCommandType(Enum):
    """Types of swarm commands."""

    # Movement
    INTERCEPT = "intercept"  # Intercept target
    PROCEED = "proceed"  # Continue on trajectory
    HOLD = "hold"  # Hold position
    RTB = "rtb"  # Return to base

    # Formation
    FORMATION = "formation"  # Change formation
    SPREAD = "spread"  # Increase spacing
    TIGHTEN = "tighten"  # Decrease spacing

    # Combat
    ATTACK = "attack"  # Engage target
    EVADE = "evade"  # Evasive maneuvers
    DEFEND = "defend"  # Defensive posture

    # Control
    STATUS = "status"  # Report status
    ABORT = "abort"  # Abort mission

    # Unknown
    UNKNOWN = "unknown"


# =============================================================================
# PARSED COMMAND
# =============================================================================


@dataclass
class SwarmCommand:
    """Parsed swarm command with parameters."""

    # Command type
    command_type: SwarmCommandType

    # Target swarm (e.g., "Alpha", "Bravo", or "all")
    target_swarm: str = "all"

    # Parameters
    heading: float | None = None  # degrees
    mach: float | None = None  # target Mach number
    altitude: float | None = None  # target altitude (meters)
    position: tuple[float, float, float] | None = None  # lat, lon, alt

    # Formation
    formation: str | None = None  # "wedge", "line", "echelon"
    spacing: float | None = None  # meters

    # Timing
    delay: float | None = None  # seconds

    # Confidence
    confidence: float = 1.0

    # Original text
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command_type.value,
            "target_swarm": self.target_swarm,
            "heading": self.heading,
            "mach": self.mach,
            "altitude": self.altitude,
            "position": self.position,
            "formation": self.formation,
            "spacing": self.spacing,
            "delay": self.delay,
            "confidence": self.confidence,
        }


# =============================================================================
# SWARM COMMAND PARSER
# =============================================================================


class SwarmCommandParser:
    """
    Parse natural language commands for swarm control.

    Examples:
        "Swarm Alpha, intercept trajectory vector 3-5-0 at Mach 8"
        "All units, formation wedge, spacing 200 meters"
        "Bravo flight, hold position"
        "Attack vector 180, altitude angels 30"
    """

    # Swarm name patterns
    SWARM_PATTERNS = {
        r"\balpha\b": "Alpha",
        r"\bbravo\b": "Bravo",
        r"\bcharlie\b": "Charlie",
        r"\bdelta\b": "Delta",
        r"\becho\b": "Echo",
        r"\ball\s*units?\b": "all",
        r"\beveryone\b": "all",
        r"\bswarm\b": "all",  # Default to all if just "swarm"
    }

    # Command patterns
    COMMAND_PATTERNS = {
        r"\bintercept\b": SwarmCommandType.INTERCEPT,
        r"\bproceed\b|\bcontinue\b|\badvance\b": SwarmCommandType.PROCEED,
        r"\bhold\b|\bstation\b|\bhover\b": SwarmCommandType.HOLD,
        r"\breturn\s*to\s*base\b|\brtb\b|\bcome\s*home\b": SwarmCommandType.RTB,
        r"\bformation\b|\bform\s*up\b": SwarmCommandType.FORMATION,
        r"\bspread\b|\bwiden\b": SwarmCommandType.SPREAD,
        r"\btighten\b|\bclose\s*up\b": SwarmCommandType.TIGHTEN,
        r"\battack\b|\bengage\b|\bfire\b": SwarmCommandType.ATTACK,
        r"\bevade\b|\bbreak\b|\bdefensive\b": SwarmCommandType.EVADE,
        r"\bdefend\b|\bprotect\b": SwarmCommandType.DEFEND,
        r"\bstatus\b|\breport\b|\bsitrep\b": SwarmCommandType.STATUS,
        r"\babort\b|\bcancel\b|\bstand\s*down\b": SwarmCommandType.ABORT,
    }

    # Formation patterns
    FORMATION_PATTERNS = {
        r"\bwedge\b|\bvee\b|\bv\s*formation\b": "wedge",
        r"\bline\b|\babreast\b": "line",
        r"\bechelon\b|\bdiagonal\b": "echelon",
        r"\bcolumn\b|\btrail\b": "column",
        r"\bdiamond\b": "diamond",
    }

    # Heading patterns (e.g., "vector 3-5-0" = 350 degrees)
    HEADING_PATTERN = r"(?:vector|heading|bearing)\s*(\d[\d\-\s]*\d?)"

    # Mach patterns
    MACH_PATTERN = r"mach\s*(\d+(?:\.\d+)?)"

    # Altitude patterns
    ALTITUDE_PATTERNS = [
        (
            r"angels?\s*(\d+)",
            lambda m: float(m.group(1)) * 1000 * 0.3048,
        ),  # Angels = 1000ft
        (r"altitude\s*(\d+)\s*(?:m|meters?)", lambda m: float(m.group(1))),
        (r"altitude\s*(\d+)\s*(?:ft|feet)", lambda m: float(m.group(1)) * 0.3048),
        (
            r"(\d+)\s*(?:km|kilometers?)\s*(?:altitude|alt)?",
            lambda m: float(m.group(1)) * 1000,
        ),
    ]

    # Spacing patterns
    SPACING_PATTERN = r"spacing\s*(\d+)\s*(?:m|meters?)?"

    def parse(self, text: str) -> SwarmCommand:
        """
        Parse a natural language command into a SwarmCommand.

        Args:
            text: Natural language command string

        Returns:
            Parsed SwarmCommand
        """
        text_lower = text.lower()

        # Extract swarm target
        target_swarm = "all"
        for pattern, swarm_name in self.SWARM_PATTERNS.items():
            if re.search(pattern, text_lower):
                target_swarm = swarm_name
                break

        # Extract command type
        command_type = SwarmCommandType.UNKNOWN
        for pattern, cmd_type in self.COMMAND_PATTERNS.items():
            if re.search(pattern, text_lower):
                command_type = cmd_type
                break

        # Extract heading
        heading = None
        heading_match = re.search(self.HEADING_PATTERN, text_lower)
        if heading_match:
            heading_str = heading_match.group(1)
            # Parse "3-5-0" or "350" format
            heading_str = re.sub(r"[\s\-]", "", heading_str)
            try:
                heading = float(heading_str)
                if heading > 360:
                    heading = heading % 360
            except ValueError:
                pass

        # Extract Mach number
        mach = None
        mach_match = re.search(self.MACH_PATTERN, text_lower)
        if mach_match:
            mach = float(mach_match.group(1))

        # Extract altitude
        altitude = None
        for pattern, converter in self.ALTITUDE_PATTERNS:
            alt_match = re.search(pattern, text_lower)
            if alt_match:
                altitude = converter(alt_match)
                break

        # Extract formation
        formation = None
        for pattern, form_name in self.FORMATION_PATTERNS.items():
            if re.search(pattern, text_lower):
                formation = form_name
                break

        # Extract spacing
        spacing = None
        spacing_match = re.search(self.SPACING_PATTERN, text_lower)
        if spacing_match:
            spacing = float(spacing_match.group(1))

        # Calculate confidence based on matches
        confidence = 1.0 if command_type != SwarmCommandType.UNKNOWN else 0.3
        if heading is not None or mach is not None:
            confidence = min(confidence + 0.2, 1.0)

        return SwarmCommand(
            command_type=command_type,
            target_swarm=target_swarm,
            heading=heading,
            mach=mach,
            altitude=altitude,
            formation=formation,
            spacing=spacing,
            confidence=confidence,
            raw_text=text,
        )


# =============================================================================
# SWARM COMMANDER
# =============================================================================


class SwarmCommander:
    """
    High-level swarm command and control interface.

    Integrates:
    - Natural language parsing
    - Trajectory planning (Phase 3)
    - Agent control (Phase 4)

    Usage:
        commander = SwarmCommander()
        result = commander.execute("Swarm Alpha, intercept vector 350 at Mach 8")

        # Result contains:
        # - Parsed command
        # - Generated trajectory
        # - Agent assignments
    """

    def __init__(self):
        self.parser = SwarmCommandParser()
        self.command_history: list[SwarmCommand] = []

        # Swarm state (would be connected to actual agents)
        self.swarms: dict[str, dict[str, Any]] = {
            "Alpha": {"entities": [1, 2, 3], "formation": "wedge"},
            "Bravo": {"entities": [4, 5, 6], "formation": "line"},
            "Charlie": {"entities": [7, 8], "formation": "echelon"},
        }

    def execute(self, command_text: str) -> dict[str, Any]:
        """
        Execute a natural language command.

        Args:
            command_text: Natural language command

        Returns:
            Execution result with status, parsed command, and actions
        """
        # Parse the command
        command = self.parser.parse(command_text)
        self.command_history.append(command)

        # Execute based on command type
        result = {
            "status": "success",
            "command": command.to_dict(),
            "actions": [],
            "message": "",
        }

        if command.command_type == SwarmCommandType.UNKNOWN:
            result["status"] = "error"
            result["message"] = f"Unknown command: {command_text}"
            return result

        # Get target swarms
        target_swarms = self._get_target_swarms(command.target_swarm)

        if command.command_type == SwarmCommandType.INTERCEPT:
            result = self._execute_intercept(command, target_swarms)

        elif command.command_type == SwarmCommandType.FORMATION:
            result = self._execute_formation(command, target_swarms)

        elif command.command_type == SwarmCommandType.HOLD:
            result = self._execute_hold(command, target_swarms)

        elif command.command_type == SwarmCommandType.STATUS:
            result = self._execute_status(target_swarms)

        elif command.command_type == SwarmCommandType.ATTACK:
            result = self._execute_attack(command, target_swarms)

        else:
            result["actions"].append(
                {
                    "type": command.command_type.value,
                    "targets": target_swarms,
                }
            )
            result["message"] = f"Command acknowledged: {command.command_type.value}"

        return result

    def _get_target_swarms(self, target: str) -> list[str]:
        """Get list of swarm names matching target."""
        if target == "all":
            return list(self.swarms.keys())
        elif target in self.swarms:
            return [target]
        else:
            return []

    def _execute_intercept(
        self,
        command: SwarmCommand,
        targets: list[str],
    ) -> dict[str, Any]:
        """Execute intercept command."""
        actions = []

        for swarm_name in targets:
            swarm = self.swarms.get(swarm_name, {})
            entities = swarm.get("entities", [])

            action = {
                "type": "intercept",
                "swarm": swarm_name,
                "entities": entities,
                "heading": command.heading or 0.0,
                "mach": command.mach or 10.0,
                "altitude": command.altitude or 30000.0,
            }
            actions.append(action)

        # In a real system, this would:
        # 1. Calculate Phase 3 trajectory to intercept point
        # 2. Assign trajectory to swarm agents
        # 3. Update agent waypoints

        heading_str = f"{command.heading:.0f}" if command.heading else "current"
        mach_str = f"Mach {command.mach:.1f}" if command.mach else "cruise"

        return {
            "status": "success",
            "command": command.to_dict(),
            "actions": actions,
            "message": f"{', '.join(targets)} intercepting on heading {heading_str} at {mach_str}",
        }

    def _execute_formation(
        self,
        command: SwarmCommand,
        targets: list[str],
    ) -> dict[str, Any]:
        """Execute formation change command."""
        actions = []

        for swarm_name in targets:
            if swarm_name in self.swarms:
                old_formation = self.swarms[swarm_name].get("formation", "none")
                new_formation = command.formation or "wedge"
                self.swarms[swarm_name]["formation"] = new_formation

                action = {
                    "type": "formation",
                    "swarm": swarm_name,
                    "old_formation": old_formation,
                    "new_formation": new_formation,
                    "spacing": command.spacing or 100.0,
                }
                actions.append(action)

        formation_name = command.formation or "wedge"
        spacing_str = f" with {command.spacing:.0f}m spacing" if command.spacing else ""

        return {
            "status": "success",
            "command": command.to_dict(),
            "actions": actions,
            "message": f"{', '.join(targets)} forming {formation_name}{spacing_str}",
        }

    def _execute_hold(
        self,
        command: SwarmCommand,
        targets: list[str],
    ) -> dict[str, Any]:
        """Execute hold position command."""
        return {
            "status": "success",
            "command": command.to_dict(),
            "actions": [{"type": "hold", "swarm": t} for t in targets],
            "message": f"{', '.join(targets)} holding position",
        }

    def _execute_status(self, targets: list[str]) -> dict[str, Any]:
        """Execute status report command."""
        status_reports = []

        for swarm_name in targets:
            swarm = self.swarms.get(swarm_name, {})
            report = {
                "swarm": swarm_name,
                "entity_count": len(swarm.get("entities", [])),
                "formation": swarm.get("formation", "unknown"),
                "status": "operational",
            }
            status_reports.append(report)

        return {
            "status": "success",
            "command": {
                "command": "status",
                "target_swarm": targets[0] if len(targets) == 1 else "all",
            },
            "actions": status_reports,
            "message": f"Status report for {len(targets)} swarm(s)",
        }

    def _execute_attack(
        self,
        command: SwarmCommand,
        targets: list[str],
    ) -> dict[str, Any]:
        """Execute attack command."""
        actions = []

        for swarm_name in targets:
            swarm = self.swarms.get(swarm_name, {})
            action = {
                "type": "attack",
                "swarm": swarm_name,
                "entities": swarm.get("entities", []),
                "heading": command.heading,
                "altitude": command.altitude,
            }
            actions.append(action)

        return {
            "status": "success",
            "command": command.to_dict(),
            "actions": actions,
            "message": f"{', '.join(targets)} engaging target",
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SWARM COMMAND INTERFACE TEST")
    print("=" * 60)

    commander = SwarmCommander()

    test_commands = [
        "Swarm Alpha, intercept trajectory vector 3-5-0 at Mach 8",
        "All units, formation wedge, spacing 200 meters",
        "Bravo flight, hold position",
        "Charlie, attack vector 180, altitude angels 30",
        "Status report",
        "Alpha, proceed to waypoint",
        "Everyone form up echelon",
    ]

    print("\nTesting command parsing:\n")

    for cmd in test_commands:
        print(f'Command: "{cmd}"')
        result = commander.execute(cmd)
        print(f"  Status: {result['status']}")
        print(f"  Message: {result['message']}")
        if result["actions"]:
            print(f"  Actions: {len(result['actions'])}")
        print()

    print("=" * 60)
    print("✓ Exit Gate: Commands parsed and executed")
    print("=" * 60)
