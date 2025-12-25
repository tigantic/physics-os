"""
Intent Parser
==============

Parse natural language queries into structured operations.

Uses pattern matching and entity extraction to convert
human intent into executable field queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
from enum import Enum

from .query import FieldQuery, Predicate, PredicateOp, Selector, Aggregator


# =============================================================================
# INTENT TYPES
# =============================================================================

class IntentType(Enum):
    """Type of user intent."""
    # Queries (read-only)
    QUERY_VALUE = "query_value"         # What is the max velocity?
    QUERY_LOCATION = "query_location"   # Where is the pressure highest?
    QUERY_REGION = "query_region"       # Show me the inlet region
    QUERY_COMPARE = "query_compare"     # Compare A vs B
    QUERY_TREND = "query_trend"         # Is velocity increasing?
    
    # Actions (modify state)
    ACTION_SET = "action_set"           # Set velocity to 1.0
    ACTION_INCREASE = "action_increase" # Increase pressure
    ACTION_DECREASE = "action_decrease" # Decrease temperature
    ACTION_OPTIMIZE = "action_optimize" # Optimize for drag
    ACTION_CONSTRAIN = "action_constrain"  # Keep pressure < 100
    
    # Control
    CONTROL_RUN = "control_run"         # Run simulation
    CONTROL_STOP = "control_stop"       # Stop simulation
    CONTROL_RESET = "control_reset"     # Reset to initial
    CONTROL_SAVE = "control_save"       # Save state
    
    # Information
    INFO_STATUS = "info_status"         # Current status?
    INFO_HELP = "info_help"             # How do I...?
    
    # Unknown
    UNKNOWN = "unknown"


# =============================================================================
# ENTITIES
# =============================================================================

@dataclass
class Entity:
    """Extracted entity from text."""
    entity_type: str  # "field", "region", "value", "operator", etc.
    value: Any
    text: str  # Original text span
    start: int = 0
    end: int = 0
    confidence: float = 1.0


@dataclass
class EntityExtractor:
    """
    Extract entities from natural language text.
    
    Uses pattern matching to identify:
    - Field names (velocity, pressure, temperature, etc.)
    - Regions (inlet, outlet, wall, center, etc.)
    - Numeric values
    - Operators (max, min, average, etc.)
    - Comparisons (greater than, less than, etc.)
    """
    
    # Known field names
    field_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        "velocity": ["velocity", "velocities", "vel", "speed", "u", "v", "w"],
        "pressure": ["pressure", "pressures", "p", "press"],
        "temperature": ["temperature", "temp", "t", "heat"],
        "density": ["density", "rho", "ρ"],
        "vorticity": ["vorticity", "vort", "curl", "rotation"],
        "mach": ["mach", "mach number", "ma"],
        "reynolds": ["reynolds", "re", "reynolds number"],
    })
    
    # Known regions
    region_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        "inlet": ["inlet", "inflow", "entrance", "entry"],
        "outlet": ["outlet", "outflow", "exit"],
        "wall": ["wall", "walls", "boundary", "surface"],
        "center": ["center", "centre", "middle", "core"],
        "left": ["left", "left side"],
        "right": ["right", "right side"],
        "top": ["top", "upper"],
        "bottom": ["bottom", "lower"],
    })
    
    # Operators
    operator_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        "max": ["max", "maximum", "highest", "largest", "peak"],
        "min": ["min", "minimum", "lowest", "smallest"],
        "mean": ["mean", "average", "avg"],
        "sum": ["sum", "total", "integrate"],
        "gradient": ["gradient", "derivative", "rate of change"],
    })
    
    # Comparison patterns
    comparison_patterns: Dict[str, str] = field(default_factory=lambda: {
        r"greater than|more than|above|over|>": "gt",
        r"less than|below|under|<": "lt",
        r"equal to|equals|=": "eq",
        r"between": "range",
    })
    
    def extract_all(self, text: str) -> List[Entity]:
        """Extract all entities from text."""
        entities = []
        text_lower = text.lower()
        
        # Extract fields
        for field_name, patterns in self.field_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    idx = text_lower.index(pattern)
                    entities.append(Entity(
                        entity_type="field",
                        value=field_name,
                        text=pattern,
                        start=idx,
                        end=idx + len(pattern),
                    ))
                    break
        
        # Extract regions
        for region_name, patterns in self.region_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    idx = text_lower.index(pattern)
                    entities.append(Entity(
                        entity_type="region",
                        value=region_name,
                        text=pattern,
                        start=idx,
                        end=idx + len(pattern),
                    ))
                    break
        
        # Extract operators
        for op_name, patterns in self.operator_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    idx = text_lower.index(pattern)
                    entities.append(Entity(
                        entity_type="operator",
                        value=op_name,
                        text=pattern,
                        start=idx,
                        end=idx + len(pattern),
                    ))
                    break
        
        # Extract numbers
        for match in re.finditer(r'-?\d+\.?\d*', text):
            entities.append(Entity(
                entity_type="number",
                value=float(match.group()),
                text=match.group(),
                start=match.start(),
                end=match.end(),
            ))
        
        # Extract comparisons
        for pattern, comp_type in self.comparison_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                entities.append(Entity(
                    entity_type="comparison",
                    value=comp_type,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                ))
        
        return entities
    
    def extract_field(self, text: str) -> Optional[str]:
        """Extract primary field from text."""
        for entity in self.extract_all(text):
            if entity.entity_type == "field":
                return entity.value
        return None
    
    def extract_region(self, text: str) -> Optional[str]:
        """Extract region from text."""
        for entity in self.extract_all(text):
            if entity.entity_type == "region":
                return entity.value
        return None


# =============================================================================
# PARSE RESULT
# =============================================================================

@dataclass
class ParseResult:
    """
    Result of parsing natural language intent.
    """
    text: str  # Original text
    intent_type: IntentType
    
    # Extracted components
    field_name: Optional[str] = None
    region: Optional[str] = None
    operator: Optional[str] = None
    value: Optional[float] = None
    comparison: Optional[str] = None
    
    # Generated query (if applicable)
    query: Optional[FieldQuery] = None
    
    # All entities
    entities: List[Entity] = field(default_factory=list)
    
    # Parsing confidence
    confidence: float = 0.0
    
    # Error message if parsing failed
    error: Optional[str] = None
    
    @property
    def is_query(self) -> bool:
        return self.intent_type.value.startswith("query_")
    
    @property
    def is_action(self) -> bool:
        return self.intent_type.value.startswith("action_")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "intent_type": self.intent_type.value,
            "field_name": self.field_name,
            "region": self.region,
            "operator": self.operator,
            "value": self.value,
            "confidence": self.confidence,
            "error": self.error,
        }


# =============================================================================
# INTENT PARSER
# =============================================================================

class IntentParser:
    """
    Parse natural language into structured intents.
    
    Example:
        parser = IntentParser()
        
        result = parser.parse("What is the maximum velocity at the inlet?")
        # result.intent_type == IntentType.QUERY_VALUE
        # result.field_name == "velocity"
        # result.region == "inlet"
        # result.operator == "max"
        
        result = parser.parse("Increase the pressure near the wall")
        # result.intent_type == IntentType.ACTION_INCREASE
        # result.field_name == "pressure"
        # result.region == "wall"
    """
    
    def __init__(self):
        self.extractor = EntityExtractor()
        
        # Intent patterns (regex -> IntentType)
        self.intent_patterns = [
            # Queries
            (r"what is|show me|display|get|find|query", IntentType.QUERY_VALUE),
            (r"where is|where are|locate|find.*location", IntentType.QUERY_LOCATION),
            (r"compare|difference|versus|vs", IntentType.QUERY_COMPARE),
            (r"is.*increasing|is.*decreasing|trend", IntentType.QUERY_TREND),
            
            # Actions
            (r"set|assign|make.*equal", IntentType.ACTION_SET),
            (r"increase|raise|boost|amplify|enhance", IntentType.ACTION_INCREASE),
            (r"decrease|reduce|lower|diminish", IntentType.ACTION_DECREASE),
            (r"optimize|maximize|minimize", IntentType.ACTION_OPTIMIZE),
            (r"constrain|limit|keep|maintain", IntentType.ACTION_CONSTRAIN),
            
            # Control
            (r"run|start|begin|simulate|step", IntentType.CONTROL_RUN),
            (r"stop|pause|halt", IntentType.CONTROL_STOP),
            (r"reset|restart|initialize", IntentType.CONTROL_RESET),
            (r"save|store|checkpoint", IntentType.CONTROL_SAVE),
            
            # Info
            (r"status|state|current", IntentType.INFO_STATUS),
            (r"help|how do|what can|explain", IntentType.INFO_HELP),
        ]
    
    def parse(self, text: str) -> ParseResult:
        """
        Parse natural language text into structured intent.
        
        Args:
            text: Natural language input
            
        Returns:
            ParseResult with intent and entities
        """
        text_lower = text.lower().strip()
        
        # Extract entities
        entities = self.extractor.extract_all(text)
        
        # Determine intent type
        intent_type = self._classify_intent(text_lower)
        
        # Extract components
        field_name = None
        region = None
        operator = None
        value = None
        comparison = None
        
        for entity in entities:
            if entity.entity_type == "field":
                field_name = entity.value
            elif entity.entity_type == "region":
                region = entity.value
            elif entity.entity_type == "operator":
                operator = entity.value
            elif entity.entity_type == "number":
                value = entity.value
            elif entity.entity_type == "comparison":
                comparison = entity.value
        
        # Build query if applicable
        query = None
        if intent_type in [IntentType.QUERY_VALUE, IntentType.QUERY_LOCATION]:
            query = self._build_query(field_name, region, operator, value, comparison)
        
        # Calculate confidence
        confidence = self._calculate_confidence(entities, intent_type)
        
        return ParseResult(
            text=text,
            intent_type=intent_type,
            field_name=field_name,
            region=region,
            operator=operator,
            value=value,
            comparison=comparison,
            query=query,
            entities=entities,
            confidence=confidence,
        )
    
    def _classify_intent(self, text: str) -> IntentType:
        """Classify text into intent type."""
        for pattern, intent_type in self.intent_patterns:
            if re.search(pattern, text):
                return intent_type
        return IntentType.UNKNOWN
    
    def _build_query(
        self,
        field_name: Optional[str],
        region: Optional[str],
        operator: Optional[str],
        value: Optional[float],
        comparison: Optional[str],
    ) -> Optional[FieldQuery]:
        """Build FieldQuery from parsed components."""
        if not field_name:
            return None
        
        query = FieldQuery(field_name)
        
        # Add comparison predicate
        if value is not None and comparison:
            if comparison == "gt":
                query = query.where(Predicate.gt(value))
            elif comparison == "lt":
                query = query.where(Predicate.lt(value))
        
        # Add aggregation
        if operator:
            query = query.aggregate(operator)
        
        return query
    
    def _calculate_confidence(
        self,
        entities: List[Entity],
        intent_type: IntentType,
    ) -> float:
        """Calculate parsing confidence."""
        if intent_type == IntentType.UNKNOWN:
            return 0.0
        
        # Base confidence
        confidence = 0.5
        
        # Boost for found entities
        has_field = any(e.entity_type == "field" for e in entities)
        has_operator = any(e.entity_type == "operator" for e in entities)
        has_region = any(e.entity_type == "region" for e in entities)
        
        if has_field:
            confidence += 0.2
        if has_operator:
            confidence += 0.15
        if has_region:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def suggest_completions(self, partial: str) -> List[str]:
        """Suggest completions for partial input."""
        suggestions = []
        partial_lower = partial.lower()
        
        # Field suggestions
        if "vel" in partial_lower or "pres" in partial_lower:
            suggestions.append(partial + " at the inlet")
            suggestions.append(partial + " near the wall")
        
        # Action suggestions
        if "incr" in partial_lower:
            suggestions.append("increase velocity")
            suggestions.append("increase pressure")
        
        if "max" in partial_lower or "get" in partial_lower:
            suggestions.append("What is the maximum velocity?")
            suggestions.append("Show maximum pressure at inlet")
        
        return suggestions[:5]
