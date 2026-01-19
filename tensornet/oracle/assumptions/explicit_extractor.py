"""
Extract explicit assumptions from require/assert statements.

Explicit assumptions are the ones developers KNOW they're making:
- require(amount > 0, "Amount must be positive")
- assert(totalSupply >= 0)
- if (condition) revert CustomError()
"""

from __future__ import annotations

import re
from typing import Optional

from tensornet.oracle.core.types import (
    Assumption,
    AssumptionType,
    Contract,
    Function,
)


class ExplicitExtractor:
    """
    Extract assumptions from require/assert/if-revert patterns.
    
    These are assumptions the developer explicitly encoded.
    """
    
    def __init__(self):
        """Initialize the extractor."""
        self._assumption_id = 0
    
    def extract(self, contract: Contract) -> list[Assumption]:
        """
        Extract all explicit assumptions from a contract.
        
        Args:
            contract: Parsed contract
            
        Returns:
            List of explicit assumptions
        """
        self._assumption_id = 0
        assumptions = []
        
        for func in contract.functions:
            # Find require() statements
            assumptions.extend(self._extract_requires(func))
            
            # Find assert() statements
            assumptions.extend(self._extract_asserts(func))
            
            # Find if-revert patterns
            assumptions.extend(self._extract_if_reverts(func))
            
            # Find modifier conditions
            assumptions.extend(self._extract_modifier_conditions(func, contract))
        
        return assumptions
    
    def _next_id(self) -> str:
        """Generate next assumption ID."""
        self._assumption_id += 1
        return f"E{self._assumption_id:03d}"
    
    def _extract_requires(self, func: Function) -> list[Assumption]:
        """Find all require(condition, message) statements."""
        assumptions = []
        source = func.source
        
        # Match require(condition) or require(condition, "message")
        require_pattern = r'require\s*\(\s*([^,)]+)(?:\s*,\s*["\']([^"\']+)["\'])?\s*\)'
        
        for match in re.finditer(require_pattern, source):
            condition = match.group(1).strip()
            message = match.group(2) or ""
            
            # Convert condition to human-readable assumption
            statement = self._condition_to_statement(condition, message)
            formal = self._condition_to_formal(condition)
            
            assumptions.append(Assumption(
                id=self._next_id(),
                type=AssumptionType.EXPLICIT,
                source=func.name,
                statement=statement,
                formal=formal,
                confidence=1.0,
                revert_message=message,
            ))
        
        return assumptions
    
    def _extract_asserts(self, func: Function) -> list[Assumption]:
        """Find all assert(condition) statements."""
        assumptions = []
        source = func.source
        
        assert_pattern = r'assert\s*\(\s*([^)]+)\s*\)'
        
        for match in re.finditer(assert_pattern, source):
            condition = match.group(1).strip()
            
            statement = f"Invariant: {self._condition_to_statement(condition, '')}"
            formal = self._condition_to_formal(condition)
            
            assumptions.append(Assumption(
                id=self._next_id(),
                type=AssumptionType.EXPLICIT,
                source=func.name,
                statement=statement,
                formal=formal,
                confidence=1.0,
            ))
        
        return assumptions
    
    def _extract_if_reverts(self, func: Function) -> list[Assumption]:
        """Find if(condition) revert patterns."""
        assumptions = []
        source = func.source
        
        # Match: if (condition) revert ... or if (condition) { revert ... }
        if_revert_pattern = r'if\s*\(\s*([^)]+)\s*\)\s*(?:revert|{[^}]*revert)'
        
        for match in re.finditer(if_revert_pattern, source):
            condition = match.group(1).strip()
            
            # The assumption is that condition is FALSE (since revert on true)
            negated = self._negate_condition(condition)
            statement = self._condition_to_statement(negated, "")
            formal = self._condition_to_formal(negated)
            
            assumptions.append(Assumption(
                id=self._next_id(),
                type=AssumptionType.EXPLICIT,
                source=func.name,
                statement=statement,
                formal=formal,
                confidence=1.0,
            ))
        
        return assumptions
    
    def _extract_modifier_conditions(self, func: Function, 
                                     contract: Contract) -> list[Assumption]:
        """Extract conditions from modifiers applied to function."""
        assumptions = []
        
        for mod_name in func.modifiers:
            # Find modifier definition
            mod = None
            for m in contract.modifiers:
                if m.name == mod_name:
                    mod = m
                    break
            
            if mod:
                # Extract requires from modifier
                require_pattern = r'require\s*\(\s*([^,)]+)(?:\s*,\s*["\']([^"\']+)["\'])?\s*\)'
                for match in re.finditer(require_pattern, mod.source):
                    condition = match.group(1).strip()
                    message = match.group(2) or ""
                    
                    statement = f"[modifier {mod_name}] {self._condition_to_statement(condition, message)}"
                    
                    assumptions.append(Assumption(
                        id=self._next_id(),
                        type=AssumptionType.EXPLICIT,
                        source=f"{func.name} (via {mod_name})",
                        statement=statement,
                        formal=self._condition_to_formal(condition),
                        confidence=1.0,
                        revert_message=message,
                    ))
            else:
                # Common modifiers we know about
                if mod_name == "onlyOwner":
                    assumptions.append(Assumption(
                        id=self._next_id(),
                        type=AssumptionType.EXPLICIT,
                        source=func.name,
                        statement="Caller must be the contract owner",
                        formal="msg.sender == owner",
                        confidence=1.0,
                    ))
                elif mod_name == "nonReentrant":
                    assumptions.append(Assumption(
                        id=self._next_id(),
                        type=AssumptionType.EXPLICIT,
                        source=func.name,
                        statement="Function cannot be re-entered during execution",
                        formal="!_locked",
                        confidence=1.0,
                    ))
        
        return assumptions
    
    def _condition_to_statement(self, condition: str, message: str) -> str:
        """Convert Solidity condition to human-readable statement."""
        
        if message:
            return message
        
        # Common pattern translations
        translations = [
            (r'(\w+)\s*>\s*0', r'\1 must be positive'),
            (r'(\w+)\s*>=\s*(\w+)', r'\1 must be at least \2'),
            (r'(\w+)\s*<=\s*(\w+)', r'\1 must be at most \2'),
            (r'(\w+)\s*==\s*(\w+)', r'\1 must equal \2'),
            (r'(\w+)\s*!=\s*(\w+)', r'\1 must not equal \2'),
            (r'(\w+)\s*!=\s*address\(0\)', r'\1 must not be zero address'),
            (r'msg\.sender\s*==\s*(\w+)', r'Caller must be \1'),
            (r'block\.timestamp\s*>=\s*(\w+)', r'Current time must be at least \1'),
            (r'block\.timestamp\s*<=\s*(\w+)', r'Current time must be at most \1'),
            (r'!(\w+)', r'\1 must be false'),
        ]
        
        result = condition
        for pattern, replacement in translations:
            result = re.sub(pattern, replacement, result)
        
        if result == condition:
            return f"Condition: {condition}"
        return result
    
    def _condition_to_formal(self, condition: str) -> str:
        """Convert Solidity condition to formal representation."""
        
        # Replace Solidity operators with formal notation
        formal = condition
        
        replacements = [
            (r'&&', '∧'),
            (r'\|\|', '∨'),
            (r'!', '¬'),
            (r'>=', '≥'),
            (r'<=', '≤'),
            (r'!=', '≠'),
            (r'msg\.sender', 'caller'),
            (r'block\.timestamp', 'now'),
            (r'address\(0\)', '0x0'),
        ]
        
        for pattern, replacement in replacements:
            formal = re.sub(pattern, replacement, formal)
        
        return formal
    
    def _negate_condition(self, condition: str) -> str:
        """Negate a condition (for if-revert patterns)."""
        
        # Simple negation patterns
        if condition.startswith("!"):
            return condition[1:].strip()
        
        # Flip comparison operators
        flips = [
            (r'==', '!='),
            (r'!=', '=='),
            (r'>=', '<'),
            (r'<=', '>'),
            (r'>', '<='),
            (r'<', '>='),
        ]
        
        result = condition
        for pattern, replacement in flips:
            if re.search(pattern, result):
                result = re.sub(pattern, replacement, result, count=1)
                return result
        
        return f"!({condition})"
