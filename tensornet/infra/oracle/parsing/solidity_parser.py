"""
Solidity Parser using tree-sitter.

Extracts Contract, Function, StateVariable structures from Solidity source,
along with Control Flow Graphs (CFG), Data Flow Graphs (DFG), and Call Graphs.

This is ORACLE's eyes - everything downstream depends on accurate parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from tensornet.infra.oracle.core.types import (
    CFGEdge,
    CFGNode,
    CallEdge,
    CallGraph,
    Contract,
    ControlFlowGraph,
    DFGEdge,
    DFGNode,
    DataFlowGraph,
    Event,
    Function,
    Modifier,
    Parameter,
    StateVariable,
)

# Try to import tree-sitter-solidity, fall back to regex parsing if not available
try:
    import tree_sitter_solidity as ts_sol
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


class SolidityParser:
    """
    Parse Solidity source code into structured Contract objects.
    
    Uses tree-sitter-solidity for AST parsing when available,
    falls back to regex-based parsing otherwise.
    """
    
    def __init__(self):
        """Initialize the parser."""
        if TREE_SITTER_AVAILABLE:
            self.parser = ts_sol.Parser()
            self._use_tree_sitter = True
        else:
            self._use_tree_sitter = False
    
    def parse(self, source: str, file_path: Optional[str] = None) -> list[Contract]:
        """
        Parse Solidity source into Contract structures.
        
        Args:
            source: Solidity source code
            file_path: Optional file path for reference
            
        Returns:
            List of Contract objects found in the source
        """
        if self._use_tree_sitter:
            return self._parse_tree_sitter(source, file_path)
        return self._parse_regex(source, file_path)
    
    def _parse_tree_sitter(self, source: str, file_path: Optional[str]) -> list[Contract]:
        """Parse using tree-sitter AST."""
        tree = self.parser.parse(bytes(source, "utf8"))
        contracts = []
        
        for child in tree.root_node.children:
            if child.type == "contract_declaration":
                contract = self._extract_contract_ts(child, source)
                contract.file_path = file_path
                contracts.append(contract)
            elif child.type == "interface_declaration":
                contract = self._extract_contract_ts(child, source)
                contract.kind = "interface"
                contract.file_path = file_path
                contracts.append(contract)
            elif child.type == "library_declaration":
                contract = self._extract_contract_ts(child, source)
                contract.kind = "library"
                contract.file_path = file_path
                contracts.append(contract)
        
        return contracts
    
    def _extract_contract_ts(self, node, source: str) -> Contract:
        """Extract contract from tree-sitter node."""
        name = ""
        inherits = []
        functions = []
        state_vars = []
        events = []
        modifiers = []
        is_abstract = False
        
        for child in node.children:
            if child.type == "identifier":
                name = source[child.start_byte:child.end_byte]
            elif child.type == "abstract":
                is_abstract = True
            elif child.type == "inheritance_specifier":
                inherits.extend(self._extract_inheritance(child, source))
            elif child.type == "contract_body":
                for member in child.children:
                    if member.type == "function_definition":
                        functions.append(self._extract_function_ts(member, source))
                    elif member.type == "state_variable_declaration":
                        state_vars.append(self._extract_state_var_ts(member, source))
                    elif member.type == "event_definition":
                        events.append(self._extract_event_ts(member, source))
                    elif member.type == "modifier_definition":
                        modifiers.append(self._extract_modifier_ts(member, source))
        
        return Contract(
            name=name,
            is_abstract=is_abstract,
            inherits=inherits,
            functions=functions,
            state_variables=state_vars,
            events=events,
            modifiers=modifiers,
            source=source,
        )
    
    def _extract_inheritance(self, node, source: str) -> list[str]:
        """Extract inherited contract names."""
        inherits = []
        for child in node.children:
            if child.type == "user_defined_type":
                inherits.append(source[child.start_byte:child.end_byte])
        return inherits
    
    def _extract_function_ts(self, node, source: str) -> Function:
        """Extract function from tree-sitter node."""
        name = ""
        visibility = "internal"
        mutability = "nonpayable"
        parameters = []
        returns = []
        modifiers = []
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        
        for child in node.children:
            if child.type == "identifier":
                name = source[child.start_byte:child.end_byte]
            elif child.type in ("public", "external", "internal", "private"):
                visibility = child.type
            elif child.type in ("pure", "view", "payable"):
                mutability = child.type
            elif child.type == "parameter_list":
                parameters = self._extract_parameters(child, source)
            elif child.type == "return_type_definition":
                returns = self._extract_returns(child, source)
            elif child.type == "modifier_invocation":
                mod_name = ""
                for subchild in child.children:
                    if subchild.type == "identifier":
                        mod_name = source[subchild.start_byte:subchild.end_byte]
                if mod_name:
                    modifiers.append(mod_name)
        
        func_source = source[node.start_byte:node.end_byte]
        
        return Function(
            name=name,
            visibility=visibility,
            mutability=mutability,
            parameters=parameters,
            returns=returns,
            modifiers=modifiers,
            source=func_source,
            start_line=start_line,
            end_line=end_line,
        )
    
    def _extract_parameters(self, node, source: str) -> list[Parameter]:
        """Extract function parameters."""
        params = []
        for child in node.children:
            if child.type == "parameter":
                param = self._extract_single_param(child, source)
                if param:
                    params.append(param)
        return params
    
    def _extract_single_param(self, node, source: str) -> Optional[Parameter]:
        """Extract a single parameter."""
        type_name = ""
        name = ""
        
        for child in node.children:
            if child.type in ("type_name", "mapping", "array_type", "user_defined_type", 
                              "elementary_type_name"):
                type_name = source[child.start_byte:child.end_byte]
            elif child.type == "identifier":
                name = source[child.start_byte:child.end_byte]
        
        if type_name:
            return Parameter(name=name, type_name=type_name)
        return None
    
    def _extract_returns(self, node, source: str) -> list[Parameter]:
        """Extract return types."""
        returns = []
        for child in node.children:
            if child.type == "parameter_list":
                returns = self._extract_parameters(child, source)
        return returns
    
    def _extract_state_var_ts(self, node, source: str) -> StateVariable:
        """Extract state variable from tree-sitter node."""
        type_name = ""
        name = ""
        visibility = "internal"
        initial_value = None
        
        for child in node.children:
            if child.type in ("type_name", "mapping", "array_type", "user_defined_type",
                              "elementary_type_name"):
                type_name = source[child.start_byte:child.end_byte]
            elif child.type == "identifier":
                name = source[child.start_byte:child.end_byte]
            elif child.type in ("public", "private", "internal"):
                visibility = child.type
            elif child.type == "expression":
                initial_value = source[child.start_byte:child.end_byte]
        
        return StateVariable(
            name=name,
            type_name=type_name,
            visibility=visibility,
            initial_value=initial_value,
            line=node.start_point[0] + 1,
        )
    
    def _extract_event_ts(self, node, source: str) -> Event:
        """Extract event from tree-sitter node."""
        name = ""
        parameters = []
        
        for child in node.children:
            if child.type == "identifier":
                name = source[child.start_byte:child.end_byte]
            elif child.type == "event_parameter_list":
                for param_child in child.children:
                    if param_child.type == "event_parameter":
                        param = self._extract_event_param(param_child, source)
                        if param:
                            parameters.append(param)
        
        return Event(name=name, parameters=parameters, line=node.start_point[0] + 1)
    
    def _extract_event_param(self, node, source: str) -> Optional[Parameter]:
        """Extract event parameter."""
        type_name = ""
        name = ""
        indexed = False
        
        for child in node.children:
            if child.type in ("type_name", "elementary_type_name"):
                type_name = source[child.start_byte:child.end_byte]
            elif child.type == "identifier":
                name = source[child.start_byte:child.end_byte]
            elif child.type == "indexed":
                indexed = True
        
        if type_name:
            return Parameter(name=name, type_name=type_name, indexed=indexed)
        return None
    
    def _extract_modifier_ts(self, node, source: str) -> Modifier:
        """Extract modifier from tree-sitter node."""
        name = ""
        parameters = []
        
        for child in node.children:
            if child.type == "identifier":
                name = source[child.start_byte:child.end_byte]
            elif child.type == "parameter_list":
                parameters = self._extract_parameters(child, source)
        
        return Modifier(
            name=name,
            parameters=parameters,
            source=source[node.start_byte:node.end_byte],
            line=node.start_point[0] + 1,
        )
    
    # =========================================================================
    # Regex-based fallback parser
    # =========================================================================
    
    def _parse_regex(self, source: str, file_path: Optional[str]) -> list[Contract]:
        """Parse using regex patterns (fallback when tree-sitter unavailable)."""
        contracts = []
        
        # Find all contract/interface/library declarations
        contract_pattern = r"(abstract\s+)?(contract|interface|library)\s+(\w+)(\s+is\s+([^{]+))?\s*\{"
        
        for match in re.finditer(contract_pattern, source):
            is_abstract = match.group(1) is not None
            kind = match.group(2)
            name = match.group(3)
            inherits_str = match.group(5) or ""
            inherits = [s.strip() for s in inherits_str.split(",") if s.strip()]
            
            # Find the contract body
            start = match.end() - 1
            body_end = self._find_matching_brace(source, start)
            body = source[start:body_end + 1]
            
            contract = Contract(
                name=name,
                kind=kind,
                is_abstract=is_abstract,
                inherits=inherits,
                functions=self._extract_functions_regex(body),
                state_variables=self._extract_state_vars_regex(body),
                events=self._extract_events_regex(body),
                source=source,
                file_path=file_path,
            )
            contracts.append(contract)
        
        return contracts
    
    def _find_matching_brace(self, source: str, start: int) -> int:
        """Find the matching closing brace."""
        depth = 0
        i = start
        while i < len(source):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return len(source) - 1
    
    def _extract_functions_regex(self, body: str) -> list[Function]:
        """Extract functions using regex."""
        functions = []
        
        # Find function signatures first
        func_pattern = r"function\s+(\w+)\s*\(([^)]*)\)\s*(public|external|internal|private)?\s*(pure|view|payable)?\s*(?:returns\s*\([^)]*\))?\s*(?:[\w\s,()]*)\s*\{"
        
        for match in re.finditer(func_pattern, body):
            name = match.group(1)
            params_str = match.group(2)
            visibility = match.group(3) or "internal"
            mutability = match.group(4) or "nonpayable"
            
            # Parse parameters
            parameters = []
            if params_str.strip():
                for param in params_str.split(","):
                    parts = param.strip().split()
                    if len(parts) >= 2:
                        parameters.append(Parameter(name=parts[-1], type_name=" ".join(parts[:-1])))
                    elif len(parts) == 1:
                        parameters.append(Parameter(name="", type_name=parts[0]))
            
            # Find the full function body including braces
            func_start = match.start()
            brace_start = match.end() - 1
            brace_end = self._find_matching_brace(body, brace_start)
            full_source = body[func_start:brace_end + 1]
            
            functions.append(Function(
                name=name,
                visibility=visibility,
                mutability=mutability,
                parameters=parameters,
                source=full_source,
            ))
        
        return functions
    
    def _extract_state_vars_regex(self, body: str) -> list[StateVariable]:
        """Extract state variables using regex."""
        state_vars = []
        
        # Match common state variable patterns
        var_pattern = r"(mapping\([^)]+\)|[\w\[\]]+)\s+(public|private|internal|constant|immutable)?\s*(\w+)\s*[;=]"
        
        for match in re.finditer(var_pattern, body):
            type_name = match.group(1)
            visibility = match.group(2) or "internal"
            name = match.group(3)
            
            # Skip if this looks like a function parameter
            if visibility in ("constant", "immutable"):
                visibility = "internal"
            
            state_vars.append(StateVariable(
                name=name,
                type_name=type_name,
                visibility=visibility,
            ))
        
        return state_vars
    
    def _extract_events_regex(self, body: str) -> list[Event]:
        """Extract events using regex."""
        events = []
        
        event_pattern = r"event\s+(\w+)\s*\(([^)]*)\)"
        
        for match in re.finditer(event_pattern, body):
            name = match.group(1)
            params_str = match.group(2)
            
            parameters = []
            if params_str.strip():
                for param in params_str.split(","):
                    param = param.strip()
                    indexed = "indexed" in param
                    param = param.replace("indexed", "").strip()
                    parts = param.split()
                    if len(parts) >= 2:
                        parameters.append(Parameter(name=parts[-1], type_name=parts[0], indexed=indexed))
                    elif len(parts) == 1:
                        parameters.append(Parameter(name="", type_name=parts[0], indexed=indexed))
            
            events.append(Event(name=name, parameters=parameters))
        
        return events
    
    # =========================================================================
    # Graph Extraction
    # =========================================================================
    
    def extract_cfg(self, func: Function) -> ControlFlowGraph:
        """
        Build control flow graph for function.
        
        Nodes: basic blocks (sequences of statements without branches)
        Edges: control flow between blocks
        """
        cfg = ControlFlowGraph(function_name=func.name)
        node_id = 0
        
        # Entry node
        entry = CFGNode(id=node_id, type="entry", source="entry", 
                        line_start=func.start_line, line_end=func.start_line)
        cfg.nodes.append(entry)
        cfg.entry = node_id
        node_id += 1
        
        # Parse function body for control structures
        source = func.source
        
        # Find if statements
        if_pattern = r"\bif\s*\(([^)]+)\)"
        for match in re.finditer(if_pattern, source):
            condition = match.group(1)
            branch = CFGNode(id=node_id, type="branch", source=f"if ({condition})",
                            line_start=0, line_end=0)
            cfg.nodes.append(branch)
            node_id += 1
        
        # Find loops (for, while)
        loop_pattern = r"\b(for|while)\s*\(([^)]+)\)"
        for match in re.finditer(loop_pattern, source):
            loop_type = match.group(1)
            condition = match.group(2)
            loop = CFGNode(id=node_id, type="loop", source=f"{loop_type} ({condition})",
                          line_start=0, line_end=0)
            cfg.nodes.append(loop)
            node_id += 1
        
        # Exit node
        exit_node = CFGNode(id=node_id, type="exit", source="exit",
                           line_start=func.end_line, line_end=func.end_line)
        cfg.nodes.append(exit_node)
        cfg.exit = node_id
        
        # Connect entry to first real node (simplified)
        if len(cfg.nodes) > 2:
            cfg.edges.append(CFGEdge(from_node=cfg.entry, to_node=1))
            cfg.edges.append(CFGEdge(from_node=len(cfg.nodes) - 2, to_node=cfg.exit))
        else:
            cfg.edges.append(CFGEdge(from_node=cfg.entry, to_node=cfg.exit))
        
        return cfg
    
    def extract_dfg(self, func: Function) -> DataFlowGraph:
        """
        Build data flow graph for function.
        
        Tracks definitions and uses of variables.
        """
        dfg = DataFlowGraph(function_name=func.name)
        node_id = 0
        var_defs: dict[str, int] = {}
        
        # Add parameters as initial definitions
        for param in func.parameters:
            node = DFGNode(id=node_id, variable=param.name, 
                          definition_line=func.start_line, is_parameter=True)
            dfg.nodes.append(node)
            var_defs[param.name] = node_id
            node_id += 1
        
        # Find assignments in function body
        assign_pattern = r"(\w+)\s*=[^=]"
        for match in re.finditer(assign_pattern, func.source):
            var_name = match.group(1)
            if var_name not in ("if", "while", "for", "return"):
                node = DFGNode(id=node_id, variable=var_name, definition_line=0)
                dfg.nodes.append(node)
                var_defs[var_name] = node_id
                node_id += 1
        
        # Find uses (simplified - look for variable names after definitions)
        for var_name, def_id in var_defs.items():
            use_pattern = rf"\b{re.escape(var_name)}\b"
            uses = list(re.finditer(use_pattern, func.source))
            for use_match in uses[1:]:  # Skip the definition itself
                # Create use edge
                use_node = DFGNode(id=node_id, variable=f"{var_name}_use", definition_line=0)
                dfg.nodes.append(use_node)
                dfg.edges.append(DFGEdge(from_node=def_id, to_node=node_id, type="use"))
                node_id += 1
        
        return dfg
    
    def extract_call_graph(self, contract: Contract) -> CallGraph:
        """
        Build inter-function call graph.
        
        Tracks which functions call which other functions.
        """
        cg = CallGraph(contract_name=contract.name)
        
        # Build set of function names in this contract
        func_names = {f.name for f in contract.functions}
        
        for func in contract.functions:
            source = func.source
            
            # Find internal calls
            call_pattern = r"\b(\w+)\s*\("
            for match in re.finditer(call_pattern, source):
                callee = match.group(1)
                
                # Skip built-in functions and keywords
                if callee in ("if", "while", "for", "require", "assert", "revert", 
                             "emit", "return", "new", "abi", "keccak256", "sha256"):
                    continue
                
                if callee in func_names:
                    cg.edges.append(CallEdge(
                        caller=func.name,
                        callee=callee,
                        call_type="internal"
                    ))
            
            # Find external calls
            external_pattern = r"(\w+)\.(\w+)\s*\("
            for match in re.finditer(external_pattern, source):
                target = match.group(1)
                method = match.group(2)
                
                # Determine call type
                call_type = "external"
                if ".delegatecall" in source:
                    call_type = "delegatecall"
                
                cg.edges.append(CallEdge(
                    caller=func.name,
                    callee=f"{target}.{method}",
                    call_type=call_type
                ))
        
        return cg
    
    def analyze_function(self, func: Function, contract: Contract) -> None:
        """
        Analyze function for state reads/writes and external calls.
        
        Populates the reads_state, writes_state, and external_calls fields.
        """
        state_var_names = {sv.name for sv in contract.state_variables}
        
        # Find state reads
        for var_name in state_var_names:
            if re.search(rf"\b{re.escape(var_name)}\b", func.source):
                func.reads_state.append(var_name)
        
        # Find state writes (assignments to state variables)
        for var_name in state_var_names:
            if re.search(rf"\b{re.escape(var_name)}\s*=[^=]", func.source):
                func.writes_state.append(var_name)
        
        # Find external calls - multiple patterns
        external_patterns = [
            r"(\w+)\.(\w+)\s*\(",              # target.method(
            r"(\w+)\.(\w+)\s*\{[^}]*\}\s*\(",  # target.method{value: x}(
            r"(\w+)\.(call|delegatecall|staticcall)\s*\{",  # target.call{
            r"(\w+)\.(call|delegatecall|staticcall)\s*\(",  # target.call(
        ]
        
        for pattern in external_patterns:
            for match in re.finditer(pattern, func.source):
                target = match.group(1)
                method = match.group(2)
                if target not in ("abi", "msg", "block", "tx", "type", "address"):
                    call_str = f"{target}.{method}"
                    if call_str not in func.external_calls:
                        func.external_calls.append(call_str)


# Convenience function
def parse_solidity(source: str, file_path: Optional[str] = None) -> list[Contract]:
    """
    Parse Solidity source code.
    
    Args:
        source: Solidity source code
        file_path: Optional file path for reference
        
    Returns:
        List of Contract objects
    """
    parser = SolidityParser()
    return parser.parse(source, file_path)
