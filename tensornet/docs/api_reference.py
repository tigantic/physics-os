"""
API Reference Generator for Project HyperTensor.

This module provides automated extraction and generation of API documentation
from Python source code, including docstrings, type annotations, and signatures.

Features:
    - Parse Python modules recursively
    - Extract docstrings (Google, NumPy, reStructuredText styles)
    - Parse type annotations
    - Generate Markdown and reStructuredText output
    - Support for classes, functions, methods, and attributes
"""

from __future__ import annotations

import ast
import inspect
import importlib
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    get_type_hints,
)


class DocstringStyle(Enum):
    """Docstring formatting style."""
    GOOGLE = auto()
    NUMPY = auto()
    RST = auto()
    AUTO = auto()


@dataclass
class ParameterDoc:
    """Documentation for a function/method parameter.
    
    Attributes:
        name: Parameter name.
        type_annotation: Type annotation string.
        description: Parameter description.
        default: Default value if any.
        optional: Whether parameter is optional.
    """
    name: str
    type_annotation: Optional[str] = None
    description: str = ""
    default: Optional[str] = None
    optional: bool = False
    
    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        type_str = f" (`{self.type_annotation}`)" if self.type_annotation else ""
        default_str = f" Default: `{self.default}`." if self.default else ""
        optional_str = " *Optional.*" if self.optional else ""
        return f"- **{self.name}**{type_str}: {self.description}{default_str}{optional_str}"
    
    def to_rst(self) -> str:
        """Convert to reStructuredText format."""
        type_str = f" ({self.type_annotation})" if self.type_annotation else ""
        return f":param {self.name}: {self.description}{type_str}"


@dataclass
class ReturnDoc:
    """Documentation for a return value.
    
    Attributes:
        type_annotation: Return type annotation.
        description: Description of return value.
    """
    type_annotation: Optional[str] = None
    description: str = ""
    
    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        type_str = f"`{self.type_annotation}`" if self.type_annotation else "value"
        return f"**Returns**: {type_str} - {self.description}"
    
    def to_rst(self) -> str:
        """Convert to reStructuredText format."""
        type_str = f" ({self.type_annotation})" if self.type_annotation else ""
        return f":returns: {self.description}{type_str}"


@dataclass
class RaisesDoc:
    """Documentation for an exception.
    
    Attributes:
        exception: Exception class name.
        description: When this exception is raised.
    """
    exception: str
    description: str = ""
    
    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        return f"- `{self.exception}`: {self.description}"
    
    def to_rst(self) -> str:
        """Convert to reStructuredText format."""
        return f":raises {self.exception}: {self.description}"


@dataclass
class ExampleDoc:
    """Documentation for a code example.
    
    Attributes:
        code: The example code.
        description: Description of what the example demonstrates.
        output: Expected output (if any).
    """
    code: str
    description: str = ""
    output: Optional[str] = None
    
    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        lines = []
        if self.description:
            lines.append(self.description)
            lines.append("")
        lines.append("```python")
        lines.append(self.code)
        lines.append("```")
        if self.output:
            lines.append("")
            lines.append("Output:")
            lines.append("```")
            lines.append(self.output)
            lines.append("```")
        return "\n".join(lines)
    
    def to_rst(self) -> str:
        """Convert to reStructuredText format."""
        lines = []
        if self.description:
            lines.append(self.description)
            lines.append("")
        lines.append(".. code-block:: python")
        lines.append("")
        for line in self.code.split("\n"):
            lines.append(f"    {line}")
        return "\n".join(lines)


@dataclass
class FunctionDoc:
    """Documentation for a function or method.
    
    Attributes:
        name: Function name.
        signature: Function signature string.
        docstring: Raw docstring.
        short_description: First line of docstring.
        long_description: Extended description.
        parameters: List of parameter documentation.
        returns: Return value documentation.
        raises: List of exception documentation.
        examples: List of code examples.
        decorators: List of decorator names.
        is_method: Whether this is a class method.
        is_classmethod: Whether this is a classmethod.
        is_staticmethod: Whether this is a staticmethod.
        is_property: Whether this is a property.
        is_async: Whether this is an async function.
        source_file: Path to source file.
        line_number: Line number in source file.
    """
    name: str
    signature: str = ""
    docstring: str = ""
    short_description: str = ""
    long_description: str = ""
    parameters: List[ParameterDoc] = field(default_factory=list)
    returns: Optional[ReturnDoc] = None
    raises: List[RaisesDoc] = field(default_factory=list)
    examples: List[ExampleDoc] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    is_method: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    is_property: bool = False
    is_async: bool = False
    source_file: Optional[str] = None
    line_number: Optional[int] = None
    
    def to_markdown(self, level: int = 3) -> str:
        """Convert to Markdown format.
        
        Args:
            level: Heading level (1-6).
            
        Returns:
            Markdown formatted documentation.
        """
        lines = []
        
        # Header
        prefix = "#" * level
        async_str = "async " if self.is_async else ""
        decorator_strs = [f"@{d}" for d in self.decorators]
        
        lines.append(f"{prefix} `{async_str}{self.name}`")
        lines.append("")
        
        # Decorators
        if decorator_strs:
            lines.append("*" + ", ".join(decorator_strs) + "*")
            lines.append("")
        
        # Signature
        if self.signature:
            lines.append("```python")
            lines.append(f"def {self.name}{self.signature}")
            lines.append("```")
            lines.append("")
        
        # Description
        if self.short_description:
            lines.append(self.short_description)
            lines.append("")
        
        if self.long_description:
            lines.append(self.long_description)
            lines.append("")
        
        # Parameters
        if self.parameters:
            lines.append("**Parameters:**")
            lines.append("")
            for param in self.parameters:
                lines.append(param.to_markdown())
            lines.append("")
        
        # Returns
        if self.returns:
            lines.append(self.returns.to_markdown())
            lines.append("")
        
        # Raises
        if self.raises:
            lines.append("**Raises:**")
            lines.append("")
            for exc in self.raises:
                lines.append(exc.to_markdown())
            lines.append("")
        
        # Examples
        if self.examples:
            lines.append("**Examples:**")
            lines.append("")
            for example in self.examples:
                lines.append(example.to_markdown())
                lines.append("")
        
        # Source location
        if self.source_file and self.line_number:
            lines.append(f"*Source: [{self.source_file}:{self.line_number}]({self.source_file}#L{self.line_number})*")
            lines.append("")
        
        return "\n".join(lines)
    
    def to_rst(self, level: int = 0) -> str:
        """Convert to reStructuredText format.
        
        Args:
            level: Section underline character index.
            
        Returns:
            RST formatted documentation.
        """
        underlines = ["=", "-", "~", "^", '"']
        underline_char = underlines[min(level, len(underlines) - 1)]
        
        lines = []
        async_str = "async " if self.is_async else ""
        
        lines.append(f"{async_str}{self.name}")
        lines.append(underline_char * len(f"{async_str}{self.name}"))
        lines.append("")
        
        # Signature as directive
        lines.append(f".. py:function:: {self.name}{self.signature}")
        lines.append("")
        
        # Description
        if self.short_description:
            lines.append(f"    {self.short_description}")
            lines.append("")
        
        if self.long_description:
            for line in self.long_description.split("\n"):
                lines.append(f"    {line}")
            lines.append("")
        
        # Parameters
        for param in self.parameters:
            lines.append(f"    {param.to_rst()}")
        
        # Returns
        if self.returns:
            lines.append(f"    {self.returns.to_rst()}")
        
        # Raises
        for exc in self.raises:
            lines.append(f"    {exc.to_rst()}")
        
        lines.append("")
        
        return "\n".join(lines)


@dataclass
class AttributeDoc:
    """Documentation for a class attribute.
    
    Attributes:
        name: Attribute name.
        type_annotation: Type annotation string.
        description: Attribute description.
        default: Default value if any.
        is_classvar: Whether this is a class variable.
    """
    name: str
    type_annotation: Optional[str] = None
    description: str = ""
    default: Optional[str] = None
    is_classvar: bool = False
    
    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        type_str = f" (`{self.type_annotation}`)" if self.type_annotation else ""
        default_str = f" = `{self.default}`" if self.default else ""
        classvar_str = " *(class variable)*" if self.is_classvar else ""
        return f"- **{self.name}**{type_str}{default_str}: {self.description}{classvar_str}"
    
    def to_rst(self) -> str:
        """Convert to reStructuredText format."""
        type_str = f" ({self.type_annotation})" if self.type_annotation else ""
        return f":ivar {self.name}: {self.description}{type_str}"


@dataclass
class ClassDoc:
    """Documentation for a class.
    
    Attributes:
        name: Class name.
        docstring: Raw docstring.
        short_description: First line of docstring.
        long_description: Extended description.
        bases: List of base class names.
        attributes: List of attribute documentation.
        methods: List of method documentation.
        class_methods: List of classmethod documentation.
        static_methods: List of staticmethod documentation.
        properties: List of property documentation.
        inner_classes: List of inner class documentation.
        decorators: List of decorator names.
        source_file: Path to source file.
        line_number: Line number in source file.
    """
    name: str
    docstring: str = ""
    short_description: str = ""
    long_description: str = ""
    bases: List[str] = field(default_factory=list)
    attributes: List[AttributeDoc] = field(default_factory=list)
    methods: List[FunctionDoc] = field(default_factory=list)
    class_methods: List[FunctionDoc] = field(default_factory=list)
    static_methods: List[FunctionDoc] = field(default_factory=list)
    properties: List[FunctionDoc] = field(default_factory=list)
    inner_classes: List['ClassDoc'] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    source_file: Optional[str] = None
    line_number: Optional[int] = None
    
    def to_markdown(self, level: int = 2) -> str:
        """Convert to Markdown format.
        
        Args:
            level: Heading level (1-6).
            
        Returns:
            Markdown formatted documentation.
        """
        lines = []
        prefix = "#" * level
        
        # Header with inheritance
        bases_str = f"({', '.join(self.bases)})" if self.bases else ""
        lines.append(f"{prefix} class `{self.name}`{bases_str}")
        lines.append("")
        
        # Decorators
        if self.decorators:
            decorator_strs = [f"@{d}" for d in self.decorators]
            lines.append("*" + ", ".join(decorator_strs) + "*")
            lines.append("")
        
        # Description
        if self.short_description:
            lines.append(self.short_description)
            lines.append("")
        
        if self.long_description:
            lines.append(self.long_description)
            lines.append("")
        
        # Attributes
        if self.attributes:
            lines.append(f"{'#' * (level + 1)} Attributes")
            lines.append("")
            for attr in self.attributes:
                lines.append(attr.to_markdown())
            lines.append("")
        
        # Properties
        if self.properties:
            lines.append(f"{'#' * (level + 1)} Properties")
            lines.append("")
            for prop in self.properties:
                lines.append(prop.to_markdown(level=level + 2))
        
        # Methods
        if self.methods:
            lines.append(f"{'#' * (level + 1)} Methods")
            lines.append("")
            for method in self.methods:
                lines.append(method.to_markdown(level=level + 2))
        
        # Class methods
        if self.class_methods:
            lines.append(f"{'#' * (level + 1)} Class Methods")
            lines.append("")
            for method in self.class_methods:
                lines.append(method.to_markdown(level=level + 2))
        
        # Static methods
        if self.static_methods:
            lines.append(f"{'#' * (level + 1)} Static Methods")
            lines.append("")
            for method in self.static_methods:
                lines.append(method.to_markdown(level=level + 2))
        
        # Inner classes
        if self.inner_classes:
            lines.append(f"{'#' * (level + 1)} Inner Classes")
            lines.append("")
            for inner in self.inner_classes:
                lines.append(inner.to_markdown(level=level + 2))
        
        # Source location
        if self.source_file and self.line_number:
            lines.append(f"*Source: [{self.source_file}:{self.line_number}]({self.source_file}#L{self.line_number})*")
            lines.append("")
        
        return "\n".join(lines)
    
    def to_rst(self, level: int = 0) -> str:
        """Convert to reStructuredText format.
        
        Args:
            level: Section underline character index.
            
        Returns:
            RST formatted documentation.
        """
        underlines = ["=", "-", "~", "^", '"']
        underline_char = underlines[min(level, len(underlines) - 1)]
        
        lines = []
        
        lines.append(f"class {self.name}")
        lines.append(underline_char * len(f"class {self.name}"))
        lines.append("")
        
        # Class directive
        bases_str = f"({', '.join(self.bases)})" if self.bases else ""
        lines.append(f".. py:class:: {self.name}{bases_str}")
        lines.append("")
        
        # Description
        if self.short_description:
            lines.append(f"    {self.short_description}")
            lines.append("")
        
        if self.long_description:
            for line in self.long_description.split("\n"):
                lines.append(f"    {line}")
            lines.append("")
        
        # Attributes
        for attr in self.attributes:
            lines.append(f"    {attr.to_rst()}")
        
        lines.append("")
        
        return "\n".join(lines)


@dataclass
class ModuleDoc:
    """Documentation for a Python module.
    
    Attributes:
        name: Module name (fully qualified).
        docstring: Module docstring.
        short_description: First line of docstring.
        long_description: Extended description.
        functions: List of function documentation.
        classes: List of class documentation.
        constants: List of module-level constants.
        submodules: List of submodule documentation.
        source_file: Path to source file.
        all_exports: Contents of __all__ if defined.
    """
    name: str
    docstring: str = ""
    short_description: str = ""
    long_description: str = ""
    functions: List[FunctionDoc] = field(default_factory=list)
    classes: List[ClassDoc] = field(default_factory=list)
    constants: List[AttributeDoc] = field(default_factory=list)
    submodules: List['ModuleDoc'] = field(default_factory=list)
    source_file: Optional[str] = None
    all_exports: List[str] = field(default_factory=list)
    
    def to_markdown(self, level: int = 1) -> str:
        """Convert to Markdown format.
        
        Args:
            level: Heading level (1-6).
            
        Returns:
            Markdown formatted documentation.
        """
        lines = []
        prefix = "#" * level
        
        # Header
        lines.append(f"{prefix} Module `{self.name}`")
        lines.append("")
        
        # Description
        if self.short_description:
            lines.append(self.short_description)
            lines.append("")
        
        if self.long_description:
            lines.append(self.long_description)
            lines.append("")
        
        # Table of contents
        toc_items = []
        if self.classes:
            toc_items.append("Classes")
        if self.functions:
            toc_items.append("Functions")
        if self.constants:
            toc_items.append("Constants")
        if self.submodules:
            toc_items.append("Submodules")
        
        if toc_items:
            lines.append("**Contents:**")
            lines.append("")
            for item in toc_items:
                lines.append(f"- [{item}](#{item.lower()})")
            lines.append("")
        
        # Constants
        if self.constants:
            lines.append(f"{'#' * (level + 1)} Constants")
            lines.append("")
            for const in self.constants:
                lines.append(const.to_markdown())
            lines.append("")
        
        # Classes
        if self.classes:
            lines.append(f"{'#' * (level + 1)} Classes")
            lines.append("")
            for cls in self.classes:
                lines.append(cls.to_markdown(level=level + 2))
        
        # Functions
        if self.functions:
            lines.append(f"{'#' * (level + 1)} Functions")
            lines.append("")
            for func in self.functions:
                lines.append(func.to_markdown(level=level + 2))
        
        # Submodules
        if self.submodules:
            lines.append(f"{'#' * (level + 1)} Submodules")
            lines.append("")
            for sub in self.submodules:
                lines.append(f"- [`{sub.name}`](#{sub.name.replace('.', '-')})")
            lines.append("")
        
        return "\n".join(lines)
    
    def to_rst(self, level: int = 0) -> str:
        """Convert to reStructuredText format.
        
        Args:
            level: Section underline character index.
            
        Returns:
            RST formatted documentation.
        """
        underlines = ["=", "-", "~", "^", '"']
        underline_char = underlines[min(level, len(underlines) - 1)]
        
        lines = []
        
        # Header
        title = f"Module {self.name}"
        lines.append(title)
        lines.append(underline_char * len(title))
        lines.append("")
        
        # Module directive
        lines.append(f".. py:module:: {self.name}")
        lines.append("")
        
        # Description
        if self.short_description:
            lines.append(self.short_description)
            lines.append("")
        
        if self.long_description:
            lines.append(self.long_description)
            lines.append("")
        
        return "\n".join(lines)


class DocstringParser:
    """Parser for Python docstrings.
    
    Supports Google, NumPy, and reStructuredText docstring styles.
    
    Attributes:
        style: Docstring style to use for parsing.
    """
    
    # Regex patterns for different docstring sections
    GOOGLE_SECTION_RE = re.compile(
        r'^(Args|Arguments|Parameters|Returns|Yields|Raises|Attributes|'
        r'Example|Examples|Note|Notes|Warning|Warnings|Todo|References):',
        re.MULTILINE
    )
    
    NUMPY_SECTION_RE = re.compile(
        r'^(Parameters|Returns|Yields|Raises|Attributes|'
        r'Examples|Notes|Warnings|References|See Also)\s*\n-+',
        re.MULTILINE
    )
    
    GOOGLE_PARAM_RE = re.compile(
        r'^\s*(\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.+?)(?=\n\s*\w+\s*(?:\([^)]+\))?\s*:|$)',
        re.MULTILINE | re.DOTALL
    )
    
    NUMPY_PARAM_RE = re.compile(
        r'^\s*(\w+)\s*:\s*([^\n]+)\n\s*(.+?)(?=\n\s*\w+\s*:|$)',
        re.MULTILINE | re.DOTALL
    )
    
    def __init__(self, style: DocstringStyle = DocstringStyle.AUTO):
        """Initialize the parser.
        
        Args:
            style: Docstring style to use. AUTO will detect automatically.
        """
        self.style = style
    
    def detect_style(self, docstring: str) -> DocstringStyle:
        """Detect the docstring style.
        
        Args:
            docstring: Raw docstring text.
            
        Returns:
            Detected DocstringStyle.
        """
        if not docstring:
            return DocstringStyle.GOOGLE
        
        # Check for NumPy style (section headers with underlines)
        if self.NUMPY_SECTION_RE.search(docstring):
            return DocstringStyle.NUMPY
        
        # Check for Google style (section headers with colons)
        if self.GOOGLE_SECTION_RE.search(docstring):
            return DocstringStyle.GOOGLE
        
        # Check for RST style (:param:, :returns:, etc.)
        if re.search(r':(param|returns|raises|type|rtype):', docstring):
            return DocstringStyle.RST
        
        return DocstringStyle.GOOGLE
    
    def parse(self, docstring: str) -> Dict[str, Any]:
        """Parse a docstring.
        
        Args:
            docstring: Raw docstring text.
            
        Returns:
            Dictionary with parsed sections:
                - short_description: First line
                - long_description: Extended description
                - parameters: List of ParameterDoc
                - returns: ReturnDoc or None
                - raises: List of RaisesDoc
                - examples: List of ExampleDoc
        """
        if not docstring:
            return {
                'short_description': '',
                'long_description': '',
                'parameters': [],
                'returns': None,
                'raises': [],
                'examples': [],
            }
        
        docstring = textwrap.dedent(docstring).strip()
        
        style = self.style if self.style != DocstringStyle.AUTO else self.detect_style(docstring)
        
        if style == DocstringStyle.NUMPY:
            return self._parse_numpy(docstring)
        elif style == DocstringStyle.RST:
            return self._parse_rst(docstring)
        else:
            return self._parse_google(docstring)
    
    def _parse_google(self, docstring: str) -> Dict[str, Any]:
        """Parse Google-style docstring."""
        result = {
            'short_description': '',
            'long_description': '',
            'parameters': [],
            'returns': None,
            'raises': [],
            'examples': [],
        }
        
        # Split into lines
        lines = docstring.split('\n')
        
        # Extract short description (first non-empty line)
        short_desc_lines = []
        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            if not line:
                idx += 1
                break
            short_desc_lines.append(line)
            idx += 1
            # Stop at first period or section header
            if line.endswith('.') or self.GOOGLE_SECTION_RE.match(line):
                break
        
        result['short_description'] = ' '.join(short_desc_lines)
        
        # Find sections
        sections = {}
        current_section = 'description'
        current_content = []
        
        for line in lines[idx:]:
            match = self.GOOGLE_SECTION_RE.match(line.strip())
            if match:
                if current_content and current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = match.group(1).lower()
                current_content = []
            else:
                current_content.append(line)
        
        if current_content and current_section:
            sections[current_section] = '\n'.join(current_content)
        
        # Extract long description
        if 'description' in sections:
            result['long_description'] = sections['description'].strip()
        
        # Parse parameters
        param_section = sections.get('args') or sections.get('arguments') or sections.get('parameters', '')
        if param_section:
            result['parameters'] = self._parse_google_params(param_section)
        
        # Parse returns
        returns_section = sections.get('returns') or sections.get('yields', '')
        if returns_section:
            result['returns'] = self._parse_google_returns(returns_section)
        
        # Parse raises
        raises_section = sections.get('raises', '')
        if raises_section:
            result['raises'] = self._parse_google_raises(raises_section)
        
        # Parse examples
        examples_section = sections.get('examples') or sections.get('example', '')
        if examples_section:
            result['examples'] = self._parse_examples(examples_section)
        
        return result
    
    def _parse_google_params(self, section: str) -> List[ParameterDoc]:
        """Parse Google-style parameter section."""
        params = []
        
        # Split by parameter entries
        lines = section.strip().split('\n')
        current_param = None
        current_desc = []
        
        for line in lines:
            # Check for new parameter
            match = re.match(r'^\s*(\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.*)$', line)
            if match:
                # Save previous parameter
                if current_param is not None:
                    current_param.description = ' '.join(current_desc).strip()
                    params.append(current_param)
                
                name, type_ann, desc = match.groups()
                optional = 'optional' in (type_ann or '').lower()
                
                # Extract default from description
                default = None
                if desc:
                    default_match = re.search(r'[Dd]efault[s]?\s*[:=]?\s*[`\'"]?([^`\'".,]+)', desc)
                    if default_match:
                        default = default_match.group(1).strip()
                
                current_param = ParameterDoc(
                    name=name,
                    type_annotation=type_ann.strip() if type_ann else None,
                    optional=optional,
                    default=default,
                )
                current_desc = [desc] if desc else []
            elif current_param is not None:
                current_desc.append(line.strip())
        
        # Save last parameter
        if current_param is not None:
            current_param.description = ' '.join(current_desc).strip()
            params.append(current_param)
        
        return params
    
    def _parse_google_returns(self, section: str) -> Optional[ReturnDoc]:
        """Parse Google-style returns section."""
        section = section.strip()
        if not section:
            return None
        
        # Check for type: description format
        match = re.match(r'^([^:]+):\s*(.+)$', section, re.DOTALL)
        if match:
            type_ann, desc = match.groups()
            return ReturnDoc(
                type_annotation=type_ann.strip(),
                description=desc.strip(),
            )
        
        return ReturnDoc(description=section)
    
    def _parse_google_raises(self, section: str) -> List[RaisesDoc]:
        """Parse Google-style raises section."""
        raises = []
        
        lines = section.strip().split('\n')
        current_exc = None
        current_desc = []
        
        for line in lines:
            match = re.match(r'^\s*(\w+)\s*:\s*(.*)$', line)
            if match:
                if current_exc is not None:
                    raises.append(RaisesDoc(
                        exception=current_exc,
                        description=' '.join(current_desc).strip(),
                    ))
                
                current_exc, desc = match.groups()
                current_desc = [desc] if desc else []
            elif current_exc is not None:
                current_desc.append(line.strip())
        
        if current_exc is not None:
            raises.append(RaisesDoc(
                exception=current_exc,
                description=' '.join(current_desc).strip(),
            ))
        
        return raises
    
    def _parse_numpy(self, docstring: str) -> Dict[str, Any]:
        """Parse NumPy-style docstring."""
        # Similar structure to Google, but different section markers
        # For brevity, delegate to Google parser with preprocessing
        # Convert NumPy sections to Google-like format
        converted = docstring
        
        # Replace NumPy section headers with Google-style
        converted = re.sub(r'^(Parameters|Returns|Raises|Examples)\s*\n-+', r'\1:', converted, flags=re.MULTILINE)
        
        return self._parse_google(converted)
    
    def _parse_rst(self, docstring: str) -> Dict[str, Any]:
        """Parse reStructuredText-style docstring."""
        result = {
            'short_description': '',
            'long_description': '',
            'parameters': [],
            'returns': None,
            'raises': [],
            'examples': [],
        }
        
        lines = docstring.split('\n')
        
        # Extract short description
        short_desc_lines = []
        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            if not line or line.startswith(':'):
                break
            short_desc_lines.append(line)
            idx += 1
        
        result['short_description'] = ' '.join(short_desc_lines)
        
        # Parse RST directives
        remaining = '\n'.join(lines[idx:])
        
        # Parameters (:param name: description)
        for match in re.finditer(r':param\s+(\w+):\s*([^\n]+)', remaining):
            name, desc = match.groups()
            result['parameters'].append(ParameterDoc(name=name, description=desc.strip()))
        
        # Parameter types (:type name: type)
        for match in re.finditer(r':type\s+(\w+):\s*([^\n]+)', remaining):
            name, type_ann = match.groups()
            for param in result['parameters']:
                if param.name == name:
                    param.type_annotation = type_ann.strip()
                    break
        
        # Returns (:returns: description)
        match = re.search(r':returns?:\s*([^\n]+)', remaining)
        if match:
            result['returns'] = ReturnDoc(description=match.group(1).strip())
        
        # Return type (:rtype: type)
        match = re.search(r':rtype:\s*([^\n]+)', remaining)
        if match and result['returns']:
            result['returns'].type_annotation = match.group(1).strip()
        
        # Raises (:raises ExceptionType: description)
        for match in re.finditer(r':raises?\s+(\w+):\s*([^\n]+)', remaining):
            exc, desc = match.groups()
            result['raises'].append(RaisesDoc(exception=exc, description=desc.strip()))
        
        return result
    
    def _parse_examples(self, section: str) -> List[ExampleDoc]:
        """Parse examples section."""
        examples = []
        
        # Look for code blocks (>>> or indented code)
        lines = section.strip().split('\n')
        current_code = []
        current_output = []
        in_code = False
        in_output = False
        
        for line in lines:
            if line.strip().startswith('>>>'):
                if in_output and current_code:
                    examples.append(ExampleDoc(
                        code='\n'.join(current_code),
                        output='\n'.join(current_output) if current_output else None,
                    ))
                    current_code = []
                    current_output = []
                
                in_code = True
                in_output = False
                current_code.append(line.strip()[4:])  # Remove >>>
            elif in_code and line.strip().startswith('...'):
                current_code.append(line.strip()[4:])  # Remove ...
            elif in_code and line.strip():
                in_output = True
                current_output.append(line.strip())
            elif not line.strip() and current_code:
                examples.append(ExampleDoc(
                    code='\n'.join(current_code),
                    output='\n'.join(current_output) if current_output else None,
                ))
                current_code = []
                current_output = []
                in_code = False
                in_output = False
        
        if current_code:
            examples.append(ExampleDoc(
                code='\n'.join(current_code),
                output='\n'.join(current_output) if current_output else None,
            ))
        
        return examples


class APIExtractor:
    """Extract API documentation from Python modules.
    
    This class provides methods to extract documentation from Python source
    code, including docstrings, type annotations, and structural information.
    
    Attributes:
        parser: DocstringParser instance for parsing docstrings.
        include_private: Whether to include private members (_name).
        include_dunder: Whether to include dunder members (__name__).
    """
    
    def __init__(
        self,
        style: DocstringStyle = DocstringStyle.AUTO,
        include_private: bool = False,
        include_dunder: bool = False,
    ):
        """Initialize the extractor.
        
        Args:
            style: Docstring style to use for parsing.
            include_private: Include private members (_name).
            include_dunder: Include dunder members (__name__).
        """
        self.parser = DocstringParser(style)
        self.include_private = include_private
        self.include_dunder = include_dunder
    
    def should_include(self, name: str) -> bool:
        """Check if a name should be included in documentation.
        
        Args:
            name: The name to check.
            
        Returns:
            True if the name should be included.
        """
        if name.startswith('__') and name.endswith('__'):
            return self.include_dunder or name == '__init__'
        if name.startswith('_'):
            return self.include_private
        return True
    
    def extract_module(self, module: Any, source_file: Optional[str] = None) -> ModuleDoc:
        """Extract documentation from a module.
        
        Args:
            module: The module to document.
            source_file: Path to source file.
            
        Returns:
            ModuleDoc for the module.
        """
        docstring = inspect.getdoc(module) or ""
        parsed = self.parser.parse(docstring)
        
        doc = ModuleDoc(
            name=module.__name__,
            docstring=docstring,
            short_description=parsed['short_description'],
            long_description=parsed['long_description'],
            source_file=source_file or getattr(module, '__file__', None),
            all_exports=getattr(module, '__all__', []),
        )
        
        # Extract functions
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if self.should_include(name):
                if obj.__module__ == module.__name__:
                    doc.functions.append(self.extract_function(obj))
        
        # Extract classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if self.should_include(name):
                if obj.__module__ == module.__name__:
                    doc.classes.append(self.extract_class(obj))
        
        return doc
    
    def extract_function(
        self,
        func: Callable,
        is_method: bool = False,
    ) -> FunctionDoc:
        """Extract documentation from a function.
        
        Args:
            func: The function to document.
            is_method: Whether this is a class method.
            
        Returns:
            FunctionDoc for the function.
        """
        docstring = inspect.getdoc(func) or ""
        parsed = self.parser.parse(docstring)
        
        # Get signature
        try:
            sig = inspect.signature(func)
            sig_str = str(sig)
        except (ValueError, TypeError):
            sig_str = "(...)"
        
        # Get decorators (limited - can't always detect)
        decorators = []
        if hasattr(func, '__wrapped__'):
            decorators.append('wraps')
        
        # Check for async
        is_async = inspect.iscoroutinefunction(func)
        
        # Get source location
        try:
            source_file = inspect.getfile(func)
            line_number = inspect.getsourcelines(func)[1]
        except (TypeError, OSError):
            source_file = None
            line_number = None
        
        # Merge parsed parameters with signature
        params = []
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}
        
        for param in parsed['parameters']:
            if param.name in hints:
                param.type_annotation = str(hints[param.name])
            params.append(param)
        
        # Update return type from hints
        returns = parsed['returns']
        if 'return' in hints and returns:
            returns.type_annotation = str(hints['return'])
        
        return FunctionDoc(
            name=func.__name__,
            signature=sig_str,
            docstring=docstring,
            short_description=parsed['short_description'],
            long_description=parsed['long_description'],
            parameters=params,
            returns=returns,
            raises=parsed['raises'],
            examples=parsed['examples'],
            decorators=decorators,
            is_method=is_method,
            is_async=is_async,
            source_file=source_file,
            line_number=line_number,
        )
    
    def extract_class(self, cls: Type) -> ClassDoc:
        """Extract documentation from a class.
        
        Args:
            cls: The class to document.
            
        Returns:
            ClassDoc for the class.
        """
        docstring = inspect.getdoc(cls) or ""
        parsed = self.parser.parse(docstring)
        
        # Get base classes
        bases = [base.__name__ for base in cls.__bases__ if base is not object]
        
        # Get source location
        try:
            source_file = inspect.getfile(cls)
            line_number = inspect.getsourcelines(cls)[1]
        except (TypeError, OSError):
            source_file = None
            line_number = None
        
        doc = ClassDoc(
            name=cls.__name__,
            docstring=docstring,
            short_description=parsed['short_description'],
            long_description=parsed['long_description'],
            bases=bases,
            source_file=source_file,
            line_number=line_number,
        )
        
        # Extract attributes from class annotations
        annotations = getattr(cls, '__annotations__', {})
        for name, type_ann in annotations.items():
            if self.should_include(name):
                doc.attributes.append(AttributeDoc(
                    name=name,
                    type_annotation=str(type_ann),
                ))
        
        # Extract methods
        for name, obj in inspect.getmembers(cls):
            if not self.should_include(name):
                continue
            
            if isinstance(obj, property):
                doc.properties.append(self.extract_function(obj.fget, is_method=True))
            elif isinstance(obj, classmethod):
                func_doc = self.extract_function(obj.__func__, is_method=True)
                func_doc.is_classmethod = True
                doc.class_methods.append(func_doc)
            elif isinstance(obj, staticmethod):
                func_doc = self.extract_function(obj.__func__, is_method=True)
                func_doc.is_staticmethod = True
                doc.static_methods.append(func_doc)
            elif inspect.isfunction(obj) or inspect.ismethod(obj):
                # Only include methods defined in this class
                try:
                    if obj.__qualname__.startswith(cls.__name__):
                        doc.methods.append(self.extract_function(obj, is_method=True))
                except AttributeError:
                    pass
        
        return doc


def extract_module_docs(
    module_path: Union[str, Path],
    recursive: bool = True,
    include_private: bool = False,
) -> ModuleDoc:
    """Extract documentation from a module path.
    
    Args:
        module_path: Path to the module or package.
        recursive: Whether to recursively document submodules.
        include_private: Whether to include private members.
        
    Returns:
        ModuleDoc for the module/package.
    """
    extractor = APIExtractor(include_private=include_private)
    
    path = Path(module_path)
    
    if path.is_file():
        # Single module
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return extractor.extract_module(module, str(path))
    
    elif path.is_dir():
        # Package
        init_path = path / '__init__.py'
        if init_path.exists():
            spec = importlib.util.spec_from_file_location(
                path.name, init_path,
                submodule_search_locations=[str(path)]
            )
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception:
                pass
            doc = extractor.extract_module(module, str(init_path))
            
            if recursive:
                for child in path.iterdir():
                    if child.is_file() and child.suffix == '.py' and child.name != '__init__.py':
                        try:
                            sub_doc = extract_module_docs(child, recursive=False, include_private=include_private)
                            sub_doc.name = f"{doc.name}.{child.stem}"
                            doc.submodules.append(sub_doc)
                        except Exception:
                            pass
                    elif child.is_dir() and (child / '__init__.py').exists():
                        try:
                            sub_doc = extract_module_docs(child, recursive=True, include_private=include_private)
                            sub_doc.name = f"{doc.name}.{child.name}"
                            doc.submodules.append(sub_doc)
                        except Exception:
                            pass
            
            return doc
    
    raise ValueError(f"Invalid module path: {module_path}")


def generate_api_markdown(
    module_doc: ModuleDoc,
    output_dir: Optional[Path] = None,
    single_file: bool = False,
) -> Dict[str, str]:
    """Generate Markdown documentation for a module.
    
    Args:
        module_doc: The ModuleDoc to generate documentation for.
        output_dir: Directory to write files to (optional).
        single_file: Whether to generate a single file.
        
    Returns:
        Dictionary mapping filenames to content.
    """
    files = {}
    
    if single_file:
        content = module_doc.to_markdown()
        for submod in module_doc.submodules:
            content += "\n\n---\n\n"
            content += submod.to_markdown()
        files[f"{module_doc.name}.md"] = content
    else:
        files[f"{module_doc.name}.md"] = module_doc.to_markdown()
        for submod in module_doc.submodules:
            sub_files = generate_api_markdown(submod, single_file=False)
            files.update(sub_files)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in files.items():
            (output_dir / filename).write_text(content)
    
    return files


def generate_api_rst(
    module_doc: ModuleDoc,
    output_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Generate reStructuredText documentation for a module.
    
    Args:
        module_doc: The ModuleDoc to generate documentation for.
        output_dir: Directory to write files to (optional).
        
    Returns:
        Dictionary mapping filenames to content.
    """
    files = {}
    
    content = module_doc.to_rst()
    files[f"{module_doc.name}.rst"] = content
    
    for submod in module_doc.submodules:
        sub_files = generate_api_rst(submod)
        files.update(sub_files)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in files.items():
            (output_dir / filename).write_text(content)
    
    return files
