"""
Runnable Code Examples for Project HyperTensor.

This module provides infrastructure for creating, validating, and running
code examples from documentation and tutorials.

Features:
    - Extract examples from docstrings
    - Validate example syntax and execution
    - Generate Jupyter notebooks from examples
    - Test example output against expected results
"""

from __future__ import annotations

import ast
import re
import sys
import traceback
import textwrap
from dataclasses import dataclass, field
from enum import Enum, auto
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class ExampleType(Enum):
    """Type of code example."""
    DOCTEST = auto()      # >>> style examples
    CODE_BLOCK = auto()   # Fenced code blocks
    SCRIPT = auto()       # Full script files
    INTERACTIVE = auto()  # REPL-style examples


class ExampleStatus(Enum):
    """Status of example validation."""
    PASSED = auto()
    FAILED = auto()
    ERROR = auto()
    SKIPPED = auto()


@dataclass
class ExampleConfig:
    """Configuration for running examples.
    
    Attributes:
        timeout_seconds: Maximum execution time.
        capture_output: Whether to capture stdout/stderr.
        show_traceback: Whether to show full tracebacks.
        require_exact_output: Whether output must match exactly.
        allow_warnings: Whether to allow warning messages.
        setup_code: Code to run before each example.
        teardown_code: Code to run after each example.
        skip_patterns: Patterns matching examples to skip.
        extra_globals: Additional globals to inject.
    """
    timeout_seconds: float = 30.0
    capture_output: bool = True
    show_traceback: bool = True
    require_exact_output: bool = False
    allow_warnings: bool = True
    setup_code: str = ""
    teardown_code: str = ""
    skip_patterns: List[str] = field(default_factory=list)
    extra_globals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExampleResult:
    """Result of running an example.
    
    Attributes:
        status: Execution status.
        output: Captured stdout output.
        stderr: Captured stderr output.
        expected_output: Expected output (if any).
        error: Exception if one occurred.
        traceback_str: Formatted traceback.
        duration_seconds: Execution time.
        line_number: Source line number (if from file).
    """
    status: ExampleStatus
    output: str = ""
    stderr: str = ""
    expected_output: Optional[str] = None
    error: Optional[Exception] = None
    traceback_str: str = ""
    duration_seconds: float = 0.0
    line_number: Optional[int] = None
    
    @property
    def passed(self) -> bool:
        """Check if the example passed."""
        return self.status == ExampleStatus.PASSED
    
    @property
    def output_matches(self) -> bool:
        """Check if output matches expected."""
        if self.expected_output is None:
            return True
        return normalize_output(self.output) == normalize_output(self.expected_output)


@dataclass
class RunnableExample:
    """A code example that can be executed.
    
    Attributes:
        code: The Python code.
        name: Example name/identifier.
        description: Description of what the example demonstrates.
        expected_output: Expected output.
        example_type: Type of example.
        source_file: Source file path.
        line_number: Line number in source file.
        tags: Tags for categorization.
        skip: Whether to skip this example.
        skip_reason: Reason for skipping.
    """
    code: str
    name: str = ""
    description: str = ""
    expected_output: Optional[str] = None
    example_type: ExampleType = ExampleType.CODE_BLOCK
    source_file: Optional[str] = None
    line_number: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    skip: bool = False
    skip_reason: str = ""
    
    def run(
        self,
        config: Optional[ExampleConfig] = None,
        globals_dict: Optional[Dict[str, Any]] = None,
    ) -> ExampleResult:
        """Run the example and return the result.
        
        Args:
            config: Execution configuration.
            globals_dict: Global namespace to use.
            
        Returns:
            ExampleResult with status and output.
        """
        config = config or ExampleConfig()
        
        if self.skip:
            return ExampleResult(
                status=ExampleStatus.SKIPPED,
                line_number=self.line_number,
            )
        
        # Check skip patterns
        for pattern in config.skip_patterns:
            if re.search(pattern, self.code):
                return ExampleResult(
                    status=ExampleStatus.SKIPPED,
                    line_number=self.line_number,
                )
        
        # Prepare namespace
        namespace = {
            "__name__": "__main__",
            "__doc__": None,
            "__package__": None,
        }
        namespace.update(config.extra_globals)
        if globals_dict:
            namespace.update(globals_dict)
        
        # Capture output
        import time
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        if config.capture_output:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
        
        start_time = time.time()
        
        try:
            # Run setup code
            if config.setup_code:
                exec(config.setup_code, namespace)
            
            # Compile and execute
            code = textwrap.dedent(self.code)
            compiled = compile(code, f"<example:{self.name}>", "exec")
            exec(compiled, namespace)
            
            # Run teardown code
            if config.teardown_code:
                exec(config.teardown_code, namespace)
            
            duration = time.time() - start_time
            
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            output = stdout_capture.getvalue()
            stderr_out = stderr_capture.getvalue()
            
            # Check output
            if self.expected_output is not None:
                if config.require_exact_output:
                    matches = output.strip() == self.expected_output.strip()
                else:
                    matches = normalize_output(output) == normalize_output(self.expected_output)
                
                if not matches:
                    return ExampleResult(
                        status=ExampleStatus.FAILED,
                        output=output,
                        stderr=stderr_out,
                        expected_output=self.expected_output,
                        duration_seconds=duration,
                        line_number=self.line_number,
                    )
            
            return ExampleResult(
                status=ExampleStatus.PASSED,
                output=output,
                stderr=stderr_out,
                expected_output=self.expected_output,
                duration_seconds=duration,
                line_number=self.line_number,
            )
            
        except Exception as e:
            duration = time.time() - start_time
            
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            tb_str = traceback.format_exc() if config.show_traceback else str(e)
            
            return ExampleResult(
                status=ExampleStatus.ERROR,
                output=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                error=e,
                traceback_str=tb_str,
                duration_seconds=duration,
                line_number=self.line_number,
            )
    
    def to_notebook_cell(self) -> Dict[str, Any]:
        """Convert to Jupyter notebook cell format.
        
        Returns:
            Dictionary representing a notebook cell.
        """
        cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": self.code.split("\n"),
        }
        
        if self.expected_output:
            cell["outputs"] = [{
                "output_type": "stream",
                "name": "stdout",
                "text": self.expected_output.split("\n"),
            }]
        
        return cell


def normalize_output(text: str) -> str:
    """Normalize output for comparison.
    
    Strips whitespace, normalizes line endings, and handles
    floating-point representation differences.
    
    Args:
        text: Output text to normalize.
        
    Returns:
        Normalized text.
    """
    if not text:
        return ""
    
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    
    # Remove empty lines at start and end
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    
    # Normalize floating-point representations
    # Match floating point numbers and round to reasonable precision
    def round_floats(match):
        value = float(match.group(0))
        return f"{value:.6g}"
    
    result = "\n".join(lines)
    result = re.sub(r'-?\d+\.\d{6,}', round_floats, result)
    
    return result


class ExampleRunner:
    """Runner for executing collections of examples.
    
    This class manages running multiple examples, collecting results,
    and generating reports.
    
    Attributes:
        config: Execution configuration.
        examples: List of examples to run.
        results: Results from running examples.
    """
    
    def __init__(self, config: Optional[ExampleConfig] = None):
        """Initialize the runner.
        
        Args:
            config: Execution configuration.
        """
        self.config = config or ExampleConfig()
        self.examples: List[RunnableExample] = []
        self.results: List[Tuple[RunnableExample, ExampleResult]] = []
    
    def add_example(self, example: RunnableExample) -> None:
        """Add an example to run.
        
        Args:
            example: The example to add.
        """
        self.examples.append(example)
    
    def add_examples_from_module(self, module: Any) -> int:
        """Extract and add examples from a module's docstrings.
        
        Args:
            module: The module to extract from.
            
        Returns:
            Number of examples added.
        """
        examples = extract_examples_from_docstrings(module)
        for example in examples:
            self.add_example(example)
        return len(examples)
    
    def run_all(
        self,
        stop_on_failure: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run all examples.
        
        Args:
            stop_on_failure: Whether to stop on first failure.
            verbose: Whether to print progress.
            
        Returns:
            Summary dictionary with counts and results.
        """
        self.results = []
        
        passed = 0
        failed = 0
        errors = 0
        skipped = 0
        
        for i, example in enumerate(self.examples, 1):
            if verbose:
                print(f"Running example {i}/{len(self.examples)}: {example.name or 'unnamed'}... ", end="")
            
            result = example.run(self.config)
            self.results.append((example, result))
            
            if result.status == ExampleStatus.PASSED:
                passed += 1
                if verbose:
                    print("PASSED")
            elif result.status == ExampleStatus.FAILED:
                failed += 1
                if verbose:
                    print("FAILED")
                    if result.expected_output:
                        print(f"  Expected: {result.expected_output[:100]}...")
                        print(f"  Got: {result.output[:100]}...")
                if stop_on_failure:
                    break
            elif result.status == ExampleStatus.ERROR:
                errors += 1
                if verbose:
                    print(f"ERROR: {result.error}")
                    if result.traceback_str:
                        print(result.traceback_str)
                if stop_on_failure:
                    break
            else:
                skipped += 1
                if verbose:
                    print("SKIPPED")
        
        return {
            "total": len(self.examples),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "success_rate": passed / max(1, len(self.examples) - skipped),
            "results": self.results,
        }
    
    def generate_report(self, format: str = "text") -> str:
        """Generate a report of the results.
        
        Args:
            format: Report format ("text", "markdown", "json").
            
        Returns:
            Formatted report string.
        """
        if format == "json":
            import json
            data = {
                "total": len(self.results),
                "passed": sum(1 for _, r in self.results if r.status == ExampleStatus.PASSED),
                "failed": sum(1 for _, r in self.results if r.status == ExampleStatus.FAILED),
                "errors": sum(1 for _, r in self.results if r.status == ExampleStatus.ERROR),
                "skipped": sum(1 for _, r in self.results if r.status == ExampleStatus.SKIPPED),
                "examples": [
                    {
                        "name": e.name,
                        "status": r.status.name,
                        "duration": r.duration_seconds,
                        "output": r.output[:500] if r.output else None,
                        "error": str(r.error) if r.error else None,
                    }
                    for e, r in self.results
                ],
            }
            return json.dumps(data, indent=2)
        
        lines = ["Example Execution Report", "=" * 25, ""]
        
        passed = sum(1 for _, r in self.results if r.status == ExampleStatus.PASSED)
        failed = sum(1 for _, r in self.results if r.status == ExampleStatus.FAILED)
        errors = sum(1 for _, r in self.results if r.status == ExampleStatus.ERROR)
        skipped = sum(1 for _, r in self.results if r.status == ExampleStatus.SKIPPED)
        
        lines.append(f"Total: {len(self.results)}")
        lines.append(f"Passed: {passed}")
        lines.append(f"Failed: {failed}")
        lines.append(f"Errors: {errors}")
        lines.append(f"Skipped: {skipped}")
        lines.append("")
        
        # Show failures and errors
        for example, result in self.results:
            if result.status in (ExampleStatus.FAILED, ExampleStatus.ERROR):
                lines.append("-" * 40)
                lines.append(f"Example: {example.name or 'unnamed'}")
                lines.append(f"Status: {result.status.name}")
                if result.error:
                    lines.append(f"Error: {result.error}")
                if result.expected_output and result.output:
                    lines.append(f"Expected: {result.expected_output[:200]}")
                    lines.append(f"Got: {result.output[:200]}")
                lines.append("")
        
        if format == "markdown":
            # Convert to markdown table
            md_lines = ["# Example Execution Report", ""]
            md_lines.append(f"- **Total**: {len(self.results)}")
            md_lines.append(f"- **Passed**: {passed} ✅")
            md_lines.append(f"- **Failed**: {failed} ❌")
            md_lines.append(f"- **Errors**: {errors} ⚠️")
            md_lines.append(f"- **Skipped**: {skipped} ⏭️")
            md_lines.append("")
            md_lines.append("| Example | Status | Duration |")
            md_lines.append("|---------|--------|----------|")
            for example, result in self.results:
                status_emoji = {
                    ExampleStatus.PASSED: "✅",
                    ExampleStatus.FAILED: "❌",
                    ExampleStatus.ERROR: "⚠️",
                    ExampleStatus.SKIPPED: "⏭️",
                }.get(result.status, "?")
                md_lines.append(
                    f"| {example.name or 'unnamed'} | {status_emoji} | {result.duration_seconds:.3f}s |"
                )
            return "\n".join(md_lines)
        
        return "\n".join(lines)
    
    def to_notebook(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Convert examples to Jupyter notebook format.
        
        Args:
            output_path: Optional path to write the notebook.
            
        Returns:
            Notebook dictionary.
        """
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3.11",
                },
            },
            "cells": [],
        }
        
        for example in self.examples:
            # Add description as markdown cell
            if example.description:
                notebook["cells"].append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [example.description],
                })
            
            # Add code cell
            notebook["cells"].append(example.to_notebook_cell())
        
        if output_path:
            import json
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(notebook, indent=2))
        
        return notebook


def extract_examples_from_docstrings(
    module: Any,
    recursive: bool = True,
) -> List[RunnableExample]:
    """Extract examples from module docstrings.
    
    Parses docstrings to find code examples in various formats:
    - Doctest format (>>> lines)
    - Code blocks (```python ... ```)
    - Examples: sections
    
    Args:
        module: Module to extract examples from.
        recursive: Whether to recurse into classes and functions.
        
    Returns:
        List of RunnableExample objects.
    """
    import inspect
    
    examples = []
    
    def extract_from_docstring(docstring: str, name: str, source_file: Optional[str] = None) -> List[RunnableExample]:
        """Extract examples from a single docstring."""
        if not docstring:
            return []
        
        found = []
        
        # Extract doctest examples (>>> lines)
        doctest_pattern = re.compile(
            r'^(\s*)>>>\s*(.+?)(?=\n\1>>>|\n\s*\n|\Z)',
            re.MULTILINE | re.DOTALL
        )
        
        for match in doctest_pattern.finditer(docstring):
            code = match.group(2)
            
            # Handle continuation lines (...)
            lines = code.split("\n")
            code_lines = []
            output_lines = []
            in_output = False
            
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(">>>"):
                    code_lines.append(stripped[4:])
                    in_output = False
                elif stripped.startswith("..."):
                    code_lines.append(stripped[4:])
                elif stripped and not stripped.startswith("#"):
                    output_lines.append(stripped)
                    in_output = True
            
            if code_lines:
                found.append(RunnableExample(
                    code="\n".join(code_lines),
                    name=f"{name}_doctest_{len(found)+1}",
                    expected_output="\n".join(output_lines) if output_lines else None,
                    example_type=ExampleType.DOCTEST,
                    source_file=source_file,
                ))
        
        # Extract fenced code blocks (```python ... ```)
        code_block_pattern = re.compile(
            r'```python\n(.+?)```',
            re.DOTALL
        )
        
        for match in code_block_pattern.finditer(docstring):
            code = match.group(1).strip()
            
            found.append(RunnableExample(
                code=code,
                name=f"{name}_codeblock_{len(found)+1}",
                example_type=ExampleType.CODE_BLOCK,
                source_file=source_file,
            ))
        
        return found
    
    # Extract from module docstring
    source_file = getattr(module, "__file__", None)
    examples.extend(extract_from_docstring(
        inspect.getdoc(module) or "",
        module.__name__,
        source_file,
    ))
    
    if recursive:
        # Extract from functions
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ == module.__name__:
                examples.extend(extract_from_docstring(
                    inspect.getdoc(obj) or "",
                    f"{module.__name__}.{name}",
                    source_file,
                ))
        
        # Extract from classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__:
                examples.extend(extract_from_docstring(
                    inspect.getdoc(obj) or "",
                    f"{module.__name__}.{name}",
                    source_file,
                ))
                
                # Extract from methods
                for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                    examples.extend(extract_from_docstring(
                        inspect.getdoc(method) or "",
                        f"{module.__name__}.{name}.{method_name}",
                        source_file,
                    ))
    
    return examples


def validate_example(
    code: str,
    expected_output: Optional[str] = None,
    config: Optional[ExampleConfig] = None,
) -> ExampleResult:
    """Validate a code example.
    
    Convenience function to quickly validate a code snippet.
    
    Args:
        code: Python code to validate.
        expected_output: Expected output.
        config: Execution configuration.
        
    Returns:
        ExampleResult with validation status.
    """
    example = RunnableExample(
        code=code,
        name="validation",
        expected_output=expected_output,
    )
    return example.run(config)


def validate_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Validate Python syntax without executing.
    
    Args:
        code: Python code to validate.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
