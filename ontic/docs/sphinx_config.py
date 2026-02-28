"""
Sphinx Configuration Generator for Project HyperTensor.

This module provides utilities for generating Sphinx documentation
configuration and building HTML/PDF documentation.

Features:
    - Configuration file generation (conf.py)
    - Theme customization (RTD, Furo, Pydata)
    - Extension management
    - Build automation
    - Multi-format output (HTML, PDF, EPUB)
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SphinxTheme(Enum):
    """Available Sphinx documentation themes."""

    RTD = "sphinx_rtd_theme"
    FURO = "furo"
    PYDATA = "pydata_sphinx_theme"
    ALABASTER = "alabaster"
    BOOK = "sphinx_book_theme"
    MATERIAL = "sphinx_material"


class OutputFormat(Enum):
    """Documentation output formats."""

    HTML = "html"
    PDF = "latexpdf"
    EPUB = "epub"
    JSON = "json"
    SINGLE_HTML = "singlehtml"


@dataclass
class SphinxExtension:
    """A Sphinx extension configuration.

    Attributes:
        name: Extension module name.
        config: Extension-specific configuration.
        required: Whether the extension is required.
    """

    name: str
    config: dict[str, Any] = field(default_factory=dict)
    required: bool = True


@dataclass
class SphinxConfig:
    """Configuration for Sphinx documentation.

    Attributes:
        project: Project name.
        author: Author name(s).
        version: Documentation version.
        release: Full release version.
        copyright_year: Copyright year.
        theme: Sphinx theme to use.
        extensions: List of Sphinx extensions.
        templates_path: Path to custom templates.
        static_path: Path to static files.
        source_suffix: Source file suffix(es).
        master_doc: Master document name.
        language: Documentation language.
        exclude_patterns: Patterns to exclude.
        html_logo: Path to logo image.
        html_favicon: Path to favicon.
        pygments_style: Code highlighting style.
        autodoc_options: Autodoc default options.
        intersphinx_mapping: Intersphinx mapping.
        napoleon_options: Napoleon docstring options.
        custom_config: Additional custom configuration.
    """

    project: str
    author: str = "HyperTensor Team"
    version: str = "1.0"
    release: str = "1.0.0"
    copyright_year: str = "2025"
    theme: SphinxTheme = SphinxTheme.FURO
    extensions: list[SphinxExtension] = field(default_factory=list)
    templates_path: str = "_templates"
    static_path: str = "_static"
    source_suffix: str = ".rst"
    master_doc: str = "index"
    language: str = "en"
    exclude_patterns: list[str] = field(
        default_factory=lambda: ["_build", "Thumbs.db", ".DS_Store"]
    )
    html_logo: str | None = None
    html_favicon: str | None = None
    pygments_style: str = "monokai"
    autodoc_options: dict[str, Any] = field(
        default_factory=lambda: {
            "members": True,
            "undoc-members": True,
            "show-inheritance": True,
            "special-members": "__init__",
        }
    )
    intersphinx_mapping: dict[str, tuple[str, str | None]] = field(
        default_factory=lambda: {
            "python": ("https://docs.python.org/3", None),
            "numpy": ("https://numpy.org/doc/stable/", None),
            "torch": ("https://pytorch.org/docs/stable/", None),
        }
    )
    napoleon_options: dict[str, bool] = field(
        default_factory=lambda: {
            "google_style": True,
            "numpy_style": True,
            "include_init_with_doc": True,
            "include_private_with_doc": False,
            "include_special_with_doc": True,
        }
    )
    custom_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Add default extensions if none specified."""
        if not self.extensions:
            self.extensions = self.get_default_extensions()

    @staticmethod
    def get_default_extensions() -> list[SphinxExtension]:
        """Get default Sphinx extensions for HyperTensor.

        Returns:
            List of default extensions.
        """
        return [
            SphinxExtension("sphinx.ext.autodoc"),
            SphinxExtension("sphinx.ext.autosummary", {"autosummary_generate": True}),
            SphinxExtension("sphinx.ext.napoleon"),
            SphinxExtension("sphinx.ext.viewcode"),
            SphinxExtension("sphinx.ext.intersphinx"),
            SphinxExtension("sphinx.ext.mathjax"),
            SphinxExtension("sphinx.ext.todo", {"todo_include_todos": True}),
            SphinxExtension("sphinx.ext.coverage"),
            SphinxExtension("sphinx.ext.graphviz"),
            SphinxExtension("sphinx_copybutton", required=False),
            SphinxExtension(
                "myst_parser",
                {
                    "myst_enable_extensions": [
                        "colon_fence",
                        "deflist",
                        "dollarmath",
                        "fieldlist",
                        "html_admonition",
                        "html_image",
                        "replacements",
                        "smartquotes",
                        "strikethrough",
                        "substitution",
                        "tasklist",
                    ]
                },
                required=False,
            ),
        ]


def generate_conf_py(
    config: SphinxConfig,
    output_path: Path | None = None,
) -> str:
    """Generate Sphinx conf.py content.

    Args:
        config: Sphinx configuration.
        output_path: Optional path to write the file.

    Returns:
        Generated conf.py content.
    """
    lines = [
        '"""',
        f"Sphinx configuration for {config.project}.",
        "",
        "Auto-generated by HyperTensor documentation module.",
        '"""',
        "",
        "import os",
        "import sys",
        "",
        "# Add project root to path",
        'sys.path.insert(0, os.path.abspath("../"))',
        "",
        "# -- Project information -----------------------------------------------------",
        "",
        f'project = "{config.project}"',
        f'copyright = "{config.copyright_year}, {config.author}"',
        f'author = "{config.author}"',
        f'version = "{config.version}"',
        f'release = "{config.release}"',
        "",
        "# -- General configuration ---------------------------------------------------",
        "",
    ]

    # Extensions
    ext_names = [ext.name for ext in config.extensions]
    lines.append("extensions = [")
    for name in ext_names:
        lines.append(f'    "{name}",')
    lines.append("]")
    lines.append("")

    # Templates and static paths
    lines.append(f'templates_path = ["{config.templates_path}"]')
    lines.append(f'source_suffix = "{config.source_suffix}"')
    lines.append(f'master_doc = "{config.master_doc}"')
    lines.append(f'language = "{config.language}"')
    lines.append("")

    # Exclude patterns
    lines.append("exclude_patterns = [")
    for pattern in config.exclude_patterns:
        lines.append(f'    "{pattern}",')
    lines.append("]")
    lines.append("")

    # Pygments
    lines.append(f'pygments_style = "{config.pygments_style}"')
    lines.append("")

    # HTML configuration
    lines.append(
        "# -- Options for HTML output -------------------------------------------------"
    )
    lines.append("")
    lines.append(f'html_theme = "{config.theme.value}"')
    lines.append(f'html_static_path = ["{config.static_path}"]')

    if config.html_logo:
        lines.append(f'html_logo = "{config.html_logo}"')
    if config.html_favicon:
        lines.append(f'html_favicon = "{config.html_favicon}"')

    lines.append("")

    # Theme-specific options
    lines.append("html_theme_options = {")
    if config.theme == SphinxTheme.FURO:
        lines.append('    "sidebar_hide_name": False,')
        lines.append('    "navigation_with_keys": True,')
        lines.append('    "dark_css_variables": {')
        lines.append('        "color-brand-primary": "#6366f1",')
        lines.append('        "color-brand-content": "#6366f1",')
        lines.append("    },")
    elif config.theme == SphinxTheme.RTD:
        lines.append('    "navigation_depth": 4,')
        lines.append('    "collapse_navigation": False,')
    elif config.theme == SphinxTheme.PYDATA:
        lines.append('    "github_url": "https://github.com/tigantic/HyperTensor",')
        lines.append('    "show_prev_next": True,')
    lines.append("}")
    lines.append("")

    # Autodoc configuration
    lines.append(
        "# -- Autodoc configuration ---------------------------------------------------"
    )
    lines.append("")
    lines.append("autodoc_default_options = {")
    for key, value in config.autodoc_options.items():
        if isinstance(value, bool):
            lines.append(f'    "{key}": {str(value)},')
        elif isinstance(value, str):
            lines.append(f'    "{key}": "{value}",')
        else:
            lines.append(f'    "{key}": {value},')
    lines.append("}")
    lines.append("")
    lines.append("autodoc_typehints = 'description'")
    lines.append("autodoc_member_order = 'bysource'")
    lines.append("")

    # Napoleon configuration
    lines.append(
        "# -- Napoleon configuration --------------------------------------------------"
    )
    lines.append("")
    for key, value in config.napoleon_options.items():
        var_name = f"napoleon_{key}"
        lines.append(f"{var_name} = {str(value)}")
    lines.append("")

    # Intersphinx configuration
    lines.append(
        "# -- Intersphinx configuration -----------------------------------------------"
    )
    lines.append("")
    lines.append("intersphinx_mapping = {")
    for name, (url, inv) in config.intersphinx_mapping.items():
        inv_str = f'"{inv}"' if inv else "None"
        lines.append(f'    "{name}": ("{url}", {inv_str}),')
    lines.append("}")
    lines.append("")

    # Extension-specific configuration
    for ext in config.extensions:
        if ext.config:
            lines.append(f"# -- {ext.name} configuration")
            for key, value in ext.config.items():
                if isinstance(value, bool):
                    lines.append(f"{key} = {str(value)}")
                elif isinstance(value, str):
                    lines.append(f'{key} = "{value}"')
                elif isinstance(value, list):
                    lines.append(f"{key} = [")
                    for item in value:
                        lines.append(f'    "{item}",')
                    lines.append("]")
                else:
                    lines.append(f"{key} = {value!r}")
            lines.append("")

    # Custom configuration
    if config.custom_config:
        lines.append(
            "# -- Custom configuration ----------------------------------------------------"
        )
        lines.append("")
        for key, value in config.custom_config.items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value!r}")
        lines.append("")

    # LaTeX configuration for PDF
    lines.append(
        "# -- LaTeX configuration -----------------------------------------------------"
    )
    lines.append("")
    lines.append("latex_elements = {")
    lines.append('    "papersize": "letterpaper",')
    lines.append('    "pointsize": "11pt",')
    lines.append('    "preamble": r"""')
    lines.append(r"        \usepackage{amsmath}")
    lines.append(r"        \usepackage{amssymb}")
    lines.append('    """,')
    lines.append("}")
    lines.append("")
    lines.append("latex_documents = [")
    lines.append(
        f'    ("{config.master_doc}", "{config.project}.tex", "{config.project} Documentation",'
    )
    lines.append(f'     "{config.author}", "manual"),')
    lines.append("]")
    lines.append("")

    content = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)

    return content


def generate_index_rst(
    config: SphinxConfig,
    modules: list[str],
    output_path: Path | None = None,
) -> str:
    """Generate index.rst content.

    Args:
        config: Sphinx configuration.
        modules: List of module names to document.
        output_path: Optional path to write the file.

    Returns:
        Generated index.rst content.
    """
    lines = [
        f"{config.project} Documentation",
        "=" * (len(config.project) + 14),
        "",
        f"Welcome to {config.project}'s documentation!",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "   :caption: Contents:",
        "",
        "   getting_started",
        "   tutorials/index",
        "   api/index",
        "   examples/index",
        "",
        "User Guides",
        "-----------",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "",
        "   guides/tensor_networks",
        "   guides/cfd",
        "   guides/deployment",
        "",
        "API Reference",
        "-------------",
        "",
        ".. toctree::",
        "   :maxdepth: 3",
        "",
    ]

    for module in modules:
        lines.append(f"   api/{module}")

    lines.extend(
        [
            "",
            "Indices and tables",
            "==================",
            "",
            "* :ref:`genindex`",
            "* :ref:`modindex`",
            "* :ref:`search`",
        ]
    )

    content = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)

    return content


def generate_makefile(
    docs_dir: Path,
    output_path: Path | None = None,
) -> str:
    """Generate Makefile for Sphinx.

    Args:
        docs_dir: Documentation source directory.
        output_path: Optional path to write the file.

    Returns:
        Generated Makefile content.
    """
    content = f"""# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = {docs_dir}
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
\t@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
\t@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets
livehtml:
\tsphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" --watch ../ontic

clean:
\trm -rf $(BUILDDIR)/*
"""

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)

    return content


@dataclass
class BuildResult:
    """Result of a Sphinx build.

    Attributes:
        success: Whether the build succeeded.
        output_dir: Path to the output directory.
        format: Output format.
        warnings: List of warning messages.
        errors: List of error messages.
        duration_seconds: Build duration in seconds.
    """

    success: bool
    output_dir: Path
    format: OutputFormat
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class SphinxBuilder:
    """Builder class for Sphinx documentation.

    This class handles the full documentation build pipeline including
    configuration, generation, and multi-format output.

    Attributes:
        config: Sphinx configuration.
        source_dir: Documentation source directory.
        build_dir: Build output directory.
    """

    def __init__(
        self,
        config: SphinxConfig,
        source_dir: str | Path,
        build_dir: str | Path | None = None,
    ):
        """Initialize the builder.

        Args:
            config: Sphinx configuration.
            source_dir: Documentation source directory.
            build_dir: Build output directory (default: source_dir/_build).
        """
        self.config = config
        self.source_dir = Path(source_dir)
        self.build_dir = Path(build_dir) if build_dir else self.source_dir / "_build"

    def setup(self, modules: list[str] | None = None) -> None:
        """Set up the documentation structure.

        Creates necessary directories and generates configuration files.

        Args:
            modules: List of module names to document.
        """
        # Create directories
        self.source_dir.mkdir(parents=True, exist_ok=True)
        (self.source_dir / "_static").mkdir(exist_ok=True)
        (self.source_dir / "_templates").mkdir(exist_ok=True)
        (self.source_dir / "api").mkdir(exist_ok=True)
        (self.source_dir / "guides").mkdir(exist_ok=True)
        (self.source_dir / "tutorials").mkdir(exist_ok=True)
        (self.source_dir / "examples").mkdir(exist_ok=True)

        # Generate conf.py
        generate_conf_py(self.config, self.source_dir / "conf.py")

        # Generate index.rst
        modules = modules or ["ontic"]
        generate_index_rst(self.config, modules, self.source_dir / "index.rst")

        # Generate Makefile
        generate_makefile(self.source_dir, self.source_dir.parent / "Makefile")

    def build(
        self,
        format: OutputFormat = OutputFormat.HTML,
        clean: bool = False,
        verbose: bool = False,
    ) -> BuildResult:
        """Build the documentation.

        Args:
            format: Output format to build.
            clean: Whether to clean the build directory first.
            verbose: Whether to show verbose output.

        Returns:
            BuildResult with success status and metadata.
        """
        import time

        output_dir = self.build_dir / format.value

        if clean and output_dir.exists():
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            "sphinx-build",
            "-b",
            format.value,
            str(self.source_dir),
            str(output_dir),
        ]

        if verbose:
            cmd.append("-v")

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            duration = time.time() - start_time

            # Parse warnings and errors
            warnings = []
            errors = []
            for line in result.stderr.split("\n"):
                if "WARNING" in line:
                    warnings.append(line)
                elif "ERROR" in line or result.returncode != 0:
                    errors.append(line)

            return BuildResult(
                success=result.returncode == 0,
                output_dir=output_dir,
                format=format,
                warnings=warnings,
                errors=(
                    errors
                    if errors
                    else [result.stderr] if result.returncode != 0 else []
                ),
                duration_seconds=duration,
            )

        except FileNotFoundError:
            return BuildResult(
                success=False,
                output_dir=output_dir,
                format=format,
                errors=["sphinx-build not found. Install Sphinx: pip install sphinx"],
            )

    def build_all(
        self,
        formats: list[OutputFormat] | None = None,
        clean: bool = False,
    ) -> dict[OutputFormat, BuildResult]:
        """Build documentation in multiple formats.

        Args:
            formats: List of output formats (default: HTML, PDF).
            clean: Whether to clean the build directory first.

        Returns:
            Dictionary mapping format to BuildResult.
        """
        formats = formats or [OutputFormat.HTML, OutputFormat.PDF]
        results = {}

        for fmt in formats:
            results[fmt] = self.build(fmt, clean=clean)

        return results

    def serve(self, port: int = 8000) -> None:
        """Serve the HTML documentation locally.

        Args:
            port: Port to serve on.
        """
        html_dir = self.build_dir / "html"

        if not html_dir.exists():
            result = self.build(OutputFormat.HTML)
            if not result.success:
                raise RuntimeError(f"Build failed: {result.errors}")

        import http.server
        import socketserver

        os.chdir(html_dir)

        Handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"Serving documentation at http://localhost:{port}")
            print("Press Ctrl+C to stop")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                pass


def build_documentation(
    project_root: str | Path,
    project_name: str = "HyperTensor",
    output_formats: list[OutputFormat] | None = None,
    theme: SphinxTheme = SphinxTheme.FURO,
    clean: bool = True,
) -> dict[OutputFormat, BuildResult]:
    """High-level function to build documentation.

    Args:
        project_root: Root directory of the project.
        project_name: Name of the project.
        output_formats: Output formats to build.
        theme: Sphinx theme to use.
        clean: Whether to clean build directory.

    Returns:
        Dictionary mapping format to BuildResult.
    """
    project_root = Path(project_root)
    docs_dir = project_root / "docs" / "source"

    config = SphinxConfig(
        project=project_name,
        theme=theme,
        version="2.2.0",
        release="2.2.0",
    )

    builder = SphinxBuilder(
        config=config,
        source_dir=docs_dir,
        build_dir=project_root / "docs" / "_build",
    )

    # Setup documentation structure
    builder.setup(modules=["ontic"])

    # Build all formats
    return builder.build_all(formats=output_formats, clean=clean)
