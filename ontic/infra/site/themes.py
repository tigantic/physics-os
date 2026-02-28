"""
Documentation site themes.

This module provides theming support for HyperTensor documentation sites.
"""

from dataclasses import dataclass, field
from enum import Enum, auto


class ColorScheme(Enum):
    """Color scheme options."""

    LIGHT = auto()
    DARK = auto()
    AUTO = auto()


@dataclass
class ThemeColors:
    """Theme color palette."""

    primary: str = "#0066cc"
    secondary: str = "#6c757d"
    success: str = "#28a745"
    warning: str = "#ffc107"
    danger: str = "#dc3545"
    info: str = "#17a2b8"

    background: str = "#ffffff"
    surface: str = "#f8f9fa"
    text: str = "#212529"
    text_muted: str = "#6c757d"
    border: str = "#dee2e6"

    code_background: str = "#1a1a2e"
    code_text: str = "#f8f8f2"

    nav_background: str = "#1a1a2e"
    nav_text: str = "#ffffff"

    def to_css_vars(self) -> str:
        """Convert to CSS custom properties."""
        return f"""
:root {{
    --color-primary: {self.primary};
    --color-secondary: {self.secondary};
    --color-success: {self.success};
    --color-warning: {self.warning};
    --color-danger: {self.danger};
    --color-info: {self.info};
    --color-bg: {self.background};
    --color-surface: {self.surface};
    --color-text: {self.text};
    --color-text-muted: {self.text_muted};
    --color-border: {self.border};
    --color-code-bg: {self.code_background};
    --color-code-text: {self.code_text};
    --color-nav-bg: {self.nav_background};
    --color-nav-text: {self.nav_text};
}}
"""


@dataclass
class ThemeTypography:
    """Typography configuration."""

    font_family: str = (
        "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    )
    font_family_mono: str = "'Fira Code', 'Consolas', 'Monaco', monospace"

    font_size_base: str = "16px"
    font_size_sm: str = "14px"
    font_size_lg: str = "18px"

    line_height: float = 1.6
    line_height_heading: float = 1.3

    heading_weight: int = 600

    def to_css(self) -> str:
        """Convert to CSS."""
        return f"""
body {{
    font-family: {self.font_family};
    font-size: {self.font_size_base};
    line-height: {self.line_height};
}}

code, pre {{
    font-family: {self.font_family_mono};
}}

h1, h2, h3, h4, h5, h6 {{
    font-weight: {self.heading_weight};
    line-height: {self.line_height_heading};
}}
"""


@dataclass
class ThemeLayout:
    """Layout configuration."""

    max_width: str = "1400px"
    content_width: str = "900px"
    sidebar_width: str = "280px"
    toc_width: str = "250px"

    nav_height: str = "64px"
    footer_height: str = "auto"

    spacing_unit: str = "8px"
    border_radius: str = "8px"

    def to_css(self) -> str:
        """Convert to CSS."""
        return f"""
.container {{
    max-width: {self.max_width};
    margin: 0 auto;
    padding: 0 calc({self.spacing_unit} * 3);
}}

.content-wrapper {{
    display: grid;
    grid-template-columns: {self.sidebar_width} 1fr {self.toc_width};
    gap: calc({self.spacing_unit} * 4);
}}

article {{
    max-width: {self.content_width};
}}

.site-nav {{
    height: {self.nav_height};
}}

* {{
    border-radius: {self.border_radius};
}}
"""


@dataclass
class ThemeConfig:
    """Complete theme configuration."""

    name: str = "default"
    colors: ThemeColors = field(default_factory=ThemeColors)
    typography: ThemeTypography = field(default_factory=ThemeTypography)
    layout: ThemeLayout = field(default_factory=ThemeLayout)
    color_scheme: ColorScheme = ColorScheme.LIGHT

    custom_css: str = ""
    custom_js: str = ""

    # Feature flags
    enable_dark_mode: bool = True
    enable_search: bool = True
    enable_toc: bool = True
    enable_breadcrumbs: bool = True
    enable_edit_link: bool = True
    enable_prev_next: bool = True

    def to_css(self) -> str:
        """Generate complete CSS for theme."""
        css_parts = [
            self.colors.to_css_vars(),
            self.typography.to_css(),
            self.layout.to_css(),
        ]

        if self.custom_css:
            css_parts.append(self.custom_css)

        return "\n".join(css_parts)


class Theme:
    """
    Documentation site theme.

    Provides styling and layout for the documentation site.
    """

    def __init__(self, config: ThemeConfig | None = None):
        """
        Initialize theme.

        Args:
            config: Theme configuration
        """
        self.config = config or ThemeConfig()

    @property
    def name(self) -> str:
        """Get theme name."""
        return self.config.name

    def get_css(self) -> str:
        """Get complete CSS for theme."""
        return self.config.to_css() + self._get_base_styles()

    def get_dark_mode_css(self) -> str:
        """Get dark mode CSS overrides."""
        if not self.config.enable_dark_mode:
            return ""

        return """
@media (prefers-color-scheme: dark) {
    :root {
        --color-bg: #1a1a2e;
        --color-surface: #16213e;
        --color-text: #eaeaea;
        --color-text-muted: #a0a0a0;
        --color-border: #3a3a5c;
    }
}

[data-theme="dark"] {
    --color-bg: #1a1a2e;
    --color-surface: #16213e;
    --color-text: #eaeaea;
    --color-text-muted: #a0a0a0;
    --color-border: #3a3a5c;
}
"""

    def _get_base_styles(self) -> str:
        """Get base styles that apply to all themes."""
        return """
/* Reset */
*, *::before, *::after {
    box-sizing: border-box;
}

body {
    margin: 0;
    padding: 0;
    background: var(--color-bg);
    color: var(--color-text);
}

/* Links */
a {
    color: var(--color-primary);
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Code */
code {
    background: var(--color-surface);
    padding: 0.2em 0.4em;
    border-radius: 4px;
    font-size: 0.9em;
}

pre {
    background: var(--color-code-bg);
    color: var(--color-code-text);
    padding: 1rem;
    overflow-x: auto;
    margin: 1rem 0;
}

pre code {
    background: none;
    padding: 0;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
}

th, td {
    padding: 0.75rem;
    text-align: left;
    border-bottom: 1px solid var(--color-border);
}

th {
    background: var(--color-surface);
    font-weight: 600;
}

/* Blockquotes */
blockquote {
    margin: 1rem 0;
    padding: 1rem;
    border-left: 4px solid var(--color-primary);
    background: var(--color-surface);
}

/* Images */
img {
    max-width: 100%;
    height: auto;
}

/* Alerts/Admonitions */
.admonition {
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 8px;
    border-left: 4px solid;
}

.admonition.note {
    background: rgba(23, 162, 184, 0.1);
    border-color: var(--color-info);
}

.admonition.warning {
    background: rgba(255, 193, 7, 0.1);
    border-color: var(--color-warning);
}

.admonition.danger {
    background: rgba(220, 53, 69, 0.1);
    border-color: var(--color-danger);
}

.admonition.tip {
    background: rgba(40, 167, 69, 0.1);
    border-color: var(--color-success);
}
"""


class HyperTensorTheme(Theme):
    """
    Custom theme for HyperTensor documentation.

    Features aerospace-inspired design with emphasis on
    technical documentation clarity.
    """

    def __init__(self):
        """Initialize HyperTensor theme."""
        config = ThemeConfig(
            name="physics_os",
            colors=ThemeColors(
                primary="#0a84ff",
                secondary="#5e5ce6",
                success="#30d158",
                warning="#ffd60a",
                danger="#ff453a",
                info="#64d2ff",
                background="#ffffff",
                surface="#f5f5f7",
                text="#1d1d1f",
                text_muted="#86868b",
                border="#d2d2d7",
                code_background="#1c1c1e",
                code_text="#f5f5f7",
                nav_background="#1c1c1e",
                nav_text="#f5f5f7",
            ),
            typography=ThemeTypography(
                font_family="'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                font_family_mono="'SF Mono', 'Fira Code', 'Consolas', monospace",
                font_size_base="17px",
                line_height=1.65,
            ),
            layout=ThemeLayout(
                max_width="1440px",
                content_width="860px",
                sidebar_width="300px",
                toc_width="260px",
                nav_height="52px",
                border_radius="12px",
            ),
        )
        super().__init__(config)

    def get_css(self) -> str:
        """Get HyperTensor theme CSS."""
        base = super().get_css()

        custom = """
/* HyperTensor Theme Customizations */

.site-nav {
    background: linear-gradient(135deg, #1c1c1e 0%, #2c2c2e 100%);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.site-nav .logo {
    font-weight: 700;
    font-size: 1.25rem;
    background: linear-gradient(135deg, #0a84ff 0%, #5e5ce6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h1 {
    font-size: 2.5rem;
    background: linear-gradient(135deg, #1d1d1f 0%, #3a3a3c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.api-signature {
    background: var(--color-surface);
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid var(--color-border);
    font-family: var(--font-mono);
}

.api-param {
    display: grid;
    grid-template-columns: 150px 1fr;
    gap: 1rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--color-border);
}

.api-param:last-child {
    border-bottom: none;
}

.badge {
    display: inline-block;
    padding: 0.25em 0.5em;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 6px;
    text-transform: uppercase;
}

.badge-new { background: var(--color-success); color: white; }
.badge-deprecated { background: var(--color-warning); color: black; }
.badge-experimental { background: var(--color-info); color: white; }

/* Tensor network diagram styling */
.tensor-diagram {
    display: flex;
    justify-content: center;
    padding: 2rem;
    background: var(--color-surface);
    border-radius: 12px;
    margin: 1.5rem 0;
}

.tensor-node {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    margin: 0 1rem;
}

.tensor-node.mps { background: #0a84ff; color: white; }
.tensor-node.mpo { background: #5e5ce6; color: white; }
.tensor-node.cfd { background: #30d158; color: white; }

/* Math equation styling */
.math-block {
    background: var(--color-surface);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1.5rem 0;
    overflow-x: auto;
}

/* Benchmark results table */
.benchmark-table {
    font-variant-numeric: tabular-nums;
}

.benchmark-table .improved {
    color: var(--color-success);
    font-weight: 600;
}

.benchmark-table .regressed {
    color: var(--color-danger);
    font-weight: 600;
}
"""
        return base + custom


# Theme registry
_themes: dict[str, Theme] = {
    "default": Theme(),
    "physics_os": HyperTensorTheme(),
}


def get_theme(name: str) -> Theme:
    """
    Get theme by name.

    Args:
        name: Theme name

    Returns:
        Theme instance

    Raises:
        KeyError: If theme not found
    """
    if name not in _themes:
        raise KeyError(f"Theme '{name}' not found. Available: {list(_themes.keys())}")
    return _themes[name]


def list_themes() -> list[str]:
    """
    List available theme names.

    Returns:
        List of theme names
    """
    return list(_themes.keys())


def register_theme(name: str, theme: Theme):
    """
    Register a custom theme.

    Args:
        name: Theme name
        theme: Theme instance
    """
    _themes[name] = theme
