"""
Static documentation site generation module.

This module provides infrastructure for generating static documentation
sites from the tensornet codebase, including:
- Sphinx-compatible configuration
- Static site generation (HTML, PDF)
- Search index generation
- API documentation rendering
- Tutorial and guide rendering
"""

from .generator import (
    SiteConfig,
    SiteBuilder,
    PageType,
    Page,
    Navigation,
    NavItem,
    build_site,
    serve_site,
)
from .themes import (
    Theme,
    ThemeConfig,
    get_theme,
    list_themes,
    HyperTensorTheme,
)
from .search import (
    SearchIndex,
    SearchResult,
    Indexer,
    build_search_index,
    search,
)
from .assets import (
    AssetType,
    Asset,
    AssetManager,
    process_assets,
    optimize_images,
)

__all__ = [
    # Generator
    'SiteConfig',
    'SiteBuilder',
    'PageType',
    'Page',
    'Navigation',
    'NavItem',
    'build_site',
    'serve_site',
    # Themes
    'Theme',
    'ThemeConfig',
    'get_theme',
    'list_themes',
    'HyperTensorTheme',
    # Search
    'SearchIndex',
    'SearchResult',
    'Indexer',
    'build_search_index',
    'search',
    # Assets
    'AssetType',
    'Asset',
    'AssetManager',
    'process_assets',
    'optimize_images',
]
