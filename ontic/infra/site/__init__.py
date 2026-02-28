"""
Static documentation site generation module.

This module provides infrastructure for generating static documentation
sites from the ontic codebase, including:
- Sphinx-compatible configuration
- Static site generation (HTML, PDF)
- Search index generation
- API documentation rendering
- Tutorial and guide rendering
"""

from .assets import Asset, AssetManager, AssetType, optimize_images, process_assets
from .generator import (
                     Navigation,
                     NavItem,
                     Page,
                     PageType,
                     SiteBuilder,
                     SiteConfig,
                     build_site,
                     serve_site,
)
from .search import Indexer, SearchIndex, SearchResult, build_search_index, search
from .themes import OnticTheme, Theme, ThemeConfig, get_theme, list_themes

__all__ = [
    # Generator
    "SiteConfig",
    "SiteBuilder",
    "PageType",
    "Page",
    "Navigation",
    "NavItem",
    "build_site",
    "serve_site",
    # Themes
    "Theme",
    "ThemeConfig",
    "get_theme",
    "list_themes",
    "OnticTheme",
    # Search
    "SearchIndex",
    "SearchResult",
    "Indexer",
    "build_search_index",
    "search",
    # Assets
    "AssetType",
    "Asset",
    "AssetManager",
    "process_assets",
    "optimize_images",
]
