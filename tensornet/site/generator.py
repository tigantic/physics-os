"""
Static site generator for HyperTensor documentation.

This module provides the core site generation functionality including:
- Page rendering from Markdown/RST
- Navigation generation
- Template processing
- Multi-format output (HTML, PDF)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum, auto
from pathlib import Path
import json
import re
import time
import shutil
import hashlib


class PageType(Enum):
    """Type of documentation page."""
    API_REFERENCE = auto()
    TUTORIAL = auto()
    GUIDE = auto()
    EXAMPLE = auto()
    CHANGELOG = auto()
    INDEX = auto()
    CUSTOM = auto()


@dataclass
class NavItem:
    """Navigation item in the site structure."""
    title: str
    path: str
    children: List['NavItem'] = field(default_factory=list)
    icon: str = ""
    external: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'path': self.path,
            'children': [c.to_dict() for c in self.children],
            'icon': self.icon,
            'external': self.external,
        }


@dataclass
class Navigation:
    """Site navigation structure."""
    items: List[NavItem] = field(default_factory=list)
    footer_items: List[NavItem] = field(default_factory=list)
    
    def add_item(self, item: NavItem, parent_path: Optional[str] = None):
        """Add navigation item."""
        if parent_path is None:
            self.items.append(item)
        else:
            self._add_to_parent(self.items, parent_path, item)
    
    def _add_to_parent(
        self, 
        items: List[NavItem], 
        parent_path: str, 
        item: NavItem
    ) -> bool:
        """Add item under parent path."""
        for nav_item in items:
            if nav_item.path == parent_path:
                nav_item.children.append(item)
                return True
            if self._add_to_parent(nav_item.children, parent_path, item):
                return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'items': [i.to_dict() for i in self.items],
            'footer_items': [i.to_dict() for i in self.footer_items],
        }


@dataclass
class Page:
    """Documentation page."""
    path: str
    title: str
    content: str
    page_type: PageType = PageType.CUSTOM
    metadata: Dict[str, Any] = field(default_factory=dict)
    template: str = "default"
    toc: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate TOC from content if not provided."""
        if not self.toc and self.content:
            self.toc = self._extract_toc()
    
    def _extract_toc(self) -> List[Dict[str, Any]]:
        """Extract table of contents from markdown headers."""
        toc = []
        # Match ## and ### headers
        pattern = r'^(#{2,4})\s+(.+)$'
        for match in re.finditer(pattern, self.content, re.MULTILINE):
            level = len(match.group(1)) - 1  # ## = 1, ### = 2, #### = 3
            title = match.group(2).strip()
            slug = self._slugify(title)
            toc.append({
                'level': level,
                'title': title,
                'slug': slug,
            })
        return toc
    
    def _slugify(self, text: str) -> str:
        """Convert text to URL-safe slug."""
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        return text.strip('-')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'path': self.path,
            'title': self.title,
            'content': self.content,
            'page_type': self.page_type.name,
            'metadata': self.metadata,
            'template': self.template,
            'toc': self.toc,
        }


@dataclass
class SiteConfig:
    """Configuration for static site generation."""
    title: str = "HyperTensor Documentation"
    description: str = "Quantum-inspired tensor network framework for hypersonic CFD"
    version: str = "2.5.0"
    base_url: str = "/"
    output_dir: str = "_site"
    source_dir: str = "docs"
    theme: str = "hypertensor"
    
    # Build options
    minify_html: bool = True
    minify_css: bool = True
    minify_js: bool = True
    generate_sitemap: bool = True
    generate_search_index: bool = True
    
    # Content options
    syntax_highlighting: bool = True
    math_rendering: bool = True
    mermaid_diagrams: bool = True
    
    # Metadata
    author: str = "HyperTensor Team"
    repository: str = "https://github.com/tigantic/HyperTensor"
    license: str = "Proprietary - Tigantic Holdings LLC"
    
    # Navigation
    nav_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'description': self.description,
            'version': self.version,
            'base_url': self.base_url,
            'output_dir': self.output_dir,
            'source_dir': self.source_dir,
            'theme': self.theme,
            'minify_html': self.minify_html,
            'minify_css': self.minify_css,
            'minify_js': self.minify_js,
            'generate_sitemap': self.generate_sitemap,
            'generate_search_index': self.generate_search_index,
            'syntax_highlighting': self.syntax_highlighting,
            'math_rendering': self.math_rendering,
            'mermaid_diagrams': self.mermaid_diagrams,
            'author': self.author,
            'repository': self.repository,
            'license': self.license,
        }


@dataclass
class BuildResult:
    """Result of site build."""
    success: bool
    pages_built: int
    assets_processed: int
    build_time: float
    output_dir: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'pages_built': self.pages_built,
            'assets_processed': self.assets_processed,
            'build_time': self.build_time,
            'output_dir': self.output_dir,
            'errors': self.errors,
            'warnings': self.warnings,
        }


class MarkdownRenderer:
    """Render Markdown to HTML."""
    
    def __init__(self, config: SiteConfig):
        """Initialize renderer."""
        self.config = config
    
    def render(self, content: str) -> str:
        """
        Render markdown to HTML.
        
        This is a simplified renderer. In production, use a full
        markdown library like mistune or markdown-it.
        """
        html = content
        
        # Headers
        html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        
        # Code blocks
        html = re.sub(
            r'```(\w+)?\n(.*?)```',
            lambda m: self._render_code_block(m.group(2), m.group(1)),
            html,
            flags=re.DOTALL
        )
        
        # Inline code
        html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
        
        # Bold and italic
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        
        # Links
        html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)
        
        # Lists
        html = self._render_lists(html)
        
        # Paragraphs
        html = self._render_paragraphs(html)
        
        # Math (KaTeX)
        if self.config.math_rendering:
            html = self._render_math(html)
        
        return html
    
    def _render_code_block(self, code: str, language: Optional[str]) -> str:
        """Render code block with syntax highlighting."""
        lang_class = f' class="language-{language}"' if language else ''
        escaped = code.replace('<', '&lt;').replace('>', '&gt;')
        return f'<pre><code{lang_class}>{escaped}</code></pre>'
    
    def _render_lists(self, html: str) -> str:
        """Render unordered and ordered lists."""
        lines = html.split('\n')
        result = []
        in_ul = False
        in_ol = False
        
        for line in lines:
            ul_match = re.match(r'^[-*]\s+(.+)$', line)
            ol_match = re.match(r'^\d+\.\s+(.+)$', line)
            
            if ul_match:
                if not in_ul:
                    result.append('<ul>')
                    in_ul = True
                result.append(f'<li>{ul_match.group(1)}</li>')
            elif ol_match:
                if not in_ol:
                    result.append('<ol>')
                    in_ol = True
                result.append(f'<li>{ol_match.group(1)}</li>')
            else:
                if in_ul:
                    result.append('</ul>')
                    in_ul = False
                if in_ol:
                    result.append('</ol>')
                    in_ol = False
                result.append(line)
        
        if in_ul:
            result.append('</ul>')
        if in_ol:
            result.append('</ol>')
        
        return '\n'.join(result)
    
    def _render_paragraphs(self, html: str) -> str:
        """Wrap loose text in paragraph tags."""
        lines = html.split('\n\n')
        result = []
        
        for block in lines:
            block = block.strip()
            if not block:
                continue
            # Skip blocks that are already wrapped in tags
            if block.startswith('<'):
                result.append(block)
            else:
                result.append(f'<p>{block}</p>')
        
        return '\n\n'.join(result)
    
    def _render_math(self, html: str) -> str:
        """Render LaTeX math expressions."""
        # Block math
        html = re.sub(
            r'\$\$(.+?)\$\$',
            r'<div class="math-block">\1</div>',
            html,
            flags=re.DOTALL
        )
        # Inline math
        html = re.sub(
            r'\$([^$]+)\$',
            r'<span class="math-inline">\1</span>',
            html
        )
        return html


class TemplateEngine:
    """Simple template engine for HTML generation."""
    
    def __init__(self):
        """Initialize template engine."""
        self.templates: Dict[str, str] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default templates."""
        self.templates['base'] = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - {{ site_title }}</title>
    <meta name="description" content="{{ description }}">
    <link rel="stylesheet" href="{{ base_url }}css/style.css">
    {% if math_rendering %}
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css">
    {% endif %}
    {% if syntax_highlighting %}
    <link rel="stylesheet" href="{{ base_url }}css/highlight.css">
    {% endif %}
</head>
<body>
    <nav class="site-nav">
        {{ navigation }}
    </nav>
    <main class="content">
        <article>
            {{ content }}
        </article>
        {% if toc %}
        <aside class="toc">
            <h4>On this page</h4>
            {{ toc }}
        </aside>
        {% endif %}
    </main>
    <footer class="site-footer">
        <p>&copy; {{ year }} {{ author }}. {{ license }} License.</p>
    </footer>
    {% if syntax_highlighting %}
    <script src="{{ base_url }}js/highlight.min.js"></script>
    <script>hljs.highlightAll();</script>
    {% endif %}
    {% if math_rendering %}
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.js"></script>
    <script>
        document.querySelectorAll('.math-inline, .math-block').forEach(el => {
            katex.render(el.textContent, el, {displayMode: el.classList.contains('math-block')});
        });
    </script>
    {% endif %}
</body>
</html>'''
        
        self.templates['default'] = self.templates['base']
        
        self.templates['api'] = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - API Reference - {{ site_title }}</title>
    <link rel="stylesheet" href="{{ base_url }}css/style.css">
    <link rel="stylesheet" href="{{ base_url }}css/api.css">
</head>
<body class="api-page">
    <nav class="site-nav">{{ navigation }}</nav>
    <aside class="api-sidebar">{{ sidebar }}</aside>
    <main class="api-content">
        <article>{{ content }}</article>
    </main>
</body>
</html>'''
    
    def render(
        self, 
        template_name: str, 
        context: Dict[str, Any]
    ) -> str:
        """Render template with context."""
        template = self.templates.get(template_name, self.templates['default'])
        
        # Simple template rendering
        result = template
        
        # Replace variables
        for key, value in context.items():
            if isinstance(value, bool):
                # Handle conditionals
                pattern = r'{%\s*if\s+' + key + r'\s*%}(.+?){%\s*endif\s*%}'
                if value:
                    result = re.sub(pattern, r'\1', result, flags=re.DOTALL)
                else:
                    result = re.sub(pattern, '', result, flags=re.DOTALL)
            else:
                result = result.replace('{{ ' + key + ' }}', str(value))
        
        # Clean up remaining conditionals
        result = re.sub(r'{%\s*if\s+\w+\s*%}.*?{%\s*endif\s*%}', '', result, flags=re.DOTALL)
        result = re.sub(r'{{\s*\w+\s*}}', '', result)
        
        return result
    
    def add_template(self, name: str, content: str):
        """Add custom template."""
        self.templates[name] = content


class SiteBuilder:
    """
    Static site builder for HyperTensor documentation.
    
    Generates a complete static documentation site from source files.
    """
    
    def __init__(self, config: Optional[SiteConfig] = None):
        """
        Initialize site builder.
        
        Args:
            config: Site configuration
        """
        self.config = config or SiteConfig()
        self.pages: List[Page] = []
        self.navigation = Navigation()
        self.renderer = MarkdownRenderer(self.config)
        self.template_engine = TemplateEngine()
        self.assets: List[str] = []
    
    def add_page(self, page: Page):
        """Add page to site."""
        self.pages.append(page)
    
    def add_pages_from_directory(self, directory: Union[str, Path]):
        """
        Add pages from markdown files in directory.
        
        Args:
            directory: Path to directory with markdown files
        """
        path = Path(directory)
        if not path.exists():
            return
        
        for md_file in path.rglob('*.md'):
            content = md_file.read_text(encoding='utf-8')
            
            # Extract frontmatter
            metadata = {}
            if content.startswith('---'):
                end = content.find('---', 3)
                if end > 0:
                    frontmatter = content[3:end].strip()
                    for line in frontmatter.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            metadata[key.strip()] = value.strip()
                    content = content[end + 3:].strip()
            
            # Determine page type
            page_type = PageType.CUSTOM
            if 'api' in str(md_file).lower():
                page_type = PageType.API_REFERENCE
            elif 'tutorial' in str(md_file).lower():
                page_type = PageType.TUTORIAL
            elif 'guide' in str(md_file).lower():
                page_type = PageType.GUIDE
            elif 'example' in str(md_file).lower():
                page_type = PageType.EXAMPLE
            
            rel_path = md_file.relative_to(path)
            page_path = str(rel_path).replace('.md', '.html').replace('\\', '/')
            
            page = Page(
                path=page_path,
                title=metadata.get('title', md_file.stem.replace('_', ' ').title()),
                content=content,
                page_type=page_type,
                metadata=metadata,
            )
            self.add_page(page)
    
    def build_navigation(self):
        """Build navigation from pages."""
        # Group pages by type
        sections = {
            'Getting Started': [],
            'API Reference': [],
            'Tutorials': [],
            'Guides': [],
            'Examples': [],
        }
        
        for page in self.pages:
            if page.page_type == PageType.API_REFERENCE:
                sections['API Reference'].append(page)
            elif page.page_type == PageType.TUTORIAL:
                sections['Tutorials'].append(page)
            elif page.page_type == PageType.GUIDE:
                sections['Guides'].append(page)
            elif page.page_type == PageType.EXAMPLE:
                sections['Examples'].append(page)
            else:
                sections['Getting Started'].append(page)
        
        # Build navigation items
        for section_name, pages in sections.items():
            if not pages:
                continue
            
            children = [
                NavItem(title=p.title, path=p.path)
                for p in sorted(pages, key=lambda x: x.title)
            ]
            
            section_item = NavItem(
                title=section_name,
                path=f"#{section_name.lower().replace(' ', '-')}",
                children=children,
            )
            self.navigation.add_item(section_item)
    
    def _render_navigation_html(self) -> str:
        """Render navigation to HTML."""
        html_parts = ['<ul class="nav-list">']
        
        for item in self.navigation.items:
            html_parts.append(self._render_nav_item(item))
        
        html_parts.append('</ul>')
        return '\n'.join(html_parts)
    
    def _render_nav_item(self, item: NavItem) -> str:
        """Render single navigation item."""
        html = f'<li class="nav-item">'
        html += f'<a href="{item.path}">{item.title}</a>'
        
        if item.children:
            html += '<ul class="nav-children">'
            for child in item.children:
                html += self._render_nav_item(child)
            html += '</ul>'
        
        html += '</li>'
        return html
    
    def _render_toc_html(self, toc: List[Dict[str, Any]]) -> str:
        """Render table of contents to HTML."""
        if not toc:
            return ''
        
        html_parts = ['<ul class="toc-list">']
        for item in toc:
            indent = '  ' * item['level']
            html_parts.append(
                f'{indent}<li class="toc-level-{item["level"]}">'
                f'<a href="#{item["slug"]}">{item["title"]}</a></li>'
            )
        html_parts.append('</ul>')
        return '\n'.join(html_parts)
    
    def build(self) -> BuildResult:
        """
        Build the static site.
        
        Returns:
            BuildResult with build statistics
        """
        start_time = time.time()
        errors = []
        warnings = []
        pages_built = 0
        assets_processed = 0
        
        output_path = Path(self.config.output_dir)
        
        # Clean output directory
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True)
        
        # Build navigation
        self.build_navigation()
        nav_html = self._render_navigation_html()
        
        # Render pages
        for page in self.pages:
            try:
                # Render markdown to HTML
                content_html = self.renderer.render(page.content)
                toc_html = self._render_toc_html(page.toc)
                
                # Build template context
                context = {
                    'title': page.title,
                    'site_title': self.config.title,
                    'description': page.metadata.get('description', self.config.description),
                    'base_url': self.config.base_url,
                    'navigation': nav_html,
                    'content': content_html,
                    'toc': toc_html,
                    'year': time.strftime('%Y'),
                    'author': self.config.author,
                    'license': self.config.license,
                    'math_rendering': self.config.math_rendering,
                    'syntax_highlighting': self.config.syntax_highlighting,
                    **page.metadata,
                }
                
                # Render full page
                html = self.template_engine.render(page.template, context)
                
                # Minify if enabled
                if self.config.minify_html:
                    html = self._minify_html(html)
                
                # Write page
                page_path = output_path / page.path
                page_path.parent.mkdir(parents=True, exist_ok=True)
                page_path.write_text(html, encoding='utf-8')
                
                pages_built += 1
                
            except Exception as e:
                errors.append(f"Error building {page.path}: {str(e)}")
        
        # Generate CSS
        css_dir = output_path / 'css'
        css_dir.mkdir(exist_ok=True)
        (css_dir / 'style.css').write_text(self._generate_css(), encoding='utf-8')
        assets_processed += 1
        
        # Generate search index
        if self.config.generate_search_index:
            search_index = self._build_search_index()
            (output_path / 'search-index.json').write_text(
                json.dumps(search_index, indent=2),
                encoding='utf-8'
            )
            assets_processed += 1
        
        # Generate sitemap
        if self.config.generate_sitemap:
            sitemap = self._generate_sitemap()
            (output_path / 'sitemap.xml').write_text(sitemap, encoding='utf-8')
            assets_processed += 1
        
        build_time = time.time() - start_time
        
        return BuildResult(
            success=len(errors) == 0,
            pages_built=pages_built,
            assets_processed=assets_processed,
            build_time=build_time,
            output_dir=str(output_path),
            errors=errors,
            warnings=warnings,
        )
    
    def _minify_html(self, html: str) -> str:
        """Minify HTML by removing extra whitespace."""
        # Remove comments
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
        # Collapse whitespace
        html = re.sub(r'\s+', ' ', html)
        # Remove whitespace around tags
        html = re.sub(r'>\s+<', '><', html)
        return html.strip()
    
    def _generate_css(self) -> str:
        """Generate base CSS styles."""
        return '''/* HyperTensor Documentation Styles */
:root {
    --primary: #0066cc;
    --secondary: #6c757d;
    --success: #28a745;
    --danger: #dc3545;
    --warning: #ffc107;
    --bg: #ffffff;
    --text: #212529;
    --code-bg: #f8f9fa;
    --border: #dee2e6;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    line-height: 1.6;
    color: var(--text);
    background: var(--bg);
}

.site-nav {
    background: #1a1a2e;
    color: #fff;
    padding: 1rem 2rem;
    position: sticky;
    top: 0;
    z-index: 100;
}

.nav-list { list-style: none; display: flex; gap: 2rem; }
.nav-item a { color: #fff; text-decoration: none; }
.nav-item a:hover { color: var(--primary); }

.content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
    display: grid;
    grid-template-columns: 1fr 250px;
    gap: 2rem;
}

article { min-width: 0; }

h1, h2, h3, h4 { margin: 1.5rem 0 1rem; color: #1a1a2e; }
h1 { font-size: 2.5rem; border-bottom: 2px solid var(--primary); padding-bottom: 0.5rem; }
h2 { font-size: 1.8rem; }
h3 { font-size: 1.4rem; }

p { margin: 1rem 0; }

a { color: var(--primary); }

code {
    background: var(--code-bg);
    padding: 0.2em 0.4em;
    border-radius: 3px;
    font-family: 'Fira Code', 'Consolas', monospace;
    font-size: 0.9em;
}

pre {
    background: #1a1a2e;
    color: #f8f8f2;
    padding: 1rem;
    border-radius: 8px;
    overflow-x: auto;
    margin: 1rem 0;
}

pre code { background: none; padding: 0; color: inherit; }

.toc {
    position: sticky;
    top: 80px;
    align-self: start;
    padding: 1rem;
    background: var(--code-bg);
    border-radius: 8px;
}

.toc h4 { margin: 0 0 0.5rem; font-size: 0.9rem; color: var(--secondary); }
.toc-list { list-style: none; font-size: 0.9rem; }
.toc-list li { padding: 0.25rem 0; }
.toc-list a { color: var(--text); text-decoration: none; }
.toc-list a:hover { color: var(--primary); }

.site-footer {
    background: #1a1a2e;
    color: #fff;
    text-align: center;
    padding: 2rem;
    margin-top: 4rem;
}

.math-block {
    text-align: center;
    margin: 1.5rem 0;
    overflow-x: auto;
}

@media (max-width: 768px) {
    .content { grid-template-columns: 1fr; }
    .toc { display: none; }
}
'''
    
    def _build_search_index(self) -> Dict[str, Any]:
        """Build search index from pages."""
        documents = []
        for page in self.pages:
            # Extract text content
            text = re.sub(r'<[^>]+>', '', self.renderer.render(page.content))
            text = re.sub(r'\s+', ' ', text).strip()
            
            documents.append({
                'id': page.path,
                'title': page.title,
                'content': text[:500],  # First 500 chars for preview
                'type': page.page_type.name,
            })
        
        return {
            'version': '1.0',
            'generated': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'documents': documents,
        }
    
    def _generate_sitemap(self) -> str:
        """Generate XML sitemap."""
        urls = []
        for page in self.pages:
            urls.append(f'''  <url>
    <loc>{self.config.base_url}{page.path}</loc>
    <lastmod>{time.strftime('%Y-%m-%d')}</lastmod>
    <priority>0.8</priority>
  </url>''')
        
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{chr(10).join(urls)}
</urlset>'''


def build_site(
    source_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    config: Optional[SiteConfig] = None,
) -> BuildResult:
    """
    Build static documentation site.
    
    Args:
        source_dir: Source directory with markdown files
        output_dir: Output directory for built site
        config: Site configuration
    
    Returns:
        BuildResult with build statistics
    """
    if config is None:
        config = SiteConfig()
    
    if source_dir:
        config.source_dir = source_dir
    if output_dir:
        config.output_dir = output_dir
    
    builder = SiteBuilder(config)
    builder.add_pages_from_directory(config.source_dir)
    
    return builder.build()


def serve_site(
    directory: str,
    port: int = 8000,
    host: str = "localhost",
) -> None:
    """
    Serve static site locally for preview.
    
    Args:
        directory: Directory containing built site
        port: Port to serve on
        host: Host to bind to
    """
    import http.server
    import socketserver
    import os
    
    os.chdir(directory)
    
    handler = http.server.SimpleHTTPRequestHandler
    
    with socketserver.TCPServer((host, port), handler) as httpd:
        print(f"Serving at http://{host}:{port}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server...")
