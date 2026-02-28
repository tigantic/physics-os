# Module `site.generator`

Static site generator for The Physics OS documentation.

This module provides the core site generation functionality including:
- Page rendering from Markdown/RST
- Navigation generation
- Template processing
- Multi-format output (HTML, PDF)

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `BuildResult`

Result of site build.

#### Attributes

- **success** (`<class 'bool'>`): 
- **pages_built** (`<class 'int'>`): 
- **assets_processed** (`<class 'int'>`): 
- **build_time** (`<class 'float'>`): 
- **output_dir** (`<class 'str'>`): 
- **errors** (`typing.List[str]`): 
- **warnings** (`typing.List[str]`): 

#### Methods

##### `__init__`

```python
def __init__(self, success: bool, pages_built: int, assets_processed: int, build_time: float, output_dir: str, errors: List[str] = <factory>, warnings: List[str] = <factory>) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py:207](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py#L207)*

### class `MarkdownRenderer`

Render Markdown to HTML.

#### Methods

##### `__init__`

```python
def __init__(self, config: generator.SiteConfig)
```

Initialize renderer.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py:223](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py#L223)*

##### `render`

```python
def render(self, content: str) -> str
```

Render markdown to HTML.

This is a simplified renderer. In production, use a full
markdown library like mistune or markdown-it.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py:227](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py#L227)*

### class `NavItem`

Navigation item in the site structure.

#### Attributes

- **title** (`<class 'str'>`): 
- **path** (`<class 'str'>`): 
- **children** (`typing.List[ForwardRef('NavItem')]`): 
- **icon** (`<class 'str'>`): 
- **external** (`<class 'bool'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, title: str, path: str, children: List[ForwardRef('NavItem')] = <factory>, icon: str = '', external: bool = False) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py:42](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py#L42)*

### class `Navigation`

Site navigation structure.

#### Attributes

- **items** (`typing.List[generator.NavItem]`): 
- **footer_items** (`typing.List[generator.NavItem]`): 

#### Methods

##### `__init__`

```python
def __init__(self, items: List[generator.NavItem] = <factory>, footer_items: List[generator.NavItem] = <factory>) -> None
```

##### `add_item`

```python
def add_item(self, item: generator.NavItem, parent_path: Optional[str] = None)
```

Add navigation item.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py:59](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py#L59)*

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py:81](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py#L81)*

### class `Page`

Documentation page.

#### Attributes

- **path** (`<class 'str'>`): 
- **title** (`<class 'str'>`): 
- **content** (`<class 'str'>`): 
- **page_type** (`<enum 'PageType'>`): 
- **metadata** (`typing.Dict[str, typing.Any]`): 
- **template** (`<class 'str'>`): 
- **toc** (`typing.List[typing.Dict[str, typing.Any]]`): 

#### Methods

##### `__init__`

```python
def __init__(self, path: str, title: str, content: str, page_type: generator.PageType = <PageType.CUSTOM: 7>, metadata: Dict[str, Any] = <factory>, template: str = 'default', toc: List[Dict[str, Any]] = <factory>) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py:128](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py#L128)*

### class `PageType`(Enum)

Type of documentation page.

### class `SiteBuilder`

Static site builder for The Physics OS documentation.

Generates a complete static documentation site from source files.

#### Methods

##### `__init__`

```python
def __init__(self, config: Optional[generator.SiteConfig] = None)
```

Initialize site builder.

**Parameters:**

- **config** (`typing.Optional[generator.SiteConfig]`): Site configuration

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py:469](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py#L469)*

##### `add_page`

```python
def add_page(self, page: generator.Page)
```

Add page to site.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py:483](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py#L483)*

##### `add_pages_from_directory`

```python
def add_pages_from_directory(self, directory: Union[str, pathlib.Path])
```

Add pages from markdown files in directory.

**Parameters:**

- **directory** (`typing.Union[str, pathlib.Path]`): Path to directory with markdown files

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py:487](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py#L487)*

##### `build`

```python
def build(self) -> generator.BuildResult
```

Build the static site.

**Returns**: `<class 'generator.BuildResult'>` - BuildResult with build statistics

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py:615](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py#L615)*

##### `build_navigation`

```python
def build_navigation(self)
```

Build navigation from pages.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py:536](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py#L536)*

### class `SiteConfig`

Configuration for static site generation.

#### Attributes

- **title** (`<class 'str'>`): 
- **description** (`<class 'str'>`): 
- **version** (`<class 'str'>`): 
- **base_url** (`<class 'str'>`): 
- **output_dir** (`<class 'str'>`): 
- **source_dir** (`<class 'str'>`): 
- **theme** (`<class 'str'>`): 
- **minify_html** (`<class 'bool'>`): 
- **minify_css** (`<class 'bool'>`): 
- **minify_js** (`<class 'bool'>`): 
- **generate_sitemap** (`<class 'bool'>`): 
- **generate_search_index** (`<class 'bool'>`): 
- **syntax_highlighting** (`<class 'bool'>`): 
- **math_rendering** (`<class 'bool'>`): 
- **mermaid_diagrams** (`<class 'bool'>`): 
- **author** (`<class 'str'>`): 
- **repository** (`<class 'str'>`): 
- **license** (`<class 'str'>`): 
- **nav_config** (`typing.Dict[str, typing.Any]`): 

#### Methods

##### `__init__`

```python
def __init__(self, title: str = 'The Physics OS Documentation', description: str = 'Quantum-inspired tensor network framework for hypersonic CFD', version: str = '2.5.0', base_url: str = '/', output_dir: str = '_site', source_dir: str = 'docs', theme: str = 'hypertensor', minify_html: bool = True, minify_css: bool = True, minify_js: bool = True, generate_sitemap: bool = True, generate_search_index: bool = True, syntax_highlighting: bool = True, math_rendering: bool = True, mermaid_diagrams: bool = True, author: str = 'Tigantic Holdings LLC', repository: str = 'https://github.com/tigantic/The Physics OS', license: str = 'MIT', nav_config: Dict[str, Any] = <factory>) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py:172](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py#L172)*

### class `TemplateEngine`

Simple template engine for HTML generation.

#### Methods

##### `__init__`

```python
def __init__(self)
```

Initialize template engine.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py:353](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py#L353)*

##### `add_template`

```python
def add_template(self, name: str, content: str)
```

Add custom template.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py:457](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py#L457)*

##### `render`

```python
def render(self, template_name: str, context: Dict[str, Any]) -> str
```

Render template with context.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py:428](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py#L428)*

## Functions

### `build_site`

```python
def build_site(source_dir: Optional[str] = None, output_dir: Optional[str] = None, config: Optional[generator.SiteConfig] = None) -> generator.BuildResult
```

Build static documentation site.

**Parameters:**

- **source_dir** (`typing.Optional[str]`): Source directory with markdown files
- **output_dir** (`typing.Optional[str]`): Output directory for built site
- **config** (`typing.Optional[generator.SiteConfig]`): Site configuration

**Returns**: `<class 'generator.BuildResult'>` - BuildResult with build statistics

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py:871](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py#L871)*

### `serve_site`

```python
def serve_site(directory: str, port: int = 8000, host: str = 'localhost') -> None
```

Serve static site locally for preview.

**Parameters:**

- **directory** (`<class 'str'>`): Directory containing built site
- **port** (`<class 'int'>`): Port to serve on
- **host** (`<class 'str'>`): Host to bind to

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py:901](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\generator.py#L901)*
