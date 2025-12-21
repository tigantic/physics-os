# Module `site.themes`

Documentation site themes.

This module provides theming support for HyperTensor documentation sites.

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `ColorScheme`(Enum)

Color scheme options.

### class `HyperTensorTheme`(Theme)

Custom theme for HyperTensor documentation.

Features aerospace-inspired design with emphasis on
technical documentation clarity.

#### Properties

##### `name`

```python
def name(self) -> str
```

Get theme name.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py:192](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py#L192)*

#### Methods

##### `__init__`

```python
def __init__(self)
```

Initialize HyperTensor theme.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py:342](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py#L342)*

##### `get_css`

```python
def get_css(self) -> str
```

Get HyperTensor theme CSS.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py:380](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py#L380)*

### class `Theme`

Documentation site theme.

Provides styling and layout for the documentation site.

#### Properties

##### `name`

```python
def name(self) -> str
```

Get theme name.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py:192](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py#L192)*

#### Methods

##### `__init__`

```python
def __init__(self, config: Optional[themes.ThemeConfig] = None)
```

Initialize theme.

**Parameters:**

- **config** (`typing.Optional[themes.ThemeConfig]`): Theme configuration

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py:183](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py#L183)*

##### `get_css`

```python
def get_css(self) -> str
```

Get complete CSS for theme.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py:197](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py#L197)*

##### `get_dark_mode_css`

```python
def get_dark_mode_css(self) -> str
```

Get dark mode CSS overrides.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py:201](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py#L201)*

### class `ThemeColors`

Theme color palette.

#### Attributes

- **primary** (`<class 'str'>`): 
- **secondary** (`<class 'str'>`): 
- **success** (`<class 'str'>`): 
- **warning** (`<class 'str'>`): 
- **danger** (`<class 'str'>`): 
- **info** (`<class 'str'>`): 
- **background** (`<class 'str'>`): 
- **surface** (`<class 'str'>`): 
- **text** (`<class 'str'>`): 
- **text_muted** (`<class 'str'>`): 
- **border** (`<class 'str'>`): 
- **code_background** (`<class 'str'>`): 
- **code_text** (`<class 'str'>`): 
- **nav_background** (`<class 'str'>`): 
- **nav_text** (`<class 'str'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, primary: str = '#0066cc', secondary: str = '#6c757d', success: str = '#28a745', warning: str = '#ffc107', danger: str = '#dc3545', info: str = '#17a2b8', background: str = '#ffffff', surface: str = '#f8f9fa', text: str = '#212529', text_muted: str = '#6c757d', border: str = '#dee2e6', code_background: str = '#1a1a2e', code_text: str = '#f8f8f2', nav_background: str = '#1a1a2e', nav_text: str = '#ffffff') -> None
```

##### `to_css_vars`

```python
def to_css_vars(self) -> str
```

Convert to CSS custom properties.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py:41](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py#L41)*

### class `ThemeConfig`

Complete theme configuration.

#### Attributes

- **name** (`<class 'str'>`): 
- **colors** (`<class 'themes.ThemeColors'>`): 
- **typography** (`<class 'themes.ThemeTypography'>`): 
- **layout** (`<class 'themes.ThemeLayout'>`): 
- **color_scheme** (`<enum 'ColorScheme'>`): 
- **custom_css** (`<class 'str'>`): 
- **custom_js** (`<class 'str'>`): 
- **enable_dark_mode** (`<class 'bool'>`): 
- **enable_search** (`<class 'bool'>`): 
- **enable_toc** (`<class 'bool'>`): 
- **enable_breadcrumbs** (`<class 'bool'>`): 
- **enable_edit_link** (`<class 'bool'>`): 
- **enable_prev_next** (`<class 'bool'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str = 'default', colors: themes.ThemeColors = <factory>, typography: themes.ThemeTypography = <factory>, layout: themes.ThemeLayout = <factory>, color_scheme: themes.ColorScheme = <ColorScheme.LIGHT: 1>, custom_css: str = '', custom_js: str = '', enable_dark_mode: bool = True, enable_search: bool = True, enable_toc: bool = True, enable_breadcrumbs: bool = True, enable_edit_link: bool = True, enable_prev_next: bool = True) -> None
```

##### `to_css`

```python
def to_css(self) -> str
```

Generate complete CSS for theme.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py:162](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py#L162)*

### class `ThemeLayout`

Layout configuration.

#### Attributes

- **max_width** (`<class 'str'>`): 
- **content_width** (`<class 'str'>`): 
- **sidebar_width** (`<class 'str'>`): 
- **toc_width** (`<class 'str'>`): 
- **nav_height** (`<class 'str'>`): 
- **footer_height** (`<class 'str'>`): 
- **spacing_unit** (`<class 'str'>`): 
- **border_radius** (`<class 'str'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, max_width: str = '1400px', content_width: str = '900px', sidebar_width: str = '280px', toc_width: str = '250px', nav_height: str = '64px', footer_height: str = 'auto', spacing_unit: str = '8px', border_radius: str = '8px') -> None
```

##### `to_css`

```python
def to_css(self) -> str
```

Convert to CSS.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py:113](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py#L113)*

### class `ThemeTypography`

Typography configuration.

#### Attributes

- **font_family** (`<class 'str'>`): 
- **font_family_mono** (`<class 'str'>`): 
- **font_size_base** (`<class 'str'>`): 
- **font_size_sm** (`<class 'str'>`): 
- **font_size_lg** (`<class 'str'>`): 
- **line_height** (`<class 'float'>`): 
- **line_height_heading** (`<class 'float'>`): 
- **heading_weight** (`<class 'int'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, font_family: str = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", font_family_mono: str = "'Fira Code', 'Consolas', 'Monaco', monospace", font_size_base: str = '16px', font_size_sm: str = '14px', font_size_lg: str = '18px', line_height: float = 1.6, line_height_heading: float = 1.3, heading_weight: int = 600) -> None
```

##### `to_css`

```python
def to_css(self) -> str
```

Convert to CSS.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py:79](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py#L79)*

## Functions

### `get_theme`

```python
def get_theme(name: str) -> themes.Theme
```

Get theme by name.

**Parameters:**

- **name** (`<class 'str'>`): Theme name

**Returns**: `<class 'themes.Theme'>` - Theme instance

**Raises:**

- `KeyError`: If theme not found

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py:499](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py#L499)*

### `list_themes`

```python
def list_themes() -> List[str]
```

List available theme names.

**Returns**: `typing.List[str]` - List of theme names

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py:517](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py#L517)*

### `register_theme`

```python
def register_theme(name: str, theme: themes.Theme)
```

Register a custom theme.

**Parameters:**

- **name** (`<class 'str'>`): Theme name
- **theme** (`<class 'themes.Theme'>`): Theme instance

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py:527](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\themes.py#L527)*
