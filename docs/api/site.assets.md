# Module `site.assets`

Asset management for documentation site.

This module handles static assets including images, stylesheets,
JavaScript files, and other resources.

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `Asset`

Static asset file.

#### Attributes

- **path** (`<class 'str'>`): 
- **asset_type** (`<enum 'AssetType'>`): 
- **content** (`<class 'bytes'>`): 
- **size** (`<class 'int'>`): 
- **hash** (`<class 'str'>`): 
- **mime_type** (`<class 'str'>`): 
- **minified** (`<class 'bool'>`): 
- **compressed** (`<class 'bool'>`): 
- **fingerprinted** (`<class 'bool'>`): 
- **output_path** (`<class 'str'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, path: str, asset_type: assets.AssetType, content: bytes = b'', size: int = 0, hash: str = '', mime_type: str = '', minified: bool = False, compressed: bool = False, fingerprinted: bool = False, output_path: str = '') -> None
```

##### `fingerprint`

```python
def fingerprint(self)
```

Add content hash to filename for cache busting.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py:118](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py#L118)*

##### `to_data_uri`

```python
def to_data_uri(self) -> str
```

Convert to data URI for embedding.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py:127](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py#L127)*

### class `AssetManager`

Manages static assets for documentation site.

Handles loading, processing, and outputting assets.

#### Methods

##### `__init__`

```python
def __init__(self, minify_css: bool = True, minify_js: bool = True, optimize_images: bool = True, fingerprint: bool = True)
```

Initialize asset manager.

**Parameters:**

- **minify_css** (`<class 'bool'>`): Minify CSS files
- **minify_js** (`<class 'bool'>`): Minify JavaScript files
- **optimize_images** (`<class 'bool'>`): Optimize images
- **fingerprint** (`<class 'bool'>`): Add content hash to filenames

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py:260](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py#L260)*

##### `add_asset`

```python
def add_asset(self, path: str, content: bytes) -> assets.Asset
```

Add asset to manager.

**Parameters:**

- **path** (`<class 'str'>`): Asset path
- **content** (`<class 'bytes'>`): Asset content

**Returns**: `<class 'assets.Asset'>` - Asset object

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py:289](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py#L289)*

##### `add_from_directory`

```python
def add_from_directory(self, directory: Union[str, pathlib.Path], prefix: str = '')
```

Add all assets from directory.

**Parameters:**

- **directory** (`typing.Union[str, pathlib.Path]`): Directory path
- **prefix** (`<class 'str'>`): Path prefix for output

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py:329](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py#L329)*

##### `add_from_file`

```python
def add_from_file(self, file_path: Union[str, pathlib.Path], output_path: str = None) -> assets.Asset
```

Add asset from file.

**Parameters:**

- **file_path** (`typing.Union[str, pathlib.Path]`): Path to asset file
- **output_path** (`<class 'str'>`): Output path (defaults to filename) Default: `to filename)`.

**Returns**: `<class 'assets.Asset'>` - Asset object

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py:312](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py#L312)*

##### `get_inline`

```python
def get_inline(self, path: str) -> str
```

Get asset content for inlining.

**Parameters:**

- **path** (`<class 'str'>`): Asset path

**Returns**: `<class 'str'>` - Content string (for CSS/JS) or data URI (for images)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py:419](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py#L419)*

##### `get_url`

```python
def get_url(self, original_path: str) -> str
```

Get output URL for asset.

**Parameters:**

- **original_path** (`<class 'str'>`): Original asset path

**Returns**: `<class 'str'>` - Output path (possibly fingerprinted)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py:407](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py#L407)*

##### `process`

```python
def process(self)
```

Process all assets (minify, optimize, fingerprint).

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py:348](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py#L348)*

##### `summary`

```python
def summary(self) -> Dict[str, Any]
```

Get asset processing summary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py:438](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py#L438)*

##### `write`

```python
def write(self, output_dir: Union[str, pathlib.Path])
```

Write assets to output directory.

**Parameters:**

- **output_dir** (`typing.Union[str, pathlib.Path]`): Output directory

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py:388](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py#L388)*

### class `AssetType`(Enum)

Type of static asset.

### class `CSSMinifier`

Simple CSS minifier.

#### Methods

##### `minify`

```python
def minify(self, content: str) -> str
```

Minify CSS content.

**Parameters:**

- **content** (`<class 'str'>`): CSS content

**Returns**: `<class 'str'>` - Minified CSS

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py:136](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py#L136)*

### class `ImageOptimizer`

Image optimization utilities.

#### Methods

##### `__init__`

```python
def __init__(self)
```

Initialize optimizer.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py:206](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py#L206)*

##### `generate_srcset`

```python
def generate_srcset(self, content: bytes, widths: List[int] = None) -> Dict[int, bytes]
```

Generate responsive image srcset.

**Parameters:**

- **content** (`<class 'bytes'>`): Original image bytes
- **widths** (`typing.List[int]`): Target widths

**Returns**: `typing.Dict[int, bytes]` - Dictionary of width -> image bytes

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py:232](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py#L232)*

##### `optimize`

```python
def optimize(self, content: bytes, format: str) -> bytes
```

Optimize image content.

**Parameters:**

- **content** (`<class 'bytes'>`): Image bytes
- **format** (`<class 'str'>`): Image format extension

**Returns**: `<class 'bytes'>` - Optimized image bytes

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py:210](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py#L210)*

### class `JSMinifier`

Simple JavaScript minifier.

#### Methods

##### `minify`

```python
def minify(self, content: str) -> str
```

Minify JavaScript content.

**Parameters:**

- **content** (`<class 'str'>`): JavaScript content

**Returns**: `<class 'str'>` - Minified JavaScript

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py:167](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py#L167)*

## Functions

### `optimize_images`

```python
def optimize_images(source_dir: Union[str, pathlib.Path], output_dir: Union[str, pathlib.Path, NoneType] = None) -> int
```

Optimize all images in a directory.

**Parameters:**

- **source_dir** (`typing.Union[str, pathlib.Path]`): Source directory
- **output_dir** (`typing.Union[str, pathlib.Path, NoneType]`): Output directory (defaults to in-place) Default: `to in-place)`.

**Returns**: `<class 'int'>` - Number of images optimized

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py:488](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py#L488)*

### `process_assets`

```python
def process_assets(source_dir: Union[str, pathlib.Path], output_dir: Union[str, pathlib.Path], minify: bool = True, fingerprint: bool = True) -> assets.AssetManager
```

Process all assets in a directory.

**Parameters:**

- **source_dir** (`typing.Union[str, pathlib.Path]`): Source directory
- **output_dir** (`typing.Union[str, pathlib.Path]`): Output directory
- **minify** (`<class 'bool'>`): Enable minification
- **fingerprint** (`<class 'bool'>`): Enable fingerprinting

**Returns**: `<class 'assets.AssetManager'>` - AssetManager with processed assets

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py:456](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\site\assets.py#L456)*
