# Module `site.search`

Search index generation and querying for documentation site.

This module provides full-text search capabilities for the
HyperTensor documentation site.

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `IndexedDocument`

Document in search index.

#### Attributes

- **id** (`<class 'str'>`): 
- **title** (`<class 'str'>`): 
- **content** (`<class 'str'>`): 
- **url** (`<class 'str'>`): 
- **doc_type** (`<class 'str'>`): 
- **metadata** (`typing.Dict[str, typing.Any]`): 
- **term_frequencies** (`typing.Dict[str, int]`): 
- **word_count** (`<class 'int'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, id: str, title: str, content: str, url: str, doc_type: str = '', metadata: Dict[str, Any] = <factory>, term_frequencies: Dict[str, int] = <factory>, word_count: int = 0) -> None
```

### class `Indexer`

Document indexer for building search indices.

Provides batch indexing and incremental updates.

#### Methods

##### `__init__`

```python
def __init__(self)
```

Initialize indexer.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py:425](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py#L425)*

##### `get_index`

```python
def get_index(self) -> search.SearchIndex
```

Get the search index.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py:502](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py#L502)*

##### `index_document`

```python
def index_document(self, doc_id: str, title: str, content: str, url: str = '', doc_type: str = '', metadata: Optional[Dict[str, Any]] = None) -> bool
```

Index a document, updating if changed.

**Parameters:**

- **doc_id** (`<class 'str'>`): Document ID
- **title** (`<class 'str'>`): Document title
- **content** (`<class 'str'>`): Document content
- **url** (`<class 'str'>`): Document URL
- **doc_type** (`<class 'str'>`): Document type
- **metadata** (`typing.Optional[typing.Dict[str, typing.Any]]`): Additional metadata

**Returns**: `<class 'bool'>` - True if document was indexed/updated

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py:430](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py#L430)*

##### `index_pages`

```python
def index_pages(self, pages: List[Any]) -> int
```

Index multiple pages.

**Parameters:**

- **pages** (`typing.List[typing.Any]`): List of Page objects

**Returns**: `<class 'int'>` - Number of pages indexed

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py:478](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py#L478)*

##### `load`

```python
def load(self, path: str)
```

Load index from file.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py:511](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py#L511)*

##### `save`

```python
def save(self, path: str)
```

Save index to file.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py:506](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py#L506)*

### class `SearchIndex`

Full-text search index using TF-IDF scoring.

Provides efficient document indexing and retrieval with
relevance-based ranking.

#### Methods

##### `__init__`

```python
def __init__(self, tokenizer: Optional[search.Tokenizer] = None)
```

Initialize search index.

**Parameters:**

- **tokenizer** (`typing.Optional[search.Tokenizer]`): Text tokenizer

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py:152](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py#L152)*

##### `add_document`

```python
def add_document(self, doc_id: str, title: str, content: str, url: str = '', doc_type: str = '', metadata: Optional[Dict[str, Any]] = None)
```

Add document to index.

**Parameters:**

- **doc_id** (`<class 'str'>`): Unique document ID
- **title** (`<class 'str'>`): Document title
- **content** (`<class 'str'>`): Document content
- **url** (`<class 'str'>`): Document URL
- **doc_type** (`<class 'str'>`): Document type (api, guide, etc.)
- **metadata** (`typing.Optional[typing.Dict[str, typing.Any]]`): Additional metadata

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py:171](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py#L171)*

##### `from_json`

```python
def from_json(json_str: str) -> 'SearchIndex'
```

Deserialize index from JSON.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py:400](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py#L400)*

##### `remove_document`

```python
def remove_document(self, doc_id: str)
```

Remove document from index.

**Parameters:**

- **doc_id** (`<class 'str'>`): Document ID to remove

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py:223](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py#L223)*

##### `search`

```python
def search(self, query: str, max_results: int = 10, doc_type_filter: Optional[str] = None) -> List[search.SearchResult]
```

Search for documents matching query.

**Parameters:**

- **query** (`<class 'str'>`): Search query
- **max_results** (`<class 'int'>`): Maximum results to return
- **doc_type_filter** (`typing.Optional[str]`): Filter by document type

**Returns**: `typing.List[search.SearchResult]` - List of search results ranked by relevance

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py:248](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py#L248)*

##### `to_json`

```python
def to_json(self) -> str
```

Serialize index to JSON.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py:382](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py#L382)*

### class `SearchResult`

Single search result.

#### Attributes

- **document_id** (`<class 'str'>`): 
- **title** (`<class 'str'>`): 
- **content_preview** (`<class 'str'>`): 
- **score** (`<class 'float'>`): 
- **highlights** (`typing.List[str]`): 
- **url** (`<class 'str'>`): 
- **doc_type** (`<class 'str'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, document_id: str, title: str, content_preview: str, score: float, highlights: List[str] = <factory>, url: str = '', doc_type: str = '') -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py:28](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py#L28)*

### class `Tokenizer`

Text tokenizer for search indexing.

#### Methods

##### `__init__`

```python
def __init__(self, min_length: int = 2, max_length: int = 50, lowercase: bool = True, remove_stop_words: bool = True, stem: bool = True)
```

Initialize tokenizer.

**Parameters:**

- **min_length** (`<class 'int'>`): Minimum token length
- **max_length** (`<class 'int'>`): Maximum token length
- **lowercase** (`<class 'bool'>`): Convert to lowercase
- **remove_stop_words** (`<class 'bool'>`): Remove common stop words
- **stem** (`<class 'bool'>`): Apply basic stemming

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py:68](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py#L68)*

##### `tokenize`

```python
def tokenize(self, text: str) -> List[str]
```

Tokenize text into terms.

**Parameters:**

- **text** (`<class 'str'>`): Input text

**Returns**: `typing.List[str]` - List of tokens

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py:92](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py#L92)*

## Functions

### `build_search_index`

```python
def build_search_index(pages: List[Any]) -> search.SearchIndex
```

Build search index from pages.

**Parameters:**

- **pages** (`typing.List[typing.Any]`): List of Page objects

**Returns**: `<class 'search.SearchIndex'>` - SearchIndex instance

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py:517](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py#L517)*

### `search`

```python
def search(index: search.SearchIndex, query: str, max_results: int = 10) -> List[search.SearchResult]
```

Search the index.

**Parameters:**

- **index** (`<class 'search.SearchIndex'>`): Search index
- **query** (`<class 'str'>`): Search query
- **max_results** (`<class 'int'>`): Maximum results

**Returns**: `typing.List[search.SearchResult]` - List of search results

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py:532](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\site\search.py#L532)*
