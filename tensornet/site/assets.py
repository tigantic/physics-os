"""
Asset management for documentation site.

This module handles static assets including images, stylesheets,
JavaScript files, and other resources.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum, auto
from pathlib import Path
import hashlib
import shutil
import re
import json
import base64


class AssetType(Enum):
    """Type of static asset."""
    CSS = auto()
    JAVASCRIPT = auto()
    IMAGE = auto()
    FONT = auto()
    VIDEO = auto()
    DOCUMENT = auto()
    DATA = auto()
    OTHER = auto()
    
    @classmethod
    def from_extension(cls, ext: str) -> 'AssetType':
        """Determine asset type from file extension."""
        ext = ext.lower().lstrip('.')
        
        if ext in ('css', 'scss', 'sass', 'less'):
            return cls.CSS
        elif ext in ('js', 'mjs', 'ts'):
            return cls.JAVASCRIPT
        elif ext in ('png', 'jpg', 'jpeg', 'gif', 'svg', 'webp', 'ico', 'bmp'):
            return cls.IMAGE
        elif ext in ('woff', 'woff2', 'ttf', 'otf', 'eot'):
            return cls.FONT
        elif ext in ('mp4', 'webm', 'ogg', 'avi'):
            return cls.VIDEO
        elif ext in ('pdf', 'doc', 'docx', 'txt', 'md'):
            return cls.DOCUMENT
        elif ext in ('json', 'xml', 'yaml', 'yml', 'csv'):
            return cls.DATA
        else:
            return cls.OTHER


@dataclass
class Asset:
    """Static asset file."""
    path: str
    asset_type: AssetType
    content: bytes = b""
    
    # Metadata
    size: int = 0
    hash: str = ""
    mime_type: str = ""
    
    # Processing flags
    minified: bool = False
    compressed: bool = False
    fingerprinted: bool = False
    
    # Output path (may differ from input for fingerprinting)
    output_path: str = ""
    
    def __post_init__(self):
        """Initialize computed fields."""
        if not self.output_path:
            self.output_path = self.path
        if self.content:
            self.size = len(self.content)
            self.hash = self._compute_hash()
        if not self.mime_type:
            self.mime_type = self._get_mime_type()
    
    def _compute_hash(self) -> str:
        """Compute content hash."""
        return hashlib.md5(self.content).hexdigest()[:8]
    
    def _get_mime_type(self) -> str:
        """Get MIME type for asset."""
        mime_types = {
            AssetType.CSS: 'text/css',
            AssetType.JAVASCRIPT: 'application/javascript',
            AssetType.IMAGE: 'image/png',  # Default, refined below
            AssetType.FONT: 'font/woff2',
            AssetType.VIDEO: 'video/mp4',
            AssetType.DOCUMENT: 'application/octet-stream',
            AssetType.DATA: 'application/json',
            AssetType.OTHER: 'application/octet-stream',
        }
        
        mime = mime_types.get(self.asset_type, 'application/octet-stream')
        
        # Refine for images
        if self.asset_type == AssetType.IMAGE:
            ext = Path(self.path).suffix.lower()
            image_mimes = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.svg': 'image/svg+xml',
                '.webp': 'image/webp',
                '.ico': 'image/x-icon',
            }
            mime = image_mimes.get(ext, mime)
        
        return mime
    
    def fingerprint(self):
        """Add content hash to filename for cache busting."""
        if self.fingerprinted:
            return
        
        path = Path(self.output_path)
        self.output_path = f"{path.stem}.{self.hash}{path.suffix}"
        self.fingerprinted = True
    
    def to_data_uri(self) -> str:
        """Convert to data URI for embedding."""
        b64 = base64.b64encode(self.content).decode('ascii')
        return f"data:{self.mime_type};base64,{b64}"


class CSSMinifier:
    """Simple CSS minifier."""
    
    def minify(self, content: str) -> str:
        """
        Minify CSS content.
        
        Args:
            content: CSS content
        
        Returns:
            Minified CSS
        """
        # Remove comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Remove whitespace around special characters
        content = re.sub(r'\s*([{};:,>+~])\s*', r'\1', content)
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove leading/trailing whitespace
        content = content.strip()
        
        # Remove last semicolon before closing brace
        content = re.sub(r';(\s*})', r'\1', content)
        
        return content


class JSMinifier:
    """Simple JavaScript minifier."""
    
    def minify(self, content: str) -> str:
        """
        Minify JavaScript content.
        
        Note: This is a simple minifier. For production, use
        a full minifier like terser or uglify-js.
        
        Args:
            content: JavaScript content
        
        Returns:
            Minified JavaScript
        """
        # Remove single-line comments (but not URLs)
        content = re.sub(r'(?<!:)//.*$', '', content, flags=re.MULTILINE)
        
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Remove extra whitespace (careful with strings)
        lines = content.split('\n')
        minified_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                minified_lines.append(line)
        
        content = ' '.join(minified_lines)
        
        # Remove spaces around operators (simplified)
        content = re.sub(r'\s*([{};:,=+\-*/&|<>!?])\s*', r'\1', content)
        
        return content


class ImageOptimizer:
    """Image optimization utilities."""
    
    def __init__(self):
        """Initialize optimizer."""
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
    
    def optimize(self, content: bytes, format: str) -> bytes:
        """
        Optimize image content.
        
        Note: This is a placeholder. For real optimization,
        use libraries like Pillow or external tools.
        
        Args:
            content: Image bytes
            format: Image format extension
        
        Returns:
            Optimized image bytes
        """
        # In a real implementation, this would:
        # - Strip metadata (EXIF)
        # - Optimize compression
        # - Resize if too large
        # - Convert to WebP if beneficial
        
        return content
    
    def generate_srcset(
        self,
        content: bytes,
        widths: List[int] = None,
    ) -> Dict[int, bytes]:
        """
        Generate responsive image srcset.
        
        Args:
            content: Original image bytes
            widths: Target widths
        
        Returns:
            Dictionary of width -> image bytes
        """
        widths = widths or [320, 640, 1024, 1920]
        
        # Placeholder - would use Pillow to resize
        return {w: content for w in widths}


class AssetManager:
    """
    Manages static assets for documentation site.
    
    Handles loading, processing, and outputting assets.
    """
    
    def __init__(
        self,
        minify_css: bool = True,
        minify_js: bool = True,
        optimize_images: bool = True,
        fingerprint: bool = True,
    ):
        """
        Initialize asset manager.
        
        Args:
            minify_css: Minify CSS files
            minify_js: Minify JavaScript files
            optimize_images: Optimize images
            fingerprint: Add content hash to filenames
        """
        self.minify_css = minify_css
        self.minify_js = minify_js
        self.optimize_images = optimize_images
        self.fingerprint = fingerprint
        
        self.assets: Dict[str, Asset] = {}
        self._css_minifier = CSSMinifier()
        self._js_minifier = JSMinifier()
        self._image_optimizer = ImageOptimizer()
        
        # Asset manifest for mapping original -> fingerprinted paths
        self.manifest: Dict[str, str] = {}
    
    def add_asset(self, path: str, content: bytes) -> Asset:
        """
        Add asset to manager.
        
        Args:
            path: Asset path
            content: Asset content
        
        Returns:
            Asset object
        """
        ext = Path(path).suffix
        asset_type = AssetType.from_extension(ext)
        
        asset = Asset(
            path=path,
            asset_type=asset_type,
            content=content,
        )
        
        self.assets[path] = asset
        return asset
    
    def add_from_file(self, file_path: Union[str, Path], output_path: str = None) -> Asset:
        """
        Add asset from file.
        
        Args:
            file_path: Path to asset file
            output_path: Output path (defaults to filename)
        
        Returns:
            Asset object
        """
        file_path = Path(file_path)
        content = file_path.read_bytes()
        
        path = output_path or file_path.name
        return self.add_asset(path, content)
    
    def add_from_directory(self, directory: Union[str, Path], prefix: str = ""):
        """
        Add all assets from directory.
        
        Args:
            directory: Directory path
            prefix: Path prefix for output
        """
        directory = Path(directory)
        if not directory.exists():
            return
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                rel_path = file_path.relative_to(directory)
                output_path = f"{prefix}/{rel_path}" if prefix else str(rel_path)
                output_path = output_path.replace('\\', '/')
                self.add_from_file(file_path, output_path)
    
    def process(self):
        """Process all assets (minify, optimize, fingerprint)."""
        for path, asset in self.assets.items():
            self._process_asset(asset)
            
            # Update manifest
            self.manifest[path] = asset.output_path
    
    def _process_asset(self, asset: Asset):
        """Process single asset."""
        # Minify CSS
        if asset.asset_type == AssetType.CSS and self.minify_css:
            content_str = asset.content.decode('utf-8')
            minified = self._css_minifier.minify(content_str)
            asset.content = minified.encode('utf-8')
            asset.minified = True
            asset.size = len(asset.content)
            asset.hash = asset._compute_hash()
        
        # Minify JavaScript
        elif asset.asset_type == AssetType.JAVASCRIPT and self.minify_js:
            content_str = asset.content.decode('utf-8')
            minified = self._js_minifier.minify(content_str)
            asset.content = minified.encode('utf-8')
            asset.minified = True
            asset.size = len(asset.content)
            asset.hash = asset._compute_hash()
        
        # Optimize images
        elif asset.asset_type == AssetType.IMAGE and self.optimize_images:
            ext = Path(asset.path).suffix
            if ext in self._image_optimizer.supported_formats:
                asset.content = self._image_optimizer.optimize(asset.content, ext)
                asset.size = len(asset.content)
                asset.hash = asset._compute_hash()
        
        # Fingerprint
        if self.fingerprint:
            asset.fingerprint()
    
    def write(self, output_dir: Union[str, Path]):
        """
        Write assets to output directory.
        
        Args:
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for asset in self.assets.values():
            output_path = output_dir / asset.output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(asset.content)
        
        # Write manifest
        manifest_path = output_dir / 'asset-manifest.json'
        manifest_path.write_text(json.dumps(self.manifest, indent=2))
    
    def get_url(self, original_path: str) -> str:
        """
        Get output URL for asset.
        
        Args:
            original_path: Original asset path
        
        Returns:
            Output path (possibly fingerprinted)
        """
        return self.manifest.get(original_path, original_path)
    
    def get_inline(self, path: str) -> str:
        """
        Get asset content for inlining.
        
        Args:
            path: Asset path
        
        Returns:
            Content string (for CSS/JS) or data URI (for images)
        """
        asset = self.assets.get(path)
        if not asset:
            return ""
        
        if asset.asset_type == AssetType.IMAGE:
            return asset.to_data_uri()
        else:
            return asset.content.decode('utf-8')
    
    def summary(self) -> Dict[str, Any]:
        """Get asset processing summary."""
        by_type: Dict[str, int] = {}
        total_size = 0
        
        for asset in self.assets.values():
            type_name = asset.asset_type.name
            by_type[type_name] = by_type.get(type_name, 0) + 1
            total_size += asset.size
        
        return {
            'total_assets': len(self.assets),
            'by_type': by_type,
            'total_size_bytes': total_size,
            'total_size_kb': round(total_size / 1024, 2),
        }


def process_assets(
    source_dir: Union[str, Path],
    output_dir: Union[str, Path],
    minify: bool = True,
    fingerprint: bool = True,
) -> AssetManager:
    """
    Process all assets in a directory.
    
    Args:
        source_dir: Source directory
        output_dir: Output directory
        minify: Enable minification
        fingerprint: Enable fingerprinting
    
    Returns:
        AssetManager with processed assets
    """
    manager = AssetManager(
        minify_css=minify,
        minify_js=minify,
        optimize_images=True,
        fingerprint=fingerprint,
    )
    
    manager.add_from_directory(source_dir)
    manager.process()
    manager.write(output_dir)
    
    return manager


def optimize_images(
    source_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> int:
    """
    Optimize all images in a directory.
    
    Args:
        source_dir: Source directory
        output_dir: Output directory (defaults to in-place)
    
    Returns:
        Number of images optimized
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir) if output_dir else source_dir
    
    optimizer = ImageOptimizer()
    count = 0
    
    for ext in optimizer.supported_formats:
        for file_path in source_dir.rglob(f'*{ext}'):
            content = file_path.read_bytes()
            optimized = optimizer.optimize(content, ext)
            
            rel_path = file_path.relative_to(source_dir)
            output_path = output_dir / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(optimized)
            
            count += 1
    
    return count
