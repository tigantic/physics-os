"""
Tests for Layer 2: HyperVisual
==============================

Verifies tile-based rendering, slicing, streaming, and colormaps.
"""

import pytest
import torch
import numpy as np

from tensornet.substrate import Field
from tensornet.hypervisual import (
    TileRenderer,
    Tile,
    TileCoord,
    LODPyramid,
    RenderConfig,
    RenderStats,
    SliceEngine,
    SlicePlane,
    SliceResult,
    VolumeRenderer,
    StreamProtocol,
    StreamConfig,
    FrameBuffer,
    StreamStats,
    ColorMap,
    TransferFunction,
    VIRIDIS,
    PLASMA,
    INFERNO,
    MAGMA,
    TURBO,
    COOLWARM,
    JET,
    GRAYSCALE,
    apply_colormap,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def field_2d():
    """Create a 2D test field."""
    return Field.create(dims=2, bits_per_dim=6, rank=4, init='taylor_green')


@pytest.fixture
def field_3d():
    """Create a 3D test field."""
    return Field.create(dims=3, bits_per_dim=4, rank=4, init='random')


@pytest.fixture
def renderer(field_2d):
    """Create a tile renderer."""
    config = RenderConfig(tile_size=64, max_lod=4)
    return TileRenderer(field_2d, config)


@pytest.fixture
def slicer(field_2d):
    """Create a slice engine."""
    return SliceEngine(field_2d)


# =============================================================================
# TILE COORD
# =============================================================================

class TestTileCoord:
    """Test TileCoord data structure."""
    
    def test_creation(self):
        """Create tile coordinate."""
        coord = TileCoord(x=1, y=2, z=0, lod=3)
        assert coord.x == 1
        assert coord.y == 2
        assert coord.lod == 3
    
    def test_hash(self):
        """Tile coords are hashable."""
        coord1 = TileCoord(x=1, y=2, lod=3)
        coord2 = TileCoord(x=1, y=2, lod=3)
        
        assert hash(coord1) == hash(coord2)
        assert coord1 == coord2
    
    def test_key(self):
        """Tile coords have string keys."""
        coord = TileCoord(x=1, y=2, lod=3)
        assert coord.key == "3_1_2_0"


# =============================================================================
# LOD PYRAMID
# =============================================================================

class TestLODPyramid:
    """Test LOD pyramid."""
    
    def test_creation(self, field_2d):
        """Create LOD pyramid."""
        pyramid = LODPyramid(field_2d, tile_size=64, max_lod=4)
        
        assert pyramid.tile_size == 64
        assert pyramid.max_lod == 4
    
    def test_lod_resolutions(self, field_2d):
        """LOD resolutions halve at each level."""
        pyramid = LODPyramid(field_2d, tile_size=64, max_lod=4)
        
        for i in range(1, len(pyramid.lod_resolutions)):
            assert pyramid.lod_resolutions[i] <= pyramid.lod_resolutions[i-1]
    
    def test_tile_bounds(self, field_2d):
        """Get tile bounds."""
        pyramid = LODPyramid(field_2d, tile_size=64, max_lod=4)
        
        bounds = pyramid.get_tile_bounds(TileCoord(x=0, y=0, lod=0))
        x_min, y_min, x_max, y_max = bounds
        
        assert x_min >= 0
        assert y_min >= 0
        assert x_max <= 1
        assert y_max <= 1
    
    def test_visible_tiles(self, field_2d):
        """Get visible tiles for view."""
        pyramid = LODPyramid(field_2d, tile_size=64, max_lod=4)
        
        coords = pyramid.get_visible_tiles(
            view_bounds=(0, 0, 0.5, 0.5),
            lod=0,
        )
        
        assert len(coords) > 0
        for coord in coords:
            assert isinstance(coord, TileCoord)


# =============================================================================
# TILE RENDERER
# =============================================================================

class TestTileRenderer:
    """Test tile-based rendering."""
    
    def test_creation(self, field_2d):
        """Create renderer."""
        renderer = TileRenderer(field_2d)
        assert renderer.field is field_2d
    
    def test_get_tile(self, renderer):
        """Render a single tile."""
        coord = TileCoord(x=0, y=0, lod=0)
        tile = renderer.get_tile(coord)
        
        assert isinstance(tile, Tile)
        assert tile.data.shape[0] > 0
        assert tile.data.shape[1] > 0
    
    def test_tile_caching(self, renderer):
        """Tiles are cached."""
        coord = TileCoord(x=0, y=0, lod=0)
        
        # First request
        tile1 = renderer.get_tile(coord)
        
        # Second request should hit cache
        tile2 = renderer.get_tile(coord)
        
        assert renderer.cache.hits > 0
    
    def test_render_view(self, renderer):
        """Render complete view."""
        image = renderer.render_view(
            bounds=(0, 0, 1, 1),
            resolution=(128, 128),
        )
        
        assert image.shape == (128, 128)
    
    def test_render_zoomed_view(self, renderer):
        """Render zoomed view."""
        image = renderer.render_view(
            bounds=(0.25, 0.25, 0.75, 0.75),
            resolution=(256, 256),
        )
        
        assert image.shape == (256, 256)
    
    def test_invalidate_cache(self, renderer):
        """Cache invalidation."""
        coord = TileCoord(x=0, y=0, lod=0)
        
        tile1 = renderer.get_tile(coord)
        gen1 = tile1.generation
        
        renderer.invalidate()
        
        tile2 = renderer.get_tile(coord)
        gen2 = tile2.generation
        
        assert gen2 > gen1
    
    def test_summary(self, renderer):
        """Get rendering summary."""
        renderer.get_tile(TileCoord(x=0, y=0, lod=0))
        
        summary = renderer.summary()
        assert "TileRenderer" in summary
        assert "Tiles Rendered" in summary


# =============================================================================
# SLICE ENGINE
# =============================================================================

class TestSliceEngine:
    """Test slice extraction."""
    
    def test_creation(self, field_2d):
        """Create slice engine."""
        slicer = SliceEngine(field_2d)
        assert slicer.field is field_2d
    
    def test_xy_slice(self, slicer):
        """Extract XY slice."""
        result = slicer.slice(plane='xy', depth=0.5, resolution=64)
        
        assert isinstance(result, SliceResult)
        assert result.plane == SlicePlane.XY
        assert result.data.shape == (64, 64)
    
    def test_slice_bounds(self, slicer):
        """Extract slice with bounds."""
        result = slicer.slice(
            plane='xy',
            depth=0.5,
            resolution=64,
            bounds=(0.25, 0.25, 0.75, 0.75),
        )
        
        assert result.bounds == (0.25, 0.25, 0.75, 0.75)
    
    def test_slice_to_image(self, slicer):
        """Convert slice to RGBA image."""
        result = slicer.slice(plane='xy', depth=0.5, resolution=64)
        image = result.to_image()
        
        assert image.shape == (64, 64, 4)
        assert image.dtype == np.uint8
    
    def test_line_profile(self, slicer):
        """Extract line profile."""
        positions, values = slicer.line_profile(
            start=(0, 0.5),
            end=(1, 0.5),
            samples=100,
        )
        
        assert len(positions) == 100
        assert len(values) == 100
    
    def test_multi_slice(self, slicer):
        """Extract multiple slices."""
        depths = [0.25, 0.5, 0.75]
        results = slicer.multi_slice(depths=depths, resolution=32)
        
        assert len(results) == 3
        for r in results:
            assert isinstance(r, SliceResult)


# =============================================================================
# VOLUME RENDERER
# =============================================================================

class TestVolumeRenderer:
    """Test volume rendering."""
    
    def test_creation(self, field_2d):
        """Create volume renderer."""
        vol = VolumeRenderer(field_2d)
        assert vol.field is field_2d
    
    def test_render(self, field_2d):
        """Render volume."""
        vol = VolumeRenderer(field_2d)
        result = vol.render(
            resolution=(64, 64),
            samples_per_ray=16,
        )
        
        assert result.image.shape == (64, 64, 4)
        assert result.depth_buffer.shape == (64, 64)


# =============================================================================
# COLORMAPS
# =============================================================================

class TestColorMaps:
    """Test colormap functionality."""
    
    def test_predefined_colormaps(self):
        """Predefined colormaps exist."""
        assert VIRIDIS.name == 'viridis'
        assert PLASMA.name == 'plasma'
        assert INFERNO.name == 'inferno'
        assert MAGMA.name == 'magma'
        assert TURBO.name == 'turbo'
        assert COOLWARM.name == 'coolwarm'
        assert JET.name == 'jet'
        assert GRAYSCALE.name == 'grayscale'
    
    def test_colormap_call(self):
        """Colormap returns RGB tuple."""
        rgb = VIRIDIS(0.5)
        
        assert len(rgb) == 3
        assert all(0 <= c <= 1 for c in rgb)
    
    def test_colormap_range(self):
        """Colormap handles value range."""
        # In range
        rgb = VIRIDIS(0.5)
        assert all(0 <= c <= 1 for c in rgb)
        
        # Out of range
        rgb_low = VIRIDIS(-0.5)
        rgb_high = VIRIDIS(1.5)
    
    def test_colormap_reverse(self):
        """Reverse colormap."""
        viridis_r = VIRIDIS.reverse()
        
        assert viridis_r.name == 'viridis_r'
        assert len(viridis_r.colors) == len(VIRIDIS.colors)
    
    def test_colormap_rescale(self):
        """Rescale colormap."""
        scaled = VIRIDIS.rescale(0, 100)
        
        assert scaled.vmin == 0
        assert scaled.vmax == 100
    
    def test_apply_colormap(self):
        """Apply colormap to array."""
        data = np.random.rand(64, 64).astype(np.float32)
        rgba = apply_colormap(data, VIRIDIS)
        
        assert rgba.shape == (64, 64, 4)
        assert rgba.dtype == np.uint8


# =============================================================================
# TRANSFER FUNCTION
# =============================================================================

class TestTransferFunction:
    """Test transfer function for volume rendering."""
    
    def test_creation(self):
        """Create transfer function."""
        tf = TransferFunction()
        assert len(tf.points) > 0
    
    def test_add_point(self):
        """Add control point."""
        tf = TransferFunction()
        tf.add_point(0.5, (1, 0, 0), 0.5)
        
        assert len(tf.points) >= 3
    
    def test_call(self):
        """Evaluate transfer function."""
        tf = TransferFunction()
        rgba = tf(0.5)
        
        assert len(rgba) == 4
        assert all(0 <= c <= 1 for c in rgba)
    
    def test_apply(self):
        """Apply to array."""
        tf = TransferFunction()
        data = np.random.rand(10, 10).astype(np.float32)
        result = tf.apply(data)
        
        assert result.shape == (10, 10, 4)


# =============================================================================
# STREAM PROTOCOL
# =============================================================================

class TestStreamProtocol:
    """Test streaming protocol."""
    
    def test_creation(self, renderer):
        """Create stream protocol."""
        protocol = StreamProtocol(renderer)
        assert protocol.renderer is renderer
    
    def test_handle_subscribe(self, renderer):
        """Handle subscription."""
        protocol = StreamProtocol(renderer)
        
        response = protocol.handle_message({
            'type': 'subscribe',
            'channel': 'test',
        })
        
        assert response is not None
        assert 'Subscribed' in response.get('message', '')
    
    def test_handle_request_view(self, renderer):
        """Handle view request."""
        protocol = StreamProtocol(renderer)
        
        response = protocol.handle_message({
            'type': 'request_view',
            'bounds': [0, 0, 1, 1],
            'resolution': [128, 128],
        })
        
        assert response is not None
        assert response['type'] == 'view_update'
        assert 'data' in response
    
    def test_handle_request_tile(self, renderer):
        """Handle tile request."""
        protocol = StreamProtocol(renderer)
        
        response = protocol.handle_message({
            'type': 'request_tile',
            'x': 0,
            'y': 0,
            'lod': 0,
        })
        
        assert response is not None
        assert response['type'] == 'tile_data'
    
    def test_get_stats(self, renderer):
        """Get streaming stats."""
        protocol = StreamProtocol(renderer)
        
        # Generate some traffic
        protocol.handle_message({'type': 'request_tile', 'x': 0, 'y': 0, 'lod': 0})
        
        stats = protocol.get_stats()
        assert 'tiles_sent' in stats
        assert stats['tiles_sent'] > 0


# =============================================================================
# FRAME BUFFER
# =============================================================================

class TestFrameBuffer:
    """Test frame buffering."""
    
    def test_write_read(self):
        """Write and read frames (sync wrapper)."""
        import asyncio
        
        async def _test():
            buffer = FrameBuffer(size=3)
            await buffer.write({'frame': 1})
            await buffer.write({'frame': 2})
            frame1 = await buffer.read()
            frame2 = await buffer.read()
            assert frame1['frame'] == 1
            assert frame2['frame'] == 2
        
        asyncio.run(_test())
    
    def test_empty_read(self):
        """Read from empty buffer (sync wrapper)."""
        import asyncio
        
        async def _test():
            buffer = FrameBuffer(size=3)
            frame = await buffer.read()
            assert frame is None
        
        asyncio.run(_test())
    
    def test_pending_frames(self):
        """Count pending frames."""
        buffer = FrameBuffer(size=3)
        assert buffer.pending_frames == 0


# =============================================================================
# INTEGRATION
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_render_and_slice(self, field_2d):
        """Render tiles and slice same field."""
        renderer = TileRenderer(field_2d)
        slicer = SliceEngine(field_2d)
        
        tile = renderer.get_tile(TileCoord(x=0, y=0, lod=0))
        slice_result = slicer.slice(plane='xy', depth=0.5, resolution=64)
        
        assert tile.data.shape[0] > 0
        assert slice_result.data.shape[0] > 0
    
    def test_colormap_on_slice(self, field_2d):
        """Apply colormap to slice."""
        slicer = SliceEngine(field_2d)
        result = slicer.slice(plane='xy', depth=0.5, resolution=64)
        
        for cmap in [VIRIDIS, PLASMA, INFERNO]:
            image = result.to_image(cmap)
            assert image.shape == (64, 64, 4)
    
    def test_full_pipeline(self, field_2d):
        """Full rendering pipeline."""
        # Renderer
        renderer = TileRenderer(field_2d)
        image = renderer.render_view(resolution=(256, 256))
        
        # Apply colormap
        rgba = apply_colormap(image, VIRIDIS)
        
        # Protocol
        protocol = StreamProtocol(renderer)
        response = protocol.handle_message({
            'type': 'request_view',
            'bounds': [0, 0, 1, 1],
            'resolution': [128, 128],
        })
        
        assert response['type'] == 'view_update'


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
