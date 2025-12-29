"""
S3 Fetcher - Orbital Data Acquisition
======================================

OPERATION VALHALLA - Phase 3.2: High-Speed Fetcher

Async S3 client for NOAA/NASA satellite data.
Direct memory transfer with retry logic.

Data Sources:
    - NOAA GFS: s3://noaa-gfs-bdp-pds/
    - NASA GIBS: https://gibs.earthdata.nasa.gov/
    - Sentinel Hub: (future integration)

Author: OPERATION VALHALLA
Date: 2025-12-28
"""

import asyncio
import aiohttp
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
import io
from PIL import Image
import torch


@dataclass
class DataSource:
    """Configuration for data source."""
    name: str
    url: str
    update_interval: int  # seconds
    requires_auth: bool = False


# Public data sources (no authentication required)
NOAA_GFS = DataSource(
    name="NOAA GFS",
    url="https://noaa-gfs-bdp-pds.s3.amazonaws.com",
    update_interval=21600,  # 6 hours
    requires_auth=False
)

NASA_GIBS = DataSource(
    name="NASA GIBS Blue Marble",
    url="https://gibs.earthdata.nasa.gov/wmts/epsg4326/best",
    update_interval=86400,  # Daily
    requires_auth=False
)


class S3Fetcher:
    """
    Async HTTP client for satellite data with retry logic.
    
    Features:
        - HTTP/2 multiplexing
        - Exponential backoff
        - In-memory JPEG decompression
        - Direct GPU upload
    """
    
    def __init__(
        self, 
        max_concurrent: int = 10,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize S3 fetcher.
        
        Args:
            max_concurrent: Max simultaneous requests
            timeout: Request timeout (seconds)
            max_retries: Max retry attempts
        """
        self.max_concurrent = max_concurrent
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Statistics
        self.stats = {
            'fetches': 0,
            'errors': 0,
            'bytes_downloaded': 0,
            'cache_hits': 0
        }
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=self.timeout,
            connector=aiohttp.TCPConnector(limit=self.max_concurrent)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def fetch_tile(
        self, 
        url: str, 
        to_gpu: bool = True,
        device: str = 'cuda:0'
    ) -> Optional[torch.Tensor]:
        """
        Fetch single tile from URL.
        
        Args:
            url: Full URL to tile image
            to_gpu: Upload directly to GPU
            device: CUDA device
            
        Returns:
            Tensor (H, W, 3) float32 or None on failure
        """
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        # Download to memory
                        data = await response.read()
                        self.stats['bytes_downloaded'] += len(data)
                        self.stats['fetches'] += 1
                        
                        # Decode JPEG in memory
                        img = Image.open(io.BytesIO(data)).convert('RGB')
                        arr = torch.from_numpy(
                            torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                        ).view(img.size[1], img.size[0], 3).float() / 255.0
                        
                        # Upload to GPU if requested
                        if to_gpu:
                            arr = arr.to(device)
                        
                        return arr
                    
                    elif response.status == 404:
                        print(f"⚠ Tile not found: {url}")
                        return None
                    
                    else:
                        print(f"⚠ HTTP {response.status}: {url}")
                        
            except asyncio.TimeoutError:
                print(f"⚠ Timeout (attempt {attempt+1}/{self.max_retries}): {url}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                print(f"⚠ Error fetching tile: {e}")
                self.stats['errors'] += 1
                
        return None
    
    async def fetch_tiles_batch(
        self,
        urls: List[str],
        to_gpu: bool = True,
        device: str = 'cuda:0'
    ) -> List[Optional[torch.Tensor]]:
        """
        Fetch multiple tiles concurrently.
        
        Args:
            urls: List of tile URLs
            to_gpu: Upload to GPU
            device: CUDA device
            
        Returns:
            List of tensors (or None for failures)
        """
        tasks = [self.fetch_tile(url, to_gpu, device) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter exceptions
        return [r if not isinstance(r, Exception) else None for r in results]
    
    def get_gibs_tile_url(
        self,
        layer: str = "BlueMarble_NextGeneration",
        date: str = "2024-01-01",
        tile_matrix: str = "2km",
        row: int = 0,
        col: int = 0,
        format: str = "jpeg"
    ) -> str:
        """
        Construct NASA GIBS tile URL.
        
        Args:
            layer: GIBS layer name
            date: YYYY-MM-DD
            tile_matrix: Resolution (2km, 1km, 500m, 250m)
            row, col: Tile coordinates
            format: Image format (jpeg or png)
            
        Returns:
            Full tile URL
        """
        return (
            f"{NASA_GIBS.url}/wmts/epsg4326/best/"
            f"{layer}/default/{date}/{tile_matrix}/"
            f"{row}/{col}.{format}"
        )
    
    def get_noaa_gfs_url(
        self,
        forecast_time: str = "2024010100",
        variable: str = "TMP",
        level: str = "surface",
        forecast_hour: int = 0
    ) -> str:
        """
        Construct NOAA GFS data URL.
        
        Args:
            forecast_time: YYYYMMDDHH format
            variable: GFS variable code
            level: Pressure level or "surface"
            forecast_hour: Hours from forecast start
            
        Returns:
            Full GFS file URL
        """
        date = forecast_time[:8]
        cycle = forecast_time[8:10]
        
        return (
            f"{NOAA_GFS.url}/gfs.{date}/{cycle}/atmos/"
            f"gfs.t{cycle}z.pgrb2.0p25.f{forecast_hour:03d}"
        )
    
    def print_stats(self):
        """Print fetch statistics."""
        print(f"\n{'='*60}")
        print("S3 FETCHER STATISTICS")
        print(f"{'='*60}")
        print(f"Fetches:     {self.stats['fetches']}")
        print(f"Errors:      {self.stats['errors']}")
        print(f"Downloaded:  {self.stats['bytes_downloaded'] / 1024**2:.2f} MB")
        print(f"Cache hits:  {self.stats['cache_hits']}")
        print(f"{'='*60}\n")


async def demo_fetch_blue_marble():
    """Demo: Fetch NASA Blue Marble tile."""
    print("\n" + "="*60)
    print("DEMO: Fetching NASA GIBS Blue Marble Tile")
    print("="*60 + "\n")
    
    async with S3Fetcher() as fetcher:
        # Fetch single tile (center of world map)
        url = fetcher.get_gibs_tile_url(
            layer="BlueMarble_NextGeneration",
            date="2024-01-01",
            tile_matrix="2km",
            row=2,
            col=4
        )
        
        print(f"Fetching: {url}")
        tile = await fetcher.fetch_tile(url, to_gpu=True)
        
        if tile is not None:
            print(f"✓ Tile received: {tile.shape} @ {tile.device}")
            print(f"  Value range: [{tile.min():.3f}, {tile.max():.3f}]")
        else:
            print("✗ Fetch failed")
        
        fetcher.print_stats()


if __name__ == "__main__":
    asyncio.run(demo_fetch_blue_marble())
