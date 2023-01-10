import os
from pathlib import Path
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import ee
from sentinelsat import SentinelAPI
from shapely.geometry import box

from config.settings import (
    SATELLITE_DATA_DIR,
    SENTINEL_HUB_KEY,
    GOOGLE_EARTH_ENGINE_KEY
)
from utils.helpers import ensure_directory, create_timestamp_id

logger = logging.getLogger(__name__)

class SatelliteDataFetcher:
    """Handles fetching satellite imagery from various sources."""
    
    def __init__(self):
        self.data_dir = ensure_directory(SATELLITE_DATA_DIR)
        self._init_apis()

    def _init_apis(self):
        """Initialize connections to satellite data APIs."""
        try:
            # Initialize Google Earth Engine
            credentials = ee.ServiceAccountCredentials(None, GOOGLE_EARTH_ENGINE_KEY)
            ee.Initialize(credentials)
            
            # Initialize Sentinel Hub
            self.sentinel_api = SentinelAPI(
                user=os.getenv('SENTINEL_USER'),
                password=os.getenv('SENTINEL_PASSWORD'),
                api_url='https://scihub.copernicus.eu/dhus'
            )
            
            logger.info("Successfully initialized satellite APIs")
        except Exception as e:
            logger.error(f"Error initializing satellite APIs: {e}")
            raise

    def fetch_landsat_imagery(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        cloud_cover_max: float = 20.0
    ) -> Path:
        """
        Fetch Landsat imagery for the specified area and time period.
        
        Args:
            bbox: Tuple of (min_lon, min_lat, max_lon, max_lat)
            start_date: Start date for imagery collection
            end_date: End date for imagery collection
            cloud_cover_max: Maximum acceptable cloud cover percentage
            
        Returns:
            Path to downloaded imagery
        """
        try:
            # Create Earth Engine geometry
            geometry = ee.Geometry.Rectangle(bbox)
            
            # Get Landsat 8 collection
            collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover_max)))
            
            # Select the least cloudy image
            image = collection.sort('CLOUD_COVER').first()
            
            if not image:
                raise ValueError("No suitable images found for the specified criteria")
            
            # Download the image
            output_path = self.data_dir / f"landsat_{create_timestamp_id()}.tif"
            url = image.getDownloadURL({
                'scale': 30,
                'region': geometry,
                'format': 'GEO_TIFF'
            })
            
            response = requests.get(url)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
                
            logger.info(f"Successfully downloaded Landsat imagery to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error fetching Landsat imagery: {e}")
            raise

    def fetch_sentinel_imagery(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        cloud_cover_max: float = 20.0
    ) -> Path:
        """
        Fetch Sentinel-2 imagery for the specified area and time period.
        
        Args:
            bbox: Tuple of (min_lon, min_lat, max_lon, max_lat)
            start_date: Start date for imagery collection
            end_date: End date for imagery collection
            cloud_cover_max: Maximum acceptable cloud cover percentage
            
        Returns:
            Path to downloaded imagery
        """
        try:
            # Create bounding box geometry
            footprint = box(*bbox)
            
            # Search for Sentinel-2 products
            products = self.sentinel_api.query(
                area=footprint,
                date=(start_date, end_date),
                platformname='Sentinel-2',
                cloudcoverpercentage=(0, cloud_cover_max)
            )
            
            if not products:
                raise ValueError("No suitable Sentinel-2 images found")
            
            # Get the product with least cloud cover
            product_df = self.sentinel_api.to_dataframe(products)
            best_product = product_df.sort_values('cloudcoverpercentage').iloc[0]
            
            # Download the product
            output_path = self.data_dir / f"sentinel_{create_timestamp_id()}.zip"
            self.sentinel_api.download(best_product.uuid, directory_path=self.data_dir)
            
            logger.info(f"Successfully downloaded Sentinel-2 imagery to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error fetching Sentinel-2 imagery: {e}")
            raise

    def get_latest_imagery(
        self,
        bbox: Tuple[float, float, float, float],
        source: str = 'both'
    ) -> Dict[str, Path]:
        """
        Get the latest available imagery for an area from specified source(s).
        
        Args:
            bbox: Tuple of (min_lon, min_lat, max_lon, max_lat)
            source: 'landsat', 'sentinel', or 'both'
            
        Returns:
            Dictionary mapping source names to image file paths
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Look back 30 days
        
        results = {}
        
        if source in ['landsat', 'both']:
            try:
                landsat_path = self.fetch_landsat_imagery(bbox, start_date, end_date)
                results['landsat'] = landsat_path
            except Exception as e:
                logger.warning(f"Failed to fetch Landsat imagery: {e}")
        
        if source in ['sentinel', 'both']:
            try:
                sentinel_path = self.fetch_sentinel_imagery(bbox, start_date, end_date)
                results['sentinel'] = sentinel_path
            except Exception as e:
                logger.warning(f"Failed to fetch Sentinel imagery: {e}")
        
        if not results:
            raise ValueError("Failed to fetch imagery from any source")
        
        return results 