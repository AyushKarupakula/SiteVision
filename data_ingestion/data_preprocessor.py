import os
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
import zipfile

import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, mapping

from config.settings import DATA_DIR
from utils.helpers import ensure_directory, create_timestamp_id

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles preprocessing of satellite imagery and GIS data."""
    
    def __init__(self):
        self.processed_dir = ensure_directory(DATA_DIR / 'processed')
        self.temp_dir = ensure_directory(DATA_DIR / 'temp')

    def preprocess_satellite_imagery(
        self,
        image_path: Path,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        target_crs: str = 'EPSG:4326'
    ) -> Path:
        """
        Preprocess satellite imagery: reproject, clip, and normalize.
        
        Args:
            image_path: Path to the input image
            bbox: Optional bounding box for clipping
            target_crs: Target coordinate reference system
            
        Returns:
            Path to preprocessed image
        """
        try:
            # Handle Sentinel-2 ZIP files
            if image_path.suffix == '.zip':
                image_path = self._extract_sentinel_data(image_path)
            
            with rasterio.open(image_path) as src:
                # Reproject if needed
                if src.crs != target_crs:
                    reprojected = self._reproject_raster(src, target_crs)
                else:
                    reprojected = src.read()
                    
                # Clip to bbox if provided
                if bbox:
                    geometry = [mapping(box(*bbox))]
                    reprojected, transform = mask(src, geometry, crop=True)
                
                # Normalize values
                normalized = self._normalize_raster(reprojected)
                
                # Save preprocessed image
                output_path = self.processed_dir / f"preprocessed_{create_timestamp_id()}.tif"
                
                profile = src.profile.copy()
                profile.update({
                    'crs': target_crs,
                    'dtype': 'float32',
                    'nodata': None
                })
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(normalized)
                
                logger.info(f"Successfully preprocessed satellite imagery to {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Error preprocessing satellite imagery: {e}")
            raise

    def preprocess_gis_data(
        self,
        data_paths: Dict[str, Path],
        bbox: Optional[Tuple[float, float, float, float]] = None,
        target_crs: str = 'EPSG:4326'
    ) -> Dict[str, Path]:
        """
        Preprocess GIS data: reproject, clip, and standardize attributes.
        
        Args:
            data_paths: Dictionary mapping data types to file paths
            bbox: Optional bounding box for clipping
            target_crs: Target coordinate reference system
            
        Returns:
            Dictionary mapping data types to preprocessed file paths
        """
        try:
            results = {}
            
            for data_type, path in data_paths.items():
                if path.suffix == '.gpkg':
                    # Process vector data
                    gdf = gpd.read_file(path)
                    
                    # Reproject if needed
                    if gdf.crs != target_crs:
                        gdf = gdf.to_crs(target_crs)
                    
                    # Clip to bbox if provided
                    if bbox:
                        clip_box = box(*bbox)
                        gdf = gdf[gdf.intersects(clip_box)]
                    
                    # Standardize column names
                    gdf = self._standardize_columns(gdf, data_type)
                    
                    # Save preprocessed data
                    output_path = self.processed_dir / f"preprocessed_{data_type}_{create_timestamp_id()}.gpkg"
                    gdf.to_file(output_path, driver='GPKG')
                    results[data_type] = output_path
                    
                elif path.suffix == '.tif':
                    # Process raster data (e.g., elevation)
                    results[data_type] = self.preprocess_satellite_imagery(path, bbox, target_crs)
            
            logger.info("Successfully preprocessed all GIS data")
            return results
            
        except Exception as e:
            logger.error(f"Error preprocessing GIS data: {e}")
            raise

    def _extract_sentinel_data(self, zip_path: Path) -> Path:
        """Extract Sentinel-2 data from ZIP archive."""
        try:
            temp_dir = self.temp_dir / f"sentinel_{create_timestamp_id()}"
            temp_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find and return path to main image file
            image_files = list(temp_dir.rglob('*.jp2'))
            if not image_files:
                raise ValueError("No image files found in Sentinel-2 archive")
                
            return image_files[0]
            
        except Exception as e:
            logger.error(f"Error extracting Sentinel data: {e}")
            raise

    def _reproject_raster(
        self,
        src: rasterio.DatasetReader,
        target_crs: str
    ) -> np.ndarray:
        """Reproject raster data to target CRS."""
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        reprojected = np.zeros((src.count, height, width), dtype=np.float32)
        
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=reprojected[i-1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )
            
        return reprojected

    def _normalize_raster(self, data: np.ndarray) -> np.ndarray:
        """Normalize raster data to [0, 1] range."""
        min_vals = np.nanmin(data, axis=(1, 2), keepdims=True)
        max_vals = np.nanmax(data, axis=(1, 2), keepdims=True)
        
        normalized = (data - min_vals) / (max_vals - min_vals)
        normalized = np.nan_to_num(normalized, 0)
        
        return normalized

    def _standardize_columns(
        self,
        gdf: gpd.GeoDataFrame,
        data_type: str
    ) -> gpd.GeoDataFrame:
        """Standardize column names and attributes based on data type."""
        if data_type == 'roads':
            # Standardize road attributes
            standard_columns = {
                'highway': 'road_type',
                'name': 'road_name',
                'lanes': 'num_lanes'
            }
        elif data_type == 'land_use':
            # Standardize land use attributes
            standard_columns = {
                'landuse': 'use_type',
                'name': 'area_name',
                'area': 'area_size'
            }
        else:
            return gdf
            
        # Rename columns if they exist
        for old_col, new_col in standard_columns.items():
            if old_col in gdf.columns:
                gdf = gdf.rename(columns={old_col: new_col})
                
        return gdf 