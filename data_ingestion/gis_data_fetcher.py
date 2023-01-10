import os
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import json

import requests
import geopandas as gpd
from shapely.geometry import box, Point
import osmnx as ox

from config.settings import GIS_DATA_DIR, OPENSTREETMAP_API_KEY
from utils.helpers import ensure_directory, create_timestamp_id

logger = logging.getLogger(__name__)

class GISDataFetcher:
    """Handles fetching GIS data from various sources."""
    
    def __init__(self):
        self.data_dir = ensure_directory(GIS_DATA_DIR)
        self.api_key = OPENSTREETMAP_API_KEY

    def fetch_osm_data(
        self,
        bbox: Tuple[float, float, float, float],
        tags: Dict[str, str]
    ) -> gpd.GeoDataFrame:
        """
        Fetch OpenStreetMap data for specified area and tags.
        
        Args:
            bbox: Tuple of (min_lon, min_lat, max_lon, max_lat)
            tags: Dictionary of OSM tags to fetch
            
        Returns:
            GeoDataFrame containing the fetched features
        """
        try:
            # Create network graph from OSM
            graph = ox.graph_from_bbox(
                bbox[3], bbox[1], bbox[2], bbox[0],
                network_type='all'
            )
            
            # Convert to GeoDataFrame
            nodes, edges = ox.graph_to_gdfs(graph)
            
            # Filter by tags
            filtered_data = edges[edges['highway'].isin(tags.get('highway', []))]
            
            # Save to file
            output_path = self.data_dir / f"osm_data_{create_timestamp_id()}.gpkg"
            filtered_data.to_file(output_path, driver='GPKG')
            
            logger.info(f"Successfully fetched OSM data to {output_path}")
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error fetching OSM data: {e}")
            raise

    def fetch_elevation_data(
        self,
        bbox: Tuple[float, float, float, float],
        resolution: float = 30.0
    ) -> Path:
        """
        Fetch elevation data from USGS.
        
        Args:
            bbox: Tuple of (min_lon, min_lat, max_lon, max_lat)
            resolution: Desired resolution in meters
            
        Returns:
            Path to downloaded elevation data
        """
        try:
            # Use USGS 3DEP service
            base_url = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEP/ImageServer/exportImage"
            
            params = {
                'bbox': f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                'format': 'tiff',
                'pixelType': 'F32',
                'noDataInterpretation': 'esriNoDataMatchAny',
                'interpolation': 'RSP_BilinearInterpolation',
                'f': 'json'
            }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            # Download the data
            output_path = self.data_dir / f"elevation_{create_timestamp_id()}.tif"
            with open(output_path, 'wb') as f:
                f.write(response.content)
                
            logger.info(f"Successfully downloaded elevation data to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error fetching elevation data: {e}")
            raise

    def fetch_land_use_data(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> gpd.GeoDataFrame:
        """
        Fetch land use data from local or national sources.
        
        Args:
            bbox: Tuple of (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            GeoDataFrame containing land use data
        """
        try:
            # Example using OSM land use data
            tags = {
                'landuse': [
                    'residential',
                    'commercial',
                    'industrial',
                    'retail',
                    'forest',
                    'farmland'
                ]
            }
            
            area = box(*bbox)
            land_use = ox.geometries_from_polygon(
                area,
                tags
            )
            
            output_path = self.data_dir / f"land_use_{create_timestamp_id()}.gpkg"
            land_use.to_file(output_path, driver='GPKG')
            
            logger.info(f"Successfully fetched land use data to {output_path}")
            return land_use
            
        except Exception as e:
            logger.error(f"Error fetching land use data: {e}")
            raise

    def fetch_all_gis_data(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> Dict[str, Path]:
        """
        Fetch all relevant GIS data for an area.
        
        Args:
            bbox: Tuple of (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            Dictionary mapping data types to file paths
        """
        results = {}
        
        try:
            # Fetch road network
            roads = self.fetch_osm_data(bbox, {'highway': ['primary', 'secondary', 'tertiary']})
            results['roads'] = self.data_dir / f"roads_{create_timestamp_id()}.gpkg"
            roads.to_file(results['roads'], driver='GPKG')
            
            # Fetch elevation data
            results['elevation'] = self.fetch_elevation_data(bbox)
            
            # Fetch land use
            land_use = self.fetch_land_use_data(bbox)
            results['land_use'] = self.data_dir / f"land_use_{create_timestamp_id()}.gpkg"
            land_use.to_file(results['land_use'], driver='GPKG')
            
            logger.info("Successfully fetched all GIS data")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching GIS data: {e}")
            raise 