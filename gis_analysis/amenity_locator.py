import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, box, mapping
from scipy.spatial import cKDTree
import osmnx as ox
from pyproj import CRS

from config.settings import DATA_DIR, ANALYSIS_SETTINGS
from utils.helpers import ensure_directory, create_timestamp_id

logger = logging.getLogger(__name__)

class AmenityLocator:
    """Analyzes proximity and accessibility to various amenities and services."""
    
    AMENITY_TYPES = {
        'education': {
            'tags': {'amenity': ['school', 'university', 'college', 'kindergarten']},
            'weight': 0.2,
            'max_distance': 2000  # meters
        },
        'healthcare': {
            'tags': {'amenity': ['hospital', 'clinic', 'doctors']},
            'weight': 0.15,
            'max_distance': 3000
        },
        'transportation': {
            'tags': {
                'highway': ['bus_stop', 'station'],
                'railway': ['station', 'subway_entrance']
            },
            'weight': 0.2,
            'max_distance': 1000
        },
        'shopping': {
            'tags': {
                'shop': ['supermarket', 'mall'],
                'amenity': ['marketplace']
            },
            'weight': 0.15,
            'max_distance': 1500
        },
        'recreation': {
            'tags': {
                'leisure': ['park', 'sports_centre'],
                'amenity': ['restaurant', 'cafe']
            },
            'weight': 0.15,
            'max_distance': 1000
        },
        'services': {
            'tags': {
                'amenity': ['bank', 'post_office', 'police', 'fire_station']
            },
            'weight': 0.15,
            'max_distance': 2000
        }
    }
    
    def __init__(self):
        self.output_dir = ensure_directory(DATA_DIR / 'analysis' / 'amenities')

    def analyze_amenities(
        self,
        area_geometry: Union[Polygon, box],
        custom_amenities: Optional[Dict[str, gpd.GeoDataFrame]] = None
    ) -> Dict[str, Union[Path, Dict]]:
        """
        Analyze proximity to amenities in the area.
        
        Args:
            area_geometry: Study area geometry
            custom_amenities: Optional dictionary of custom amenity GeoDataFrames
            
        Returns:
            Dictionary containing analysis results and maps
        """
        try:
            results = {}
            combined_score = None
            
            # Get amenities from OpenStreetMap
            osm_amenities = self._fetch_osm_amenities(area_geometry)
            
            # Combine with custom amenities if provided
            if custom_amenities:
                amenities = self._merge_amenities(osm_amenities, custom_amenities)
            else:
                amenities = osm_amenities
            
            # Analyze each amenity type
            for amenity_type, amenity_data in amenities.items():
                if amenity_type not in self.AMENITY_TYPES:
                    logger.warning(f"Unsupported amenity type: {amenity_type}")
                    continue
                
                # Calculate proximity scores
                proximity_scores = self._calculate_proximity_scores(
                    amenity_data,
                    area_geometry,
                    self.AMENITY_TYPES[amenity_type]['max_distance']
                )
                
                # Weight the scores
                weighted_scores = proximity_scores * self.AMENITY_TYPES[amenity_type]['weight']
                
                if combined_score is None:
                    combined_score = weighted_scores
                else:
                    combined_score += weighted_scores
                
                results[amenity_type] = proximity_scores
            
            # Generate accessibility map
            accessibility_map = self._generate_accessibility_map(
                combined_score,
                area_geometry
            )
            results['accessibility'] = accessibility_map
            
            # Save results
            output_paths = self._save_results(results)
            
            logger.info("Successfully completed amenity analysis")
            return output_paths
            
        except Exception as e:
            logger.error(f"Error in amenity analysis: {e}")
            raise

    def _fetch_osm_amenities(
        self,
        area_geometry: Union[Polygon, box]
    ) -> Dict[str, gpd.GeoDataFrame]:
        """Fetch amenities from OpenStreetMap."""
        try:
            amenities = {}
            
            # Convert area geometry to bounds
            bounds = area_geometry.bounds
            
            for amenity_type, config in self.AMENITY_TYPES.items():
                amenity_gdfs = []
                
                # Fetch amenities for each tag type
                for tag_key, tag_values in config['tags'].items():
                    tags = {tag_key: tag_values}
                    
                    gdf = ox.geometries_from_bbox(
                        bounds[3], bounds[1], bounds[2], bounds[0],
                        tags
                    )
                    
                    if not gdf.empty:
                        gdf['amenity_type'] = amenity_type
                        amenity_gdfs.append(gdf)
                
                if amenity_gdfs:
                    amenities[amenity_type] = pd.concat(amenity_gdfs)
            
            return amenities
            
        except Exception as e:
            logger.error(f"Error fetching OSM amenities: {e}")
            raise

    def _merge_amenities(
        self,
        osm_amenities: Dict[str, gpd.GeoDataFrame],
        custom_amenities: Dict[str, gpd.GeoDataFrame]
    ) -> Dict[str, gpd.GeoDataFrame]:
        """Merge OSM and custom amenities."""
        try:
            merged = osm_amenities.copy()
            
            for amenity_type, custom_gdf in custom_amenities.items():
                if amenity_type in merged:
                    merged[amenity_type] = pd.concat([
                        merged[amenity_type],
                        custom_gdf
                    ])
                else:
                    merged[amenity_type] = custom_gdf
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging amenities: {e}")
            raise

    def _calculate_proximity_scores(
        self,
        amenities: gpd.GeoDataFrame,
        area_geometry: Union[Polygon, box],
        max_distance: float
    ) -> np.ndarray:
        """Calculate proximity scores using KD-tree."""
        try:
            # Create grid points
            grid_size = 50  # meters
            minx, miny, maxx, maxy = area_geometry.bounds
            x = np.arange(minx, maxx, grid_size)
            y = np.arange(miny, maxy, grid_size)
            xx, yy = np.meshgrid(x, y)
            grid_points = np.column_stack((xx.ravel(), yy.ravel()))
            
            # Extract amenity coordinates
            amenity_coords = np.array([
                [geom.x, geom.y] for geom in amenities.geometry
            ])
            
            # Build KD-tree
            tree = cKDTree(amenity_coords)
            
            # Calculate distances
            distances, _ = tree.query(grid_points, k=1)
            
            # Convert distances to scores (1 at amenity, 0 at max_distance)
            scores = 1 - np.clip(distances / max_distance, 0, 1)
            
            # Reshape to grid
            return scores.reshape(len(y), len(x))
            
        except Exception as e:
            logger.error(f"Error calculating proximity scores: {e}")
            raise

    def _generate_accessibility_map(
        self,
        scores: np.ndarray,
        area_geometry: Union[Polygon, box]
    ) -> gpd.GeoDataFrame:
        """Generate accessibility map from proximity scores."""
        try:
            # Define accessibility levels
            levels = {
                'very_high': 0.8,
                'high': 0.6,
                'medium': 0.4,
                'low': 0.2,
                'very_low': 0.0
            }
            
            accessibility_zones = []
            for level, threshold in levels.items():
                mask = scores > threshold
                if np.any(mask):
                    # Convert masked array to polygons
                    shapes = self._array_to_polygons(mask, area_geometry)
                    
                    for shape_geom in shapes:
                        accessibility_zones.append({
                            'geometry': shape_geom,
                            'properties': {
                                'accessibility': level,
                                'score': float(np.mean(scores[mask]))
                            }
                        })
            
            return gpd.GeoDataFrame.from_features(accessibility_zones)
            
        except Exception as e:
            logger.error(f"Error generating accessibility map: {e}")
            raise

    def _array_to_polygons(
        self,
        mask: np.ndarray,
        area_geometry: Union[Polygon, box]
    ) -> List[Polygon]:
        """Convert boolean mask to list of polygons."""
        try:
            from rasterio import features
            
            # Get bounds and create transform
            bounds = area_geometry.bounds
            transform = self._create_transform(mask.shape, bounds)
            
            # Generate shapes from mask
            shapes = list(features.shapes(
                mask.astype(np.int8),
                mask=mask,
                transform=transform
            ))
            
            return [shape(geom) for geom, val in shapes if val == 1]
            
        except Exception as e:
            logger.error(f"Error converting array to polygons: {e}")
            raise

    def _create_transform(
        self,
        shape: Tuple[int, int],
        bounds: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float, float, float]:
        """Create affine transform for raster operations."""
        try:
            width = shape[1]
            height = shape[0]
            
            x_res = (bounds[2] - bounds[0]) / width
            y_res = (bounds[3] - bounds[1]) / height
            
            return (
                x_res, 0.0, bounds[0],
                0.0, -y_res, bounds[3]
            )
            
        except Exception as e:
            logger.error(f"Error creating transform: {e}")
            raise

    def _save_results(
        self,
        results: Dict[str, Union[np.ndarray, gpd.GeoDataFrame]]
    ) -> Dict[str, Path]:
        """Save analysis results."""
        try:
            timestamp = create_timestamp_id()
            output_paths = {}
            
            for name, data in results.items():
                if isinstance(data, np.ndarray):
                    # Save raster data
                    output_path = self.output_dir / f"{name}_scores_{timestamp}.tif"
                    with rasterio.open(output_path, 'w',
                        driver='GTiff',
                        height=data.shape[0],
                        width=data.shape[1],
                        count=1,
                        dtype=data.dtype,
                        crs='EPSG:4326'
                    ) as dst:
                        dst.write(data, 1)
                else:
                    # Save vector data
                    output_path = self.output_dir / f"{name}_{timestamp}.gpkg"
                    data.to_file(output_path, driver='GPKG')
                
                output_paths[name] = output_path
            
            return output_paths
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise 