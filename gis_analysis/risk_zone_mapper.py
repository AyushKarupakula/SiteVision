import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, mapping, Point, Polygon, box
from shapely.ops import unary_union
import rasterio
from rasterio.features import shapes
from scipy.spatial import cKDTree

from config.settings import DATA_DIR, ANALYSIS_SETTINGS
from utils.helpers import ensure_directory, create_timestamp_id

logger = logging.getLogger(__name__)

class RiskZoneMapper:
    """Maps and analyzes environmental and geological risk zones."""
    
    RISK_TYPES = {
        'flood': {'weight': 0.3, 'buffer': 100},
        'landslide': {'weight': 0.25, 'buffer': 50},
        'earthquake': {'weight': 0.25, 'threshold': 6.0},
        'fire': {'weight': 0.2, 'buffer': 200}
    }
    
    def __init__(self):
        self.output_dir = ensure_directory(DATA_DIR / 'analysis' / 'risk_zones')
        
    def analyze_risks(
        self,
        area_geometry: Union[Polygon, box],
        data_sources: Dict[str, Path]
    ) -> Dict[str, Union[Path, Dict]]:
        """
        Perform comprehensive risk analysis for an area.
        
        Args:
            area_geometry: Study area geometry
            data_sources: Dictionary mapping risk types to data file paths
            
        Returns:
            Dictionary containing risk analysis results and maps
        """
        try:
            results = {}
            combined_risk = None
            
            # Analyze each risk type
            for risk_type, path in data_sources.items():
                if risk_type not in self.RISK_TYPES:
                    logger.warning(f"Unsupported risk type: {risk_type}")
                    continue
                
                risk_zones = self._analyze_risk_type(
                    risk_type, path, area_geometry
                )
                
                # Combine risks with weights
                risk_array = self._rasterize_risk_zones(risk_zones, area_geometry)
                weighted_risk = risk_array * self.RISK_TYPES[risk_type]['weight']
                
                if combined_risk is None:
                    combined_risk = weighted_risk
                else:
                    combined_risk += weighted_risk
                
                results[risk_type] = risk_zones
            
            # Generate final risk map
            risk_map = self._generate_risk_map(combined_risk, area_geometry)
            results['combined_risk'] = risk_map
            
            # Save results
            output_paths = self._save_results(results)
            
            logger.info("Successfully completed risk analysis")
            return output_paths
            
        except Exception as e:
            logger.error(f"Error in risk analysis: {e}")
            raise

    def _analyze_risk_type(
        self,
        risk_type: str,
        data_path: Path,
        area_geometry: Union[Polygon, box]
    ) -> gpd.GeoDataFrame:
        """Analyze specific type of risk."""
        try:
            if risk_type == 'flood':
                return self._analyze_flood_risk(data_path, area_geometry)
            elif risk_type == 'landslide':
                return self._analyze_landslide_risk(data_path, area_geometry)
            elif risk_type == 'earthquake':
                return self._analyze_earthquake_risk(data_path, area_geometry)
            elif risk_type == 'fire':
                return self._analyze_fire_risk(data_path, area_geometry)
            else:
                raise ValueError(f"Unsupported risk type: {risk_type}")
                
        except Exception as e:
            logger.error(f"Error analyzing {risk_type} risk: {e}")
            raise

    def _analyze_flood_risk(
        self,
        flood_data_path: Path,
        area_geometry: Union[Polygon, box]
    ) -> gpd.GeoDataFrame:
        """Analyze flood risk zones."""
        try:
            # Load flood data
            flood_data = gpd.read_file(flood_data_path)
            
            # Clip to area of interest
            flood_zones = gpd.clip(flood_data, area_geometry)
            
            # Buffer flood zones
            flood_zones['geometry'] = flood_zones.geometry.buffer(
                self.RISK_TYPES['flood']['buffer']
            )
            
            # Calculate risk levels based on flood depth or frequency
            if 'depth' in flood_zones.columns:
                flood_zones['risk_level'] = pd.qcut(
                    flood_zones['depth'],
                    q=5,
                    labels=['very_low', 'low', 'medium', 'high', 'very_high']
                )
            
            return flood_zones
            
        except Exception as e:
            logger.error(f"Error analyzing flood risk: {e}")
            raise

    def _analyze_landslide_risk(
        self,
        slope_data_path: Path,
        area_geometry: Union[Polygon, box]
    ) -> gpd.GeoDataFrame:
        """Analyze landslide risk zones."""
        try:
            with rasterio.open(slope_data_path) as src:
                slope_data = src.read(1)
                transform = src.transform
            
            # Define risk thresholds
            risk_levels = {
                'very_high': 35,
                'high': 25,
                'medium': 15,
                'low': 10,
                'very_low': 5
            }
            
            # Create risk zones
            risk_zones = []
            for level, threshold in risk_levels.items():
                mask = slope_data > threshold
                for geom, value in shapes(mask.astype(np.int8), mask=mask):
                    if value == 1:
                        risk_zones.append({
                            'geometry': shape(geom),
                            'properties': {'risk_level': level}
                        })
            
            return gpd.GeoDataFrame.from_features(risk_zones)
            
        except Exception as e:
            logger.error(f"Error analyzing landslide risk: {e}")
            raise

    def _analyze_earthquake_risk(
        self,
        fault_data_path: Path,
        area_geometry: Union[Polygon, box]
    ) -> gpd.GeoDataFrame:
        """Analyze earthquake risk zones."""
        try:
            # Load fault line data
            fault_data = gpd.read_file(fault_data_path)
            
            # Create buffer zones around fault lines
            buffer_distances = {
                'very_high': 1000,
                'high': 2000,
                'medium': 5000,
                'low': 10000,
                'very_low': 20000
            }
            
            risk_zones = []
            for level, distance in buffer_distances.items():
                buffer = fault_data.geometry.buffer(distance)
                risk_zones.append({
                    'geometry': unary_union(buffer),
                    'properties': {'risk_level': level}
                })
            
            return gpd.GeoDataFrame.from_features(risk_zones)
            
        except Exception as e:
            logger.error(f"Error analyzing earthquake risk: {e}")
            raise

    def _analyze_fire_risk(
        self,
        vegetation_data_path: Path,
        area_geometry: Union[Polygon, box]
    ) -> gpd.GeoDataFrame:
        """Analyze fire risk zones."""
        try:
            # Load vegetation data
            vegetation_data = gpd.read_file(vegetation_data_path)
            
            # Define risk levels for vegetation types
            risk_levels = {
                'forest': 'very_high',
                'shrubland': 'high',
                'grassland': 'medium',
                'wetland': 'low',
                'barren': 'very_low'
            }
            
            # Assign risk levels
            vegetation_data['risk_level'] = vegetation_data['veg_type'].map(risk_levels)
            
            # Buffer high-risk areas
            vegetation_data['geometry'] = vegetation_data.apply(
                lambda x: x.geometry.buffer(
                    self.RISK_TYPES['fire']['buffer']
                    if x.risk_level in ['very_high', 'high']
                    else 0
                ),
                axis=1
            )
            
            return vegetation_data
            
        except Exception as e:
            logger.error(f"Error analyzing fire risk: {e}")
            raise

    def _rasterize_risk_zones(
        self,
        risk_zones: gpd.GeoDataFrame,
        area_geometry: Union[Polygon, box],
        cell_size: float = 30.0
    ) -> np.ndarray:
        """Convert risk zones to raster format."""
        try:
            # Create raster template
            bounds = area_geometry.bounds
            width = int((bounds[2] - bounds[0]) / cell_size)
            height = int((bounds[3] - bounds[1]) / cell_size)
            
            risk_levels = {
                'very_low': 0.2,
                'low': 0.4,
                'medium': 0.6,
                'high': 0.8,
                'very_high': 1.0
            }
            
            # Rasterize each risk level
            raster = np.zeros((height, width))
            for level, value in risk_levels.items():
                mask = risk_zones['risk_level'] == level
                if mask.any():
                    shapes_to_rasterize = [
                        (geom, value) for geom in risk_zones[mask].geometry
                    ]
                    level_raster = features.rasterize(
                        shapes_to_rasterize,
                        out_shape=(height, width),
                        transform=transform.from_bounds(*bounds, width, height)
                    )
                    raster = np.maximum(raster, level_raster)
            
            return raster
            
        except Exception as e:
            logger.error(f"Error rasterizing risk zones: {e}")
            raise

    def _generate_risk_map(
        self,
        risk_array: np.ndarray,
        area_geometry: Union[Polygon, box]
    ) -> gpd.GeoDataFrame:
        """Generate final risk map from combined risk array."""
        try:
            # Define risk level thresholds
            thresholds = {
                'very_high': 0.8,
                'high': 0.6,
                'medium': 0.4,
                'low': 0.2,
                'very_low': 0.0
            }
            
            risk_zones = []
            for level, threshold in thresholds.items():
                mask = risk_array > threshold
                for geom, value in shapes(mask.astype(np.int8), mask=mask):
                    if value == 1:
                        risk_zones.append({
                            'geometry': shape(geom),
                            'properties': {
                                'risk_level': level,
                                'risk_value': float(np.mean(risk_array[mask]))
                            }
                        })
            
            return gpd.GeoDataFrame.from_features(risk_zones)
            
        except Exception as e:
            logger.error(f"Error generating risk map: {e}")
            raise

    def _save_results(
        self,
        results: Dict[str, gpd.GeoDataFrame]
    ) -> Dict[str, Path]:
        """Save risk analysis results."""
        try:
            timestamp = create_timestamp_id()
            output_paths = {}
            
            for risk_type, gdf in results.items():
                output_path = self.output_dir / f"{risk_type}_{timestamp}.gpkg"
                gdf.to_file(output_path, driver='GPKG')
                output_paths[risk_type] = output_path
            
            return output_paths
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise 