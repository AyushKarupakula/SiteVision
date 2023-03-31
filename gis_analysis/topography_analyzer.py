import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, mapping
import richdem as rd

from config.settings import DATA_DIR
from utils.helpers import ensure_directory, create_timestamp_id

logger = logging.getLogger(__name__)

class TopographyAnalyzer:
    """Analyzes topographical features from elevation data."""
    
    def __init__(self):
        self.output_dir = ensure_directory(DATA_DIR / 'analysis' / 'topography')

    def analyze_topography(
        self,
        dem_path: Path,
        output_format: str = 'gpkg'
    ) -> Dict[str, Union[Path, Dict]]:
        """
        Perform comprehensive topographic analysis.
        
        Args:
            dem_path: Path to Digital Elevation Model file
            output_format: Output format ('gpkg' or 'geojson')
            
        Returns:
            Dictionary containing paths to analysis results and summary statistics
        """
        try:
            # Load DEM
            dem_array = self._load_dem(dem_path)
            
            # Calculate derived features
            slope = self._calculate_slope(dem_array)
            aspect = self._calculate_aspect(dem_array)
            twi = self._calculate_twi(dem_array)
            
            # Generate contours
            contours = self._generate_contours(dem_array)
            
            # Calculate statistics
            stats = self._calculate_statistics(dem_array, slope, aspect, twi)
            
            # Save results
            results = self._save_results(
                dem_array, slope, aspect, twi, contours,
                output_format=output_format
            )
            
            results['statistics'] = stats
            
            logger.info("Successfully completed topographic analysis")
            return results
            
        except Exception as e:
            logger.error(f"Error in topographic analysis: {e}")
            raise

    def _load_dem(self, dem_path: Path) -> rd.TerrainAttribute:
        """Load and prepare DEM for analysis."""
        try:
            with rasterio.open(dem_path) as src:
                dem_array = src.read(1)
                profile = src.profile
            
            # Convert to RichDEM format
            dem = rd.rdarray(dem_array, no_data=profile['nodata'])
            rd.FillDepressions(dem)
            
            return dem
            
        except Exception as e:
            logger.error(f"Error loading DEM: {e}")
            raise

    def _calculate_slope(self, dem: rd.TerrainAttribute) -> np.ndarray:
        """Calculate slope in degrees."""
        try:
            slope = rd.TerrainAttribute(dem, attrib='slope_degrees')
            return slope
            
        except Exception as e:
            logger.error(f"Error calculating slope: {e}")
            raise

    def _calculate_aspect(self, dem: rd.TerrainAttribute) -> np.ndarray:
        """Calculate aspect in degrees."""
        try:
            aspect = rd.TerrainAttribute(dem, attrib='aspect')
            return aspect
            
        except Exception as e:
            logger.error(f"Error calculating aspect: {e}")
            raise

    def _calculate_twi(self, dem: rd.TerrainAttribute) -> np.ndarray:
        """Calculate Topographic Wetness Index."""
        try:
            # Calculate flow accumulation
            flow_acc = rd.FlowAccumulation(dem, method='D8')
            
            # Calculate TWI
            slope = self._calculate_slope(dem)
            slope = np.where(slope == 0, 0.001, slope)  # Avoid division by zero
            twi = np.log(flow_acc / np.tan(np.deg2rad(slope)))
            
            return twi
            
        except Exception as e:
            logger.error(f"Error calculating TWI: {e}")
            raise

    def _generate_contours(
        self,
        dem: rd.TerrainAttribute,
        interval: float = 10.0
    ) -> gpd.GeoDataFrame:
        """Generate contour lines from DEM."""
        try:
            # Generate contours using matplotlib
            import matplotlib.pyplot as plt
            
            plt.ioff()  # Turn off interactive mode
            fig, ax = plt.subplots()
            contours = ax.contour(dem, levels=np.arange(
                dem.min(),
                dem.max(),
                interval
            ))
            plt.close(fig)
            
            # Convert to GeoDataFrame
            contour_shapes = []
            for level, paths in zip(contours.levels, contours.collections):
                for path in paths.get_paths():
                    coords = path.vertices
                    contour_shapes.append({
                        'geometry': shape({'type': 'LineString', 'coordinates': coords}),
                        'properties': {'elevation': float(level)}
                    })
            
            return gpd.GeoDataFrame.from_features(contour_shapes)
            
        except Exception as e:
            logger.error(f"Error generating contours: {e}")
            raise

    def _calculate_statistics(
        self,
        dem: rd.TerrainAttribute,
        slope: np.ndarray,
        aspect: np.ndarray,
        twi: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for topographic features."""
        try:
            stats = {
                'elevation': {
                    'min': float(np.min(dem)),
                    'max': float(np.max(dem)),
                    'mean': float(np.mean(dem)),
                    'std': float(np.std(dem))
                },
                'slope': {
                    'min': float(np.min(slope)),
                    'max': float(np.max(slope)),
                    'mean': float(np.mean(slope)),
                    'std': float(np.std(slope))
                },
                'aspect': {
                    'min': float(np.min(aspect)),
                    'max': float(np.max(aspect)),
                    'mean': float(np.mean(aspect)),
                    'std': float(np.std(aspect))
                },
                'twi': {
                    'min': float(np.min(twi)),
                    'max': float(np.max(twi)),
                    'mean': float(np.mean(twi)),
                    'std': float(np.std(twi))
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            raise

    def _save_results(
        self,
        dem: rd.TerrainAttribute,
        slope: np.ndarray,
        aspect: np.ndarray,
        twi: np.ndarray,
        contours: gpd.GeoDataFrame,
        output_format: str = 'gpkg'
    ) -> Dict[str, Path]:
        """Save analysis results to files."""
        try:
            timestamp = create_timestamp_id()
            results = {}
            
            # Save raster data
            for name, data in [
                ('dem', dem),
                ('slope', slope),
                ('aspect', aspect),
                ('twi', twi)
            ]:
                output_path = self.output_dir / f"{name}_{timestamp}.tif"
                with rasterio.open(output_path, 'w',
                    driver='GTiff',
                    height=data.shape[0],
                    width=data.shape[1],
                    count=1,
                    dtype=data.dtype,
                    crs='EPSG:4326'
                ) as dst:
                    dst.write(data, 1)
                results[name] = output_path
            
            # Save contours
            contours_path = self.output_dir / f"contours_{timestamp}.{output_format}"
            contours.to_file(contours_path, driver='GPKG' if output_format == 'gpkg' else 'GeoJSON')
            results['contours'] = contours_path
            
            return results
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    def identify_steep_slopes(
        self,
        slope: np.ndarray,
        threshold: float = 15.0
    ) -> gpd.GeoDataFrame:
        """
        Identify areas with slopes exceeding the threshold.
        
        Args:
            slope: Slope array in degrees
            threshold: Slope threshold in degrees
            
        Returns:
            GeoDataFrame of steep slope areas
        """
        try:
            # Create mask of steep slopes
            steep_mask = slope > threshold
            
            # Convert to polygons
            polygons = []
            for geom, value in shapes(steep_mask.astype(np.int8), mask=steep_mask):
                if value == 1:
                    polygons.append({
                        'geometry': shape(geom),
                        'properties': {'slope': float(value)}
                    })
            
            return gpd.GeoDataFrame.from_features(polygons)
            
        except Exception as e:
            logger.error(f"Error identifying steep slopes: {e}")
            raise

    def calculate_slope_aspects(
        self,
        aspect: np.ndarray
    ) -> Dict[str, float]:
        """Calculate distribution of slope aspects."""
        try:
            # Define aspect categories
            categories = {
                'N': (337.5, 22.5),
                'NE': (22.5, 67.5),
                'E': (67.5, 112.5),
                'SE': (112.5, 157.5),
                'S': (157.5, 202.5),
                'SW': (202.5, 247.5),
                'W': (247.5, 292.5),
                'NW': (292.5, 337.5)
            }
            
            distribution = {}
            total = len(aspect.flatten())
            
            for direction, (min_angle, max_angle) in categories.items():
                if min_angle < max_angle:
                    mask = (aspect >= min_angle) & (aspect < max_angle)
                else:  # Handle north case
                    mask = (aspect >= min_angle) | (aspect < max_angle)
                    
                distribution[direction] = float(np.sum(mask) / total * 100)
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error calculating slope aspects: {e}")
            raise 