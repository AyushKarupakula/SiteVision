import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import pandas as pd
from scipy.stats import gmean
from dataclasses import dataclass

from config.settings import DATA_DIR
from utils.helpers import ensure_directory, create_timestamp_id

logger = logging.getLogger(__name__)

@dataclass
class CriteriaWeight:
    """Data class for storing criteria weights and metadata."""
    name: str
    weight: float
    description: str
    subcriteria: Optional[Dict[str, float]] = None

class CriteriaWeighting:
    """Handles the weighting of different criteria for site suitability analysis."""
    
    # Default criteria hierarchy and weights
    DEFAULT_CRITERIA = {
        'environmental': {
            'weight': 0.25,
            'description': 'Environmental factors and constraints',
            'subcriteria': {
                'flood_risk': 0.3,
                'landslide_risk': 0.25,
                'soil_quality': 0.25,
                'ecosystem_sensitivity': 0.2
            }
        },
        'infrastructure': {
            'weight': 0.2,
            'description': 'Infrastructure availability and quality',
            'subcriteria': {
                'road_access': 0.3,
                'utilities': 0.3,
                'public_transport': 0.2,
                'telecommunications': 0.2
            }
        },
        'social': {
            'weight': 0.2,
            'description': 'Social and community factors',
            'subcriteria': {
                'amenities_proximity': 0.3,
                'community_services': 0.25,
                'education_access': 0.25,
                'healthcare_access': 0.2
            }
        },
        'economic': {
            'weight': 0.2,
            'description': 'Economic and market factors',
            'subcriteria': {
                'market_demand': 0.3,
                'development_cost': 0.3,
                'property_value': 0.2,
                'economic_growth': 0.2
            }
        },
        'regulatory': {
            'weight': 0.15,
            'description': 'Regulatory and legal constraints',
            'subcriteria': {
                'zoning_compliance': 0.4,
                'building_regulations': 0.3,
                'environmental_regulations': 0.3
            }
        }
    }
    
    def __init__(self):
        self.output_dir = ensure_directory(DATA_DIR / 'evaluation' / 'weights')
        self.criteria = self._initialize_criteria()

    def _initialize_criteria(self) -> Dict[str, CriteriaWeight]:
        """Initialize criteria with default weights."""
        try:
            criteria = {}
            for name, data in self.DEFAULT_CRITERIA.items():
                criteria[name] = CriteriaWeight(
                    name=name,
                    weight=data['weight'],
                    description=data['description'],
                    subcriteria=data.get('subcriteria')
                )
            return criteria
            
        except Exception as e:
            logger.error(f"Error initializing criteria: {e}")
            raise

    def set_weights(
        self,
        weights: Dict[str, Union[float, Dict[str, float]]]
    ) -> None:
        """
        Set custom weights for criteria and subcriteria.
        
        Args:
            weights: Dictionary of weights to update
        """
        try:
            # Validate weights sum to 1
            main_weights = [w['weight'] if isinstance(w, dict) else w 
                          for w in weights.values()]
            if not np.isclose(sum(main_weights), 1.0, rtol=1e-5):
                raise ValueError("Main criteria weights must sum to 1.0")
            
            # Update weights
            for criterion, weight in weights.items():
                if criterion not in self.criteria:
                    logger.warning(f"Unknown criterion: {criterion}")
                    continue
                
                if isinstance(weight, dict):
                    # Update main criterion weight
                    self.criteria[criterion].weight = weight['weight']
                    
                    # Update subcriteria if provided
                    if 'subcriteria' in weight:
                        if not np.isclose(sum(weight['subcriteria'].values()), 1.0, rtol=1e-5):
                            raise ValueError(f"Subcriteria weights for {criterion} must sum to 1.0")
                        self.criteria[criterion].subcriteria = weight['subcriteria']
                else:
                    self.criteria[criterion].weight = weight
            
            logger.info("Successfully updated criteria weights")
            
        except Exception as e:
            logger.error(f"Error setting weights: {e}")
            raise

    def calculate_ahp_weights(
        self,
        comparison_matrix: np.ndarray,
        criteria_names: List[str]
    ) -> Dict[str, float]:
        """
        Calculate weights using Analytic Hierarchy Process (AHP).
        
        Args:
            comparison_matrix: Pairwise comparison matrix
            criteria_names: List of criteria names
            
        Returns:
            Dictionary of calculated weights
        """
        try:
            # Check matrix dimensions
            n = len(criteria_names)
            if comparison_matrix.shape != (n, n):
                raise ValueError("Comparison matrix dimensions do not match criteria count")
            
            # Calculate eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eig(comparison_matrix)
            
            # Get principal eigenvector
            principal_eigenval_idx = np.argmax(eigenvals.real)
            principal_eigenvec = eigenvecs[:, principal_eigenval_idx].real
            
            # Normalize weights
            weights = principal_eigenvec / np.sum(principal_eigenvec)
            
            # Calculate consistency ratio
            ci = (np.max(eigenvals.real) - n) / (n - 1)
            ri = self._get_random_index(n)
            cr = ci / ri if ri != 0 else 0
            
            if cr > 0.1:
                logger.warning(f"Consistency ratio ({cr:.3f}) exceeds 0.1")
            
            return dict(zip(criteria_names, weights))
            
        except Exception as e:
            logger.error(f"Error calculating AHP weights: {e}")
            raise

    def _get_random_index(self, n: int) -> float:
        """Get random index for AHP consistency check."""
        random_indices = {
            1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
        }
        return random_indices.get(n, 1.5)

    def calculate_weighted_scores(
        self,
        scores: Dict[str, Union[float, Dict[str, float]]]
    ) -> float:
        """
        Calculate final weighted score.
        
        Args:
            scores: Dictionary of scores for each criterion
            
        Returns:
            Final weighted score
        """
        try:
            final_score = 0.0
            
            for criterion, criterion_data in self.criteria.items():
                if criterion not in scores:
                    logger.warning(f"Missing scores for criterion: {criterion}")
                    continue
                
                if isinstance(scores[criterion], dict) and criterion_data.subcriteria:
                    # Calculate weighted subscore
                    subscore = sum(
                        scores[criterion].get(sub, 0) * weight
                        for sub, weight in criterion_data.subcriteria.items()
                    )
                else:
                    subscore = scores[criterion]
                
                final_score += subscore * criterion_data.weight
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating weighted scores: {e}")
            raise

    def save_weights(self, path: Optional[Path] = None) -> Path:
        """Save current weights configuration."""
        try:
            if path is None:
                path = self.output_dir / f"weights_{create_timestamp_id()}.json"
            
            weights_data = {
                name: {
                    'weight': cw.weight,
                    'description': cw.description,
                    'subcriteria': cw.subcriteria
                }
                for name, cw in self.criteria.items()
            }
            
            with open(path, 'w') as f:
                json.dump(weights_data, f, indent=2)
            
            logger.info(f"Successfully saved weights to {path}")
            return path
            
        except Exception as e:
            logger.error(f"Error saving weights: {e}")
            raise

    def load_weights(self, path: Path) -> None:
        """Load weights configuration from file."""
        try:
            with open(path, 'r') as f:
                weights_data = json.load(f)
            
            self.criteria = {
                name: CriteriaWeight(
                    name=name,
                    weight=data['weight'],
                    description=data['description'],
                    subcriteria=data.get('subcriteria')
                )
                for name, data in weights_data.items()
            }
            
            logger.info(f"Successfully loaded weights from {path}")
            
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            raise

    def generate_weight_report(self) -> pd.DataFrame:
        """Generate a report of current weights configuration."""
        try:
            rows = []
            
            for name, cw in self.criteria.items():
                # Add main criterion
                rows.append({
                    'level': 'Main',
                    'criterion': name,
                    'weight': cw.weight,
                    'description': cw.description
                })
                
                # Add subcriteria
                if cw.subcriteria:
                    for sub_name, sub_weight in cw.subcriteria.items():
                        rows.append({
                            'level': 'Sub',
                            'criterion': f"{name} - {sub_name}",
                            'weight': sub_weight,
                            'description': ''
                        })
            
            return pd.DataFrame(rows)
            
        except Exception as e:
            logger.error(f"Error generating weight report: {e}")
            raise 