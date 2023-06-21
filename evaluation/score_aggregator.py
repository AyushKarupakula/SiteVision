import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import DATA_DIR
from utils.helpers import ensure_directory, create_timestamp_id
from .criteria_weighting import CriteriaWeighting

logger = logging.getLogger(__name__)

class ScoreAggregator:
    """Aggregates and analyzes scores from various evaluation criteria."""
    
    def __init__(self):
        self.output_dir = ensure_directory(DATA_DIR / 'evaluation' / 'scores')
        self.weighting = CriteriaWeighting()

    def aggregate_scores(
        self,
        site_data: Dict[str, Dict[str, Union[float, Dict[str, float]]]],
        weights_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Aggregate scores for multiple sites.
        
        Args:
            site_data: Dictionary mapping site IDs to their criteria scores
            weights_path: Optional path to custom weights configuration
            
        Returns:
            DataFrame with aggregated scores
        """
        try:
            # Load custom weights if provided
            if weights_path:
                self.weighting.load_weights(weights_path)
            
            results = []
            for site_id, scores in site_data.items():
                # Calculate weighted score
                final_score = self.weighting.calculate_weighted_scores(scores)
                
                # Compile results
                result = {
                    'site_id': site_id,
                    'total_score': final_score
                }
                
                # Add individual criteria scores
                for criterion, score in scores.items():
                    if isinstance(score, dict):
                        for sub, sub_score in score.items():
                            result[f"{criterion}_{sub}"] = sub_score
                    else:
                        result[criterion] = score
                
                results.append(result)
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error aggregating scores: {e}")
            raise

    def rank_sites(
        self,
        scores_df: pd.DataFrame,
        min_score: float = 0.0,
        max_sites: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Rank sites based on their scores.
        
        Args:
            scores_df: DataFrame with site scores
            min_score: Minimum acceptable score
            max_sites: Maximum number of sites to return
            
        Returns:
            DataFrame with ranked sites
        """
        try:
            # Filter by minimum score
            qualified_sites = scores_df[scores_df['total_score'] >= min_score].copy()
            
            # Sort by total score
            ranked_sites = qualified_sites.sort_values(
                'total_score',
                ascending=False
            )
            
            # Add rank column
            ranked_sites['rank'] = range(1, len(ranked_sites) + 1)
            
            # Limit number of sites if specified
            if max_sites:
                ranked_sites = ranked_sites.head(max_sites)
            
            return ranked_sites
            
        except Exception as e:
            logger.error(f"Error ranking sites: {e}")
            raise

    def analyze_score_distribution(
        self,
        scores_df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze the distribution of scores.
        
        Args:
            scores_df: DataFrame with site scores
            
        Returns:
            Dictionary of statistical measures for each criterion
        """
        try:
            stats = {}
            
            # Analyze total score
            stats['total_score'] = {
                'mean': scores_df['total_score'].mean(),
                'median': scores_df['total_score'].median(),
                'std': scores_df['total_score'].std(),
                'min': scores_df['total_score'].min(),
                'max': scores_df['total_score'].max(),
                'q1': scores_df['total_score'].quantile(0.25),
                'q3': scores_df['total_score'].quantile(0.75)
            }
            
            # Analyze individual criteria
            criteria_cols = [col for col in scores_df.columns 
                           if col not in ['site_id', 'total_score', 'rank']]
            
            for criterion in criteria_cols:
                stats[criterion] = {
                    'mean': scores_df[criterion].mean(),
                    'median': scores_df[criterion].median(),
                    'std': scores_df[criterion].std(),
                    'min': scores_df[criterion].min(),
                    'max': scores_df[criterion].max(),
                    'q1': scores_df[criterion].quantile(0.25),
                    'q3': scores_df[criterion].quantile(0.75)
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error analyzing score distribution: {e}")
            raise

    def generate_score_report(
        self,
        scores_df: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate a comprehensive score report.
        
        Args:
            scores_df: DataFrame with site scores
            output_path: Optional path for saving the report
            
        Returns:
            Path to the generated report
        """
        try:
            if output_path is None:
                output_path = self.output_dir / f"score_report_{create_timestamp_id()}"
                output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate statistics
            stats = self.analyze_score_distribution(scores_df)
            
            # Create visualizations
            self._create_score_visualizations(scores_df, output_path)
            
            # Generate report content
            report = {
                'summary': {
                    'total_sites': len(scores_df),
                    'average_score': float(scores_df['total_score'].mean()),
                    'top_score': float(scores_df['total_score'].max()),
                    'bottom_score': float(scores_df['total_score'].min())
                },
                'statistics': stats,
                'top_sites': scores_df.head(10).to_dict(orient='records'),
                'visualizations': [str(p) for p in output_path.glob('*.png')]
            }
            
            # Save report
            report_path = output_path / 'report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Successfully generated score report at {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating score report: {e}")
            raise

    def _create_score_visualizations(
        self,
        scores_df: pd.DataFrame,
        output_dir: Path
    ) -> None:
        """Create visualizations of score distributions."""
        try:
            # Set style
            plt.style.use('seaborn')
            
            # Distribution of total scores
            plt.figure(figsize=(10, 6))
            sns.histplot(data=scores_df, x='total_score', bins=30)
            plt.title('Distribution of Total Scores')
            plt.savefig(output_dir / 'total_score_dist.png')
            plt.close()
            
            # Criteria comparison boxplot
            criteria_cols = [col for col in scores_df.columns 
                           if col not in ['site_id', 'total_score', 'rank']]
            
            plt.figure(figsize=(12, 6))
            scores_df[criteria_cols].boxplot()
            plt.xticks(rotation=45)
            plt.title('Distribution of Criteria Scores')
            plt.tight_layout()
            plt.savefig(output_dir / 'criteria_comparison.png')
            plt.close()
            
            # Correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                scores_df[criteria_cols].corr(),
                annot=True,
                cmap='coolwarm',
                center=0
            )
            plt.title('Criteria Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(output_dir / 'correlation_heatmap.png')
            plt.close()
            
            # Top sites comparison
            top_sites = scores_df.head(10)
            plt.figure(figsize=(12, 6))
            sns.barplot(data=top_sites, x='site_id', y='total_score')
            plt.title('Top 10 Sites Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'top_sites.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            raise

    def export_scores(
        self,
        scores_df: pd.DataFrame,
        output_format: str = 'csv'
    ) -> Path:
        """
        Export scores to file.
        
        Args:
            scores_df: DataFrame with site scores
            output_format: Format to export ('csv' or 'excel')
            
        Returns:
            Path to exported file
        """
        try:
            timestamp = create_timestamp_id()
            
            if output_format == 'csv':
                output_path = self.output_dir / f"scores_{timestamp}.csv"
                scores_df.to_csv(output_path, index=False)
            elif output_format == 'excel':
                output_path = self.output_dir / f"scores_{timestamp}.xlsx"
                scores_df.to_excel(output_path, index=False)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            logger.info(f"Successfully exported scores to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting scores: {e}")
            raise 