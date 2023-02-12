import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import cv2
import rasterio
from rasterio.features import shapes
from sklearn.cluster import KMeans
from skimage import feature, segmentation
import tensorflow as tf

from config.settings import DATA_DIR
from utils.helpers import ensure_directory, create_timestamp_id

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extracts features from satellite imagery for land suitability analysis."""
    
    def __init__(self):
        self.features_dir = ensure_directory(DATA_DIR / 'features')
        self._init_models()

    def _init_models(self):
        """Initialize any required models."""
        try:
            # Initialize pre-trained CNN for feature extraction
            base_model = tf.keras.applications.ResNet50V2(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3)
            )
            self.feature_model = tf.keras.Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('conv4_block6_out').output
            )
            
            logger.info("Successfully initialized feature extraction models")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def extract_features(
        self,
        image_path: Path,
        include_features: Optional[List[str]] = None
    ) -> Dict[str, Union[np.ndarray, Path]]:
        """
        Extract various features from satellite imagery.
        
        Args:
            image_path: Path to the preprocessed image
            include_features: List of feature types to extract
                            ('texture', 'edges', 'segments', 'deep')
                            
        Returns:
            Dictionary of extracted features
        """
        try:
            with rasterio.open(image_path) as src:
                image = src.read()
                profile = src.profile
                
            # Ensure image is in correct format (HWC)
            image = np.transpose(image, (1, 2, 0))
            
            features = {}
            all_features = ['texture', 'edges', 'segments', 'deep']
            feature_types = include_features or all_features
            
            # Extract requested features
            if 'texture' in feature_types:
                features['texture'] = self._extract_texture_features(image)
                
            if 'edges' in feature_types:
                features['edges'] = self._extract_edge_features(image)
                
            if 'segments' in feature_types:
                features['segments'] = self._extract_segmentation_features(image)
                
            if 'deep' in feature_types:
                features['deep'] = self._extract_deep_features(image)
            
            # Save features
            output_path = self.features_dir / f"features_{create_timestamp_id()}.npz"
            np.savez_compressed(
                output_path,
                **{k: v for k, v in features.items() if isinstance(v, np.ndarray)}
            )
            
            # Save metadata
            metadata = {
                'source_image': str(image_path),
                'feature_types': feature_types,
                'shape': image.shape,
                'crs': profile['crs'].to_string(),
                'transform': list(profile['transform'])
            }
            
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully extracted features to {output_path}")
            return {'features': output_path, 'metadata': metadata_path}
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise

    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture features using GLCM."""
        try:
            # Convert to grayscale if needed
            if image.shape[-1] > 1:
                gray = cv2.cvtColor(
                    (image * 255).astype(np.uint8),
                    cv2.COLOR_RGB2GRAY
                )
            else:
                gray = (image[:, :, 0] * 255).astype(np.uint8)
            
            # Calculate GLCM properties
            distances = [1, 2, 4]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            
            features = []
            glcm = feature.graycomatrix(gray, distances, angles, symmetric=True, normed=True)
            
            for prop in properties:
                features.append(feature.graycoprops(glcm, prop))
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting texture features: {e}")
            raise

    def _extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """Extract edge features using multiple methods."""
        try:
            # Convert to grayscale if needed
            if image.shape[-1] > 1:
                gray = cv2.cvtColor(
                    (image * 255).astype(np.uint8),
                    cv2.COLOR_RGB2GRAY
                )
            else:
                gray = (image[:, :, 0] * 255).astype(np.uint8)
            
            # Canny edges
            edges_canny = feature.canny(gray, sigma=2)
            
            # Sobel edges
            edges_sobel = np.hypot(
                cv2.Sobel(gray, cv2.CV_64F, 1, 0),
                cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            )
            
            return np.stack([edges_canny, edges_sobel])
            
        except Exception as e:
            logger.error(f"Error extracting edge features: {e}")
            raise

    def _extract_segmentation_features(self, image: np.ndarray) -> np.ndarray:
        """Extract segmentation features using multiple methods."""
        try:
            # Convert to appropriate format
            img = (image * 255).astype(np.uint8)
            
            # SLIC superpixels
            segments_slic = segmentation.slic(
                img,
                n_segments=100,
                compactness=10,
                sigma=1
            )
            
            # K-means clustering
            reshaped = img.reshape(-1, img.shape[-1])
            kmeans = KMeans(n_clusters=5, random_state=42)
            segments_kmeans = kmeans.fit_predict(reshaped).reshape(img.shape[:2])
            
            return np.stack([segments_slic, segments_kmeans])
            
        except Exception as e:
            logger.error(f"Error extracting segmentation features: {e}")
            raise

    def _extract_deep_features(self, image: np.ndarray) -> np.ndarray:
        """Extract deep features using pre-trained CNN."""
        try:
            # Preprocess image for ResNet
            img = cv2.resize(image, (224, 224))
            img = tf.keras.applications.resnet_v2.preprocess_input(img)
            img = np.expand_dims(img, axis=0)
            
            # Extract features
            features = self.feature_model.predict(img)
            
            return features.squeeze()
            
        except Exception as e:
            logger.error(f"Error extracting deep features: {e}")
            raise

    def combine_features(
        self,
        feature_paths: List[Path],
        output_path: Optional[Path] = None
    ) -> Path:
        """Combine multiple feature sets into a single array."""
        try:
            combined_features = []
            
            for path in feature_paths:
                features = np.load(path)
                combined_features.append(
                    np.concatenate([features[k] for k in features.files])
                )
            
            combined = np.concatenate(combined_features, axis=0)
            
            if output_path is None:
                output_path = self.features_dir / f"combined_features_{create_timestamp_id()}.npz"
            
            np.savez_compressed(output_path, features=combined)
            
            logger.info(f"Successfully combined features to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error combining features: {e}")
            raise 