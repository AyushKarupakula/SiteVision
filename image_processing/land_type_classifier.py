import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib

from config.settings import DATA_DIR, ML_SETTINGS
from utils.helpers import ensure_directory, create_timestamp_id

logger = logging.getLogger(__name__)

class LandTypeClassifier:
    """Classifies land types using extracted features from satellite imagery."""
    
    LAND_TYPES = {
        0: 'urban',
        1: 'agricultural',
        2: 'forest',
        3: 'water',
        4: 'barren',
        5: 'wetland'
    }
    
    def __init__(self, model_type: str = 'rf'):
        """
        Initialize the land type classifier.
        
        Args:
            model_type: Type of model to use ('rf' for Random Forest or 'cnn' for CNN)
        """
        self.model_type = model_type
        self.models_dir = ensure_directory(DATA_DIR / 'models' / 'land_type')
        self.model = None
        self.scaler = StandardScaler()

    def build_model(self, input_shape: Tuple[int, ...]) -> None:
        """
        Build the classification model.
        
        Args:
            input_shape: Shape of input features
        """
        try:
            if self.model_type == 'rf':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42
                )
            else:  # CNN
                inputs = layers.Input(shape=input_shape)
                
                x = layers.Conv2D(64, (3, 3), activation='relu')(inputs)
                x = layers.MaxPooling2D((2, 2))(x)
                x = layers.Conv2D(128, (3, 3), activation='relu')(x)
                x = layers.MaxPooling2D((2, 2))(x)
                x = layers.Conv2D(128, (3, 3), activation='relu')(x)
                x = layers.Flatten()(x)
                x = layers.Dense(128, activation='relu')(x)
                x = layers.Dropout(0.5)(x)
                outputs = layers.Dense(len(self.LAND_TYPES), activation='softmax')(x)
                
                self.model = models.Model(inputs=inputs, outputs=outputs)
                self.model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
            
            logger.info(f"Successfully built {self.model_type.upper()} model")
            
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise

    def train(
        self,
        features: Union[np.ndarray, Path],
        labels: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the classifier on extracted features.
        
        Args:
            features: Array of features or path to features file
            labels: Array of land type labels
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary of training metrics
        """
        try:
            # Load features if path provided
            if isinstance(features, Path):
                features = np.load(features)['features']
            
            # Scale features
            features_scaled = self.scaler.fit_transform(
                features.reshape(len(features), -1)
            )
            
            if self.model_type == 'rf':
                # Train Random Forest
                self.model.fit(features_scaled, labels)
                
                # Calculate metrics
                train_score = self.model.score(features_scaled, labels)
                metrics = {
                    'accuracy': train_score
                }
                
            else:  # CNN
                # Reshape features for CNN
                features_reshaped = features_scaled.reshape(
                    len(features_scaled), *self.model.input_shape[1:]
                )
                
                # Train CNN
                history = self.model.fit(
                    features_reshaped,
                    labels,
                    epochs=ML_SETTINGS['epochs'],
                    batch_size=ML_SETTINGS['batch_size'],
                    validation_split=validation_split
                )
                
                metrics = {
                    'accuracy': history.history['accuracy'][-1],
                    'val_accuracy': history.history['val_accuracy'][-1],
                    'loss': history.history['loss'][-1],
                    'val_loss': history.history['val_loss'][-1]
                }
            
            # Save model and scaler
            self.save_model()
            
            logger.info(f"Successfully trained model with metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def predict(
        self,
        features: Union[np.ndarray, Path]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict land types from features.
        
        Args:
            features: Array of features or path to features file
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        try:
            # Load features if path provided
            if isinstance(features, Path):
                features = np.load(features)['features']
            
            # Scale features
            features_scaled = self.scaler.transform(
                features.reshape(len(features), -1)
            )
            
            if self.model_type == 'rf':
                # Get predictions and probabilities
                predictions = self.model.predict(features_scaled)
                probabilities = self.model.predict_proba(features_scaled)
            else:  # CNN
                # Reshape features for CNN
                features_reshaped = features_scaled.reshape(
                    len(features_scaled), *self.model.input_shape[1:]
                )
                
                # Get predictions and probabilities
                probabilities = self.model.predict(features_reshaped)
                predictions = np.argmax(probabilities, axis=1)
            
            logger.info("Successfully generated predictions")
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            raise

    def save_model(self, path: Optional[Path] = None) -> Path:
        """Save the trained model and scaler."""
        try:
            if path is None:
                path = self.models_dir / f"model_{create_timestamp_id()}"
            
            path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            if self.model_type == 'rf':
                model_path = path / 'model.joblib'
                joblib.dump(self.model, model_path)
            else:  # CNN
                model_path = path / 'model.h5'
                self.model.save(model_path)
            
            # Save scaler
            scaler_path = path / 'scaler.joblib'
            joblib.dump(self.scaler, scaler_path)
            
            # Save metadata
            metadata = {
                'model_type': self.model_type,
                'land_types': self.LAND_TYPES,
                'timestamp': create_timestamp_id()
            }
            
            metadata_path = path / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully saved model to {path}")
            return path
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, path: Path) -> None:
        """Load a trained model and scaler."""
        try:
            # Load metadata
            with open(path / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            
            self.model_type = metadata['model_type']
            
            # Load model
            if self.model_type == 'rf':
                model_path = path / 'model.joblib'
                self.model = joblib.load(model_path)
            else:  # CNN
                model_path = path / 'model.h5'
                self.model = models.load_model(model_path)
            
            # Load scaler
            scaler_path = path / 'scaler.joblib'
            self.scaler = joblib.load(scaler_path)
            
            logger.info(f"Successfully loaded model from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_land_type_distribution(
        self,
        predictions: np.ndarray
    ) -> pd.DataFrame:
        """
        Get distribution of predicted land types.
        
        Args:
            predictions: Array of predicted land type indices
            
        Returns:
            DataFrame with land type distribution
        """
        try:
            counts = pd.Series(predictions).value_counts()
            percentages = counts / len(predictions) * 100
            
            distribution = pd.DataFrame({
                'land_type': [self.LAND_TYPES[i] for i in counts.index],
                'count': counts.values,
                'percentage': percentages.values
            })
            
            return distribution.sort_values('count', ascending=False)
            
        except Exception as e:
            logger.error(f"Error calculating land type distribution: {e}")
            raise 