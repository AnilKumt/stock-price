"""
Model Interface for Stock Prediction Algorithms

This module defines the abstract base class that all ML models must implement
to ensure consistent interface across different algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class ModelInterface(ABC):
    """
    Abstract base class defining the interface for all ML models.
    
    All models must implement these methods to ensure consistency
    across the prediction pipeline.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the model with a name and optional parameters.
        
        Args:
            model_name: Human-readable name for the model
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.is_trained = False
        self.training_metrics = {}
        self.model_params = kwargs
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """
        Train the model on the provided data.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            
        Returns:
            self: Returns self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on (n_samples, n_features)
            
        Returns:
            predictions: Predicted values (n_samples,)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: File path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> 'ModelInterface':
        """
        Load a previously saved model from disk.
        
        Args:
            path: File path to load the model from
            
        Returns:
            self: Returns self for method chaining
        """
        pass
    
    def get_training_metrics(self) -> Dict[str, float]:
        """
        Get training metrics from the last fit operation.
        
        Returns:
            metrics: Dictionary of training metrics (MAE, RMSE, etc.)
        """
        return self.training_metrics.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information including parameters and metadata.
        
        Returns:
            info: Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'model_params': self.model_params,
            'training_metrics': self.training_metrics
        }
    
    def set_training_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Set training metrics after model training.
        
        Args:
            metrics: Dictionary of training metrics
        """
        self.training_metrics = metrics.copy()
        self.is_trained = True
    
    def validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Validate input data format and raise appropriate errors.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            
        Raises:
            ValueError: If input data is invalid
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (n_samples, n_features)")
        
        if X.shape[0] == 0:
            raise ValueError("X cannot be empty")
        
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("X contains NaN or infinity values")
        
        if y is not None:
            if not isinstance(y, np.ndarray):
                raise ValueError("y must be a numpy array")
            
            if y.ndim != 1:
                raise ValueError("y must be a 1D array")
            
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must have the same number of samples")
                
            if np.isnan(y).any() or np.isinf(y).any():
                raise ValueError("y contains NaN or infinity values")


class PredictionResult:
    """
    Container class for prediction results with confidence and metadata.
    """
    
    def __init__(self, 
                 predicted_price: float,
                 confidence: float,
                 price_range: Tuple[float, float],
                 time_frame_days: int,
                 model_info: Dict[str, Any],
                 data_points_used: int,
                 last_updated: str,
                 currency: str = "USD"):
        """
        Initialize prediction result.
        
        Args:
            predicted_price: The predicted price
            confidence: Confidence level (0-100)
            price_range: Lower and upper bounds of prediction
            time_frame_days: Number of days ahead predicted
            model_info: Information about the model used
            data_points_used: Number of data points used for training
            last_updated: ISO timestamp of when prediction was made
            currency: Currency of the prediction
        """
        # Validate and sanitize predicted price
        try:
            self.predicted_price = max(0.0, float(predicted_price))
            if np.isnan(self.predicted_price) or np.isinf(self.predicted_price):
                self.predicted_price = 0.0
        except (ValueError, TypeError):
            self.predicted_price = 0.0
            
        # Validate and clamp confidence level (0-100)
        try:
            conf_val = float(confidence)
            if np.isnan(conf_val) or np.isinf(conf_val):
                conf_val = 50.0
            if 0.0 <= conf_val <= 1.0:
                conf_val = conf_val * 100.0
            self.confidence = max(0.0, min(100.0, conf_val))
        except (ValueError, TypeError):
            self.confidence = 50.0
            
        # Validate price range bounds
        try:
            lower, upper = float(price_range[0]), float(price_range[1])
            if np.isnan(lower) or np.isinf(lower) or lower < 0:
                lower = self.predicted_price * 0.95
            if np.isnan(upper) or np.isinf(upper) or upper < 0:
                upper = self.predicted_price * 1.05
            if lower > upper:
                lower, upper = upper, lower
            self.price_range = (lower, upper)
        except (ValueError, TypeError, IndexError):
            self.price_range = (self.predicted_price * 0.95, self.predicted_price * 1.05)
            
        self.time_frame_days = max(1, int(time_frame_days)) if time_frame_days else 1
        self.model_info = model_info if isinstance(model_info, dict) else {}
        self.data_points_used = max(0, int(data_points_used)) if data_points_used else 0
        self.last_updated = str(last_updated) if last_updated else ""
        self.currency = str(currency).upper() if currency else "USD"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction result to dictionary."""
        return {
            'predicted_price': self.predicted_price,
            'confidence': self.confidence,
            'price_range': list(self.price_range),
            'time_frame_days': self.time_frame_days,
            'model_info': self.model_info,
            'data_points_used': self.data_points_used,
            'last_updated': self.last_updated,
            'currency': self.currency
        }
