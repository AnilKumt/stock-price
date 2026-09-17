"""
Confidence Calculator Module

Calculates prediction confidence based on multiple factors:
- Model accuracy (R² score)
- Prediction variance (agreement between models)
- Stock volatility (historical price stability)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class ConfidenceCalculator:
    """
    Calculates confidence scores for stock predictions.
    
    Confidence is calculated on a scale of 0-100, where:
    - 0-50: Low confidence (red)
    - 50-70: Medium confidence (orange)
    - 70-100: High confidence (green)
    """
    
    # Weights for ensemble predictions
    WEIGHT_MODEL_ACCURACY = 0.30  # How well models perform on this stock
    WEIGHT_PREDICTION_VARIANCE = 0.25  # Agreement between models
    WEIGHT_STOCK_VOLATILITY = 0.20  # Historical price stability
    WEIGHT_TIME_DECAY = 0.25  # Time horizon impact (longer = less confident)
    
    # Weights for single model predictions
    WEIGHT_SINGLE_ACCURACY = 0.50  # Model R² score
    WEIGHT_SINGLE_VOLATILITY = 0.25  # Stock stability
    WEIGHT_SINGLE_TIME_DECAY = 0.25  # Time horizon impact
    
    # Model complexity multipliers
    # Complex models get slight boost as they can capture non-linear patterns
    MODEL_COMPLEXITY_MULTIPLIERS = {
        # Advanced models - better at capturing complex patterns
        'ann': 1.05,
        'cnn': 1.05,
        'autoencoder': 1.05,
        'arima': 1.05,
        
        # Ensemble models - inherently more robust
        'random_forest': 1.03,
        
        # Basic models - baseline
        'linear_regression': 1.0,
        'decision_tree': 1.0,
        'knn': 1.0,
        'svm': 1.0
    }
    
    def __init__(self):
        """Initialize the confidence calculator."""
        pass
    
    def calculate_ensemble_confidence(
        self,
        model_accuracies: Dict[str, float],
        model_predictions: Dict[str, float],
        historical_prices: np.ndarray,
        time_horizon_days: int = 1
    ) -> float:
        """
        Calculate confidence for ensemble predictions.
        
        Args:
            model_accuracies: Dictionary of model names to R² scores (0-1)
            model_predictions: Dictionary of model names to predicted prices
            historical_prices: Array of historical closing prices
            time_horizon_days: Number of days into future for prediction
            
        Returns:
            Confidence score (0-100)
        """
        try:
            # Component 1: Average model accuracy (0-100)
            accuracy_score = self._calculate_accuracy_score(model_accuracies)
            
            # Component 2: Prediction agreement (0-100)
            variance_score = self._calculate_variance_score(model_predictions)
            
            # Component 3: Stock stability (0-100)
            volatility_score = self._calculate_volatility_score(historical_prices)
            
            # Component 4: Time decay (0-100)
            time_decay_score = self._calculate_time_decay_score(time_horizon_days)
            
            # Weighted combination
            confidence = (
                self.WEIGHT_MODEL_ACCURACY * accuracy_score +
                self.WEIGHT_PREDICTION_VARIANCE * variance_score +
                self.WEIGHT_STOCK_VOLATILITY * volatility_score +
                self.WEIGHT_TIME_DECAY * time_decay_score
            )
            
            # Fallback if non-finite
            if not np.isfinite(confidence):
                return 50.0

            # Ensure bounds
            confidence = max(0.0, min(100.0, float(confidence)))
            
            logger.debug(
                f"Ensemble confidence: {confidence:.1f}% "
                f"(accuracy: {accuracy_score:.1f}, variance: {variance_score:.1f}, "
                f"volatility: {volatility_score:.1f}, time_decay: {time_decay_score:.1f})"
            )
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Error calculating ensemble confidence: {e}")
            return 50.0  # Default medium confidence
    
    def calculate_single_model_confidence(
        self,
        model_accuracy: float,
        historical_prices: np.ndarray,
        time_horizon_days: int = 1,
        model_name: Optional[str] = None
    ) -> float:
        """
        Calculate confidence for single model predictions.
        
        Args:
            model_accuracy: R² score for this model (0-1)
            historical_prices: Array of historical closing prices
            time_horizon_days: Number of days into future for prediction
            model_name: Name of the model for complexity multiplier
            
        Returns:
            Confidence score (0-100)
        """
        try:
            # Component 1: Model accuracy (0-100) with complexity multiplier
            accuracy_score = self._calculate_accuracy_score(
                model_accuracy, 
                model_name=model_name
            )
            
            # Component 2: Stock stability (0-100)
            volatility_score = self._calculate_volatility_score(historical_prices)
            
            # Component 3: Time decay (0-100)
            time_decay_score = self._calculate_time_decay_score(time_horizon_days)
            
            # Weighted combination
            confidence = (
                self.WEIGHT_SINGLE_ACCURACY * accuracy_score +
                self.WEIGHT_SINGLE_VOLATILITY * volatility_score +
                self.WEIGHT_SINGLE_TIME_DECAY * time_decay_score
            )
            
            # Fallback if non-finite
            if not np.isfinite(confidence):
                return 50.0

            # Ensure bounds
            confidence = max(0.0, min(100.0, float(confidence)))
            
            logger.debug(
                f"Single model confidence ({model_name}): {confidence:.1f}% "
                f"(accuracy: {accuracy_score:.1f}, volatility: {volatility_score:.1f}, "
                f"time_decay: {time_decay_score:.1f})"
            )
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Error calculating single model confidence: {e}")
            return 50.0  # Default medium confidence
    
    def _calculate_accuracy_score(self, model_accuracies: Dict[str, float], 
                                  model_name: Optional[str] = None) -> float:
        """
        Calculate score from model accuracies.
        
        Args:
            model_accuracies: Dictionary of R² scores (0-1) or single float for model_name
            model_name: Optional model name for single model calculation
            
        Returns:
            Score (0-100)
        """
        # Single model calculation
        if model_name is not None:
            if isinstance(model_accuracies, dict):
                accuracy = model_accuracies.get(model_name, 0.5)
            else:
                accuracy = model_accuracies if model_accuracies is not None else 0.5
            
            if accuracy is None or not np.isfinite(accuracy):
                accuracy = 0.5
            
            # CRITICAL FIX: Handle negative R² scores (model worse than baseline)
            # Negative R² means the model performs worse than just predicting the mean
            if accuracy < 0:
                # Map negative R² to very low confidence (0-10%)
                # R² of -1.0 → 0%, R² of 0.0 → 10%
                score = max(0.0, 10.0 * (1.0 + accuracy))
            else:
                # Apply complexity multiplier for positive R² scores
                multiplier = self.MODEL_COMPLEXITY_MULTIPLIERS.get(model_name, 1.0)
                score = (accuracy * 100) * multiplier
            
            return max(0.0, min(100.0, score))
        
        # Ensemble calculation
        if not model_accuracies:
            return 50.0
        
        # Process each model's R² score, handling negative values
        processed_accuracies = []
        for name, accuracy in model_accuracies.items():
            if accuracy is None or not np.isfinite(accuracy):
                accuracy = 0.5
            if accuracy < 0:
                # Negative R² scores get mapped to 0-0.1 range (0-10% contribution)
                processed_accuracy = max(0.0, 0.1 * (1.0 + accuracy))
            else:
                # Positive R² scores get complexity multiplier
                multiplier = self.MODEL_COMPLEXITY_MULTIPLIERS.get(name, 1.0)
                processed_accuracy = accuracy * multiplier
            processed_accuracies.append(processed_accuracy)
        
        if not processed_accuracies:
            return 50.0
            
        avg_accuracy = np.mean(processed_accuracies)
        if not np.isfinite(avg_accuracy):
            return 50.0
        
        # Convert to 0-100 scale
        score = avg_accuracy * 100
        
        return max(0.0, min(100.0, score))
    
    def _calculate_variance_score(self, model_predictions: Dict[str, float]) -> float:
        """
        Calculate score from prediction variance (agreement between models).
        
        Lower variance = higher agreement = higher confidence
        
        Args:
            model_predictions: Dictionary of predicted prices
            
        Returns:
            Score (0-100)
        """
        if len(model_predictions) < 2:
            return 70.0  # Default for single prediction
        
        predictions = [p for p in model_predictions.values() if p is not None and np.isfinite(p)]
        if len(predictions) < 2:
            return 70.0

        mean_pred = np.mean(predictions)
        
        if mean_pred == 0 or not np.isfinite(mean_pred):
            return 50.0
        
        # Calculate coefficient of variation (CV)
        std_dev = np.std(predictions)
        if not np.isfinite(std_dev):
            return 50.0

        cv = std_dev / abs(mean_pred)
        if not np.isfinite(cv):
            return 50.0
        
        # Convert CV to confidence score
        # CV of 0 (perfect agreement) = 100
        # CV of 0.1 (10% variation) = ~70
        # CV of 0.3 (30% variation) = ~30
        # CV of 0.5+ (50%+ variation) = 0
        
        if cv <= 0.02:  # Very tight agreement (< 2%)
            score = 100.0
        elif cv <= 0.05:  # Good agreement (< 5%)
            score = 100.0 - (cv - 0.02) * (100.0 - 85.0) / (0.05 - 0.02)
        elif cv <= 0.10:  # Moderate agreement (< 10%)
            score = 85.0 - (cv - 0.05) * (85.0 - 70.0) / (0.10 - 0.05)
        elif cv <= 0.20:  # Some disagreement (< 20%)
            score = 70.0 - (cv - 0.10) * (70.0 - 40.0) / (0.20 - 0.10)
        elif cv <= 0.50:  # High disagreement (< 50%)
            score = 40.0 - (cv - 0.20) * (40.0 - 10.0) / (0.50 - 0.20)
        else:  # Very high disagreement
            score = 10.0
        
        return max(0.0, min(100.0, score))
    
    def _calculate_volatility_score(self, historical_prices: np.ndarray) -> float:
        """
        Calculate score from stock volatility.
        
        Lower volatility = more stable = higher confidence
        
        Args:
            historical_prices: Array of historical closing prices
            
        Returns:
            Score (0-100)
        """
        if historical_prices is None or len(historical_prices) < 30:
            return 50.0  # Not enough data
        
        # Calculate daily returns
        prices = historical_prices[-252:]  # Last year (approx 252 trading days)
        
        denominators = prices[:-1]
        numerators = np.diff(prices)
        
        # Filter out zero/negative denominators or non-finite price values
        valid_mask = (denominators > 0) & np.isfinite(denominators) & np.isfinite(prices[1:])
        if not np.any(valid_mask):
            return 50.0
            
        returns = numerators[valid_mask] / denominators[valid_mask]
        returns = returns[np.isfinite(returns)]
        
        if len(returns) == 0:
            return 50.0

        # Calculate annualized volatility
        daily_volatility = np.std(returns)
        if not np.isfinite(daily_volatility):
            return 50.0

        annualized_volatility = daily_volatility * np.sqrt(252)
        if not np.isfinite(annualized_volatility):
            return 50.0
        
        # Convert volatility to confidence score
        # Volatility of 0.15 (15% annual) = ~85 score (low-medium volatility stock)
        # Volatility of 0.30 (30% annual) = ~70 score (medium volatility stock)
        # Volatility of 0.50 (50% annual) = ~50 score (high volatility stock)
        # Volatility of 1.0+ (100%+ annual) = ~20 score (very high volatility)
        
        if annualized_volatility <= 0.15:  # Low volatility
            score = 100.0 - annualized_volatility * (100.0 - 85.0) / 0.15
        elif annualized_volatility <= 0.30:  # Medium-low volatility
            score = 85.0 - (annualized_volatility - 0.15) * (85.0 - 70.0) / (0.30 - 0.15)
        elif annualized_volatility <= 0.50:  # Medium-high volatility
            score = 70.0 - (annualized_volatility - 0.30) * (70.0 - 50.0) / (0.50 - 0.30)
        elif annualized_volatility <= 1.0:  # High volatility
            score = 50.0 - (annualized_volatility - 0.50) * (50.0 - 20.0) / (1.0 - 0.50)
        else:  # Very high volatility
            score = 20.0 - min(20.0, (annualized_volatility - 1.0) * 10.0)
        
        return max(0.0, min(100.0, score))
    
    def _calculate_time_decay_score(self, time_horizon_days: int) -> float:
        """
        Calculate score from time horizon.
        
        Longer time horizons = higher uncertainty = lower confidence
        
        Args:
            time_horizon_days: Number of days into the future for prediction
            
        Returns:
            Score (0-100)
        """
        if time_horizon_days <= 0:
            return 100.0
        
        # Convert time horizon to confidence score
        # 1 day (1D): ~95-100 (very high confidence)
        # 7 days (1W): ~90-95 (high confidence)
        # 30 days (1M): ~80-85 (good confidence)
        # 365 days (1Y): ~65-70 (moderate confidence)
        # 1825 days (5Y): ~45-50 (lower confidence)
        
        if time_horizon_days <= 1:  # 1 day
            score = 100.0
        elif time_horizon_days <= 7:  # 1 week
            score = 100.0 - (time_horizon_days - 1) * (100.0 - 90.0) / (7 - 1)
        elif time_horizon_days <= 30:  # 1 month
            score = 90.0 - (time_horizon_days - 7) * (90.0 - 80.0) / (30 - 7)
        elif time_horizon_days <= 365:  # 1 year
            score = 80.0 - (time_horizon_days - 30) * (80.0 - 65.0) / (365 - 30)
        elif time_horizon_days <= 1825:  # 5 years
            score = 65.0 - (time_horizon_days - 365) * (65.0 - 45.0) / (1825 - 365)
        else:  # Beyond 5 years
            score = 45.0 - min(20.0, (time_horizon_days - 1825) * 0.01)
        
        return max(0.0, min(100.0, score))


# Global instance
confidence_calculator = ConfidenceCalculator()
