"""
Data Pipeline and Feature Engineering Utilities

This module provides comprehensive data processing and feature engineering
capabilities for stock prediction models.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering class for stock data"""
    
    @staticmethod
    def calculate_sma(data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = FeatureEngineer.calculate_ema(data, fast)
        ema_slow = FeatureEngineer.calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = FeatureEngineer.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = FeatureEngineer.calculate_sma(data, window)
        std = data.rolling(window=window).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    
    @staticmethod
    def add_lag_features(data: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """Add lag features for time series"""
        df = data.copy()
        for lag in lags:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
        return df


class DataPipeline:
    """Main data pipeline class for stock prediction"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data pipeline.
        
        Args:
            data_dir: Root directory containing stock data
        """
        self.data_dir = data_dir
        self.feature_engineer = FeatureEngineer()
        
    def load_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load stock data from CSV files.
        
        Args:
            symbol: Stock symbol to load
            
        Returns:
            Combined DataFrame with historical data
        """
        try:
            # Determine stock category reliably
            category = self._categorize_stock(symbol)
            
            # Load past data (2020-2024)
            past_file = os.path.join(self.data_dir, 'past', category, 'individual_files', f'{symbol}.csv')
            latest_file = os.path.join(self.data_dir, 'latest', category, 'individual_files', f'{symbol}.csv')
            
            historical_data = []
            
            # Load past data
            if os.path.exists(past_file):
                try:
                    df_past = pd.read_csv(past_file)
                    df_past['date'] = pd.to_datetime(df_past['date']).dt.date
                    historical_data.append(df_past)
                    logger.info(f"Loaded {len(df_past)} records from past data")
                except Exception as e:
                    logger.warning(f"Could not read past data for {symbol}: {e}")
            
            # Load latest data
            if os.path.exists(latest_file):
                try:
                    df_latest = pd.read_csv(latest_file)
                    df_latest['date'] = pd.to_datetime(df_latest['date']).dt.date
                    historical_data.append(df_latest)
                    logger.info(f"Loaded {len(df_latest)} records from latest data")
                except Exception as e:
                    logger.warning(f"Could not read latest data for {symbol}: {e}")
            
            if not historical_data:
                logger.error(f"No data files found for {symbol}")
                return None
            
            # Combine all data
            combined_df = pd.concat(historical_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['date']).sort_values('date')
            
            logger.info(f"Combined data for {symbol}: {len(combined_df)} records from {combined_df['date'].min()} to {combined_df['date'].max()}")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return None
    
    def _categorize_stock(self, symbol: str) -> str:
        """Categorize stock as US or Indian using shared utilities"""
        try:
            from shared.utilities import categorize_stock
            return categorize_stock(symbol)
        except Exception:
            # Robust fallback if shared utility cannot be imported
            symbol_upper = symbol.upper().strip() if symbol else ''
            if symbol_upper.endswith(('.NS', '.BO')):
                return 'ind_stocks'
            elif symbol_upper.endswith('.US'):
                return 'us_stocks'
            return 'us_stocks'
    
    def build_features(self, df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
        """
        Build comprehensive features for stock prediction.
        
        Args:
            df: Stock data DataFrame
            lookback: Number of days to look back for features
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Ensure we have enough data
        if len(df) < lookback:
            logger.warning(f"Insufficient data: {len(df)} rows, need at least {lookback}")
            return df
        
        # Basic price features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Moving averages
        df['sma_20'] = self.feature_engineer.calculate_sma(df['close'], 20)
        df['sma_50'] = self.feature_engineer.calculate_sma(df['close'], 50)
        df['sma_200'] = self.feature_engineer.calculate_sma(df['close'], 200)
        
        # Exponential moving averages
        df['ema_12'] = self.feature_engineer.calculate_ema(df['close'], 12)
        df['ema_26'] = self.feature_engineer.calculate_ema(df['close'], 26)
        
        # MACD
        macd_data = self.feature_engineer.calculate_macd(df['close'], 12, 26, 9)
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        # RSI
        df['rsi_14'] = self.feature_engineer.calculate_rsi(df['close'], 14)
        
        # Bollinger Bands
        bb_data = self.feature_engineer.calculate_bollinger_bands(df['close'], 20, 2)
        df['bb_upper'] = bb_data['upper']
        df['bb_middle'] = bb_data['middle']
        df['bb_lower'] = bb_data['lower']
        df['bb_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
        df['bb_position'] = (df['close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
        
        # Lag features
        lag_periods = [1, 2, 3, 5, 10, 30]
        df = self.feature_engineer.add_lag_features(df, lag_periods)
        
        # Additional technical indicators
        df['price_volatility'] = df['close'].rolling(window=20).std()
        
        # Drop rows with NaN values (due to rolling calculations)
        df = df.dropna()
        
        logger.info(f"Feature engineering complete. Final dataset shape: {df.shape}")
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, target_col: str = 'close', 
                            test_size: float = 0.2, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Prepare training and test data for time series prediction.
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            test_size: Fraction of data to use for testing
            lookback: Number of previous days to use for prediction
            
        Returns:
            X_train, X_test, y_train, y_test, metadata
        """
        # Select feature columns (exclude date, target, and non-numeric columns)
        feature_cols = [col for col in df.columns if col not in ['date', target_col, 'currency'] and df[col].dtype in ['float64', 'int64']]
        
        # Create sequences for time series prediction
        X, y = [], []
        for i in range(lookback, len(df)):
            X.append(df[feature_cols].iloc[i-lookback:i].values)
            y.append(df[target_col].iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Time series split (chronological)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Metadata
        metadata = {
            'feature_columns': feature_cols,
            'n_features': len(feature_cols),
            'lookback': lookback,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'last_date': df['date'].iloc[-1].isoformat() if 'date' in df.columns else None,
            'n_points': len(df)
        }
        
        logger.info(f"Training data prepared: {X_train.shape} train, {X_test.shape} test")
        return X_train, X_test, y_train, y_test, metadata
    
    def get_scalers(self, X_train: np.ndarray, model_type: str = 'standard') -> Dict[str, Any]:
        """
        Get appropriate scalers for the data.
        
        Args:
            X_train: Training features
            model_type: Type of model ('lstm' uses MinMaxScaler, others use StandardScaler)
            
        Returns:
            Dictionary with scalers
        """
        if model_type == 'lstm':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        # Reshape for scaling (flatten time series for scaling)
        original_shape = X_train.shape
        X_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_scaled = scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(original_shape)
        
        return {
            'scaler': scaler,
            'scaled_data': X_scaled,
            'scaler_type': model_type
        }


def build_features_for_symbol(symbol: str, csv_path: Optional[str] = None, 
                            lookback: int = 60, max_data_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Main function to build features for a stock symbol.
    
    Args:
        symbol: Stock symbol
        csv_path: Optional path to CSV file (if None, will search in data directory)
        lookback: Number of days to look back
        max_data_points: Maximum number of data points to use (None for all)
        
    Returns:
        X_train, X_test, y_train, y_test, scalers, metadata
    """
    pipeline = DataPipeline()
    
    # Load data
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date']).dt.date
    else:
        df = pipeline.load_stock_data(symbol)
        if df is None:
            raise ValueError(f"Could not load data for symbol: {symbol}")
    
    # Limit data points if specified
    if max_data_points and len(df) > max_data_points:
        df = df.tail(max_data_points)
        logger.info(f"Limited data to last {max_data_points} points")
    
    # Build features
    df_features = pipeline.build_features(df, lookback)
    
    if len(df_features) == 0:
        raise ValueError(f"No features could be built for symbol: {symbol}")
    
    # Prepare training data
    X_train, X_test, y_train, y_test, metadata = pipeline.prepare_training_data(
        df_features, lookback=lookback
    )
    
    # Get scalers
    scalers = pipeline.get_scalers(X_train, 'standard')
    
    return X_train, X_test, y_train, y_test, scalers, metadata


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }


def horizon_to_days(horizon: str) -> int:
    """
    Convert horizon string to number of days.
    
    Args:
        horizon: Horizon string (1d, 1w, 1m, 1y, 5y)
        
    Returns:
        Number of days
    """
    horizon_map = {
        '1d': 1,
        '1w': 7,
        '1m': 30,
        '1y': 365,
        '5y': 1825
    }
    
    if horizon not in horizon_map:
        raise ValueError(f"Invalid horizon: {horizon}. Must be one of {list(horizon_map.keys())}")
    
    return horizon_map[horizon]


class EnsemblePredictor:
    """
    Ensemble prediction system that combines multiple models
    using weighted averaging based on validation performance.
    """
    
    def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble predictor.
        
        Args:
            models: Dictionary of trained models {name: model}
            weights: Optional weights for each model (if None, will be calculated from RMSE)
        """
        self.models = models
        self.weights = weights or {}
        self.model_metrics = {}
        
    def calculate_weights_from_rmse(self, validation_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate model weights based on inverse RMSE.
        
        Args:
            validation_metrics: Dictionary of metrics for each model
            
        Returns:
            Dictionary of weights for each model
        """
        weights = {}
        rmse_values = {}
        
        # Extract RMSE values
        for model_name, metrics in validation_metrics.items():
            if 'rmse' in metrics and metrics['rmse'] > 0:
                rmse_values[model_name] = metrics['rmse']
        
        if not rmse_values:
            # Equal weights if no RMSE available
            equal_weight = 1.0 / len(self.models)
            return {name: equal_weight for name in self.models.keys()}
        
        # Calculate inverse RMSE weights
        inverse_rmse = {name: 1.0 / rmse for name, rmse in rmse_values.items()}
        total_inverse_rmse = sum(inverse_rmse.values())
        
        # Normalize weights
        for name in self.models.keys():
            if name in inverse_rmse:
                weights[name] = inverse_rmse[name] / total_inverse_rmse
            else:
                weights[name] = 0.0
        
        return weights
    
    def predict_ensemble(self, X: np.ndarray, horizon: str = '1d') -> Dict[str, Any]:
        """
        Make ensemble prediction for given horizon.
        
        Args:
            X: Input features
            horizon: Prediction horizon
            
        Returns:
            Dictionary with ensemble prediction and metadata
        """
        predictions = {}
        individual_results = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict_sequence') and horizon != '1d':
                    # For multi-step prediction (LSTM)
                    steps = horizon_to_days(horizon)
                    pred = model.predict_sequence(X, steps)
                    if len(pred) == 1:
                        pred = pred[0]
                    else:
                        pred = pred[-1]  # Use last prediction for the horizon
                elif hasattr(model, 'predict_with_intervals'):
                    # For models with prediction intervals (ARIMA)
                    result = model.predict_with_intervals(X, horizon_to_days(horizon))
                    pred = result['predictions']
                    if hasattr(pred, '__len__') and len(pred) > 1:
                        pred = pred[-1]
                    else:
                        pred = pred[0] if hasattr(pred, '__len__') else pred
                else:
                    # Standard prediction
                    pred = model.predict(X)
                    if hasattr(pred, '__len__') and len(pred) > 1:
                        pred = pred[-1]
                    else:
                        pred = pred[0] if hasattr(pred, '__len__') else pred
                
                predictions[model_name] = float(pred)
                individual_results[model_name] = {
                    'prediction': float(pred),
                    'model_info': model.get_model_info()
                }
                
            except Exception as e:
                logger.warning(f"Model {model_name} prediction failed: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models produced valid predictions")
        
        # Calculate ensemble prediction
        if not self.weights:
            # Use equal weights if no weights specified
            equal_weight = 1.0 / len(predictions)
            self.weights = {name: equal_weight for name in predictions.keys()}
        
        # Weighted average
        ensemble_prediction = sum(
            predictions[model_name] * self.weights.get(model_name, 0)
            for model_name in predictions.keys()
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(predictions, individual_results)
        
        # Calculate price range
        price_range = self._calculate_price_range(ensemble_prediction, predictions)
        
        return {
            'predicted_price': ensemble_prediction,
            'confidence': confidence,
            'price_range': price_range,
            'individual_predictions': individual_results,
            'weights': self.weights,
            'ensemble_size': len(predictions)
        }
    
    def _calculate_confidence(self, predictions: Dict[str, float], 
                            individual_results: Dict[str, Any]) -> float:
        """
        Calculate ensemble confidence based on prediction variance.
        
        Args:
            predictions: Dictionary of individual predictions
            individual_results: Detailed results from each model
            
        Returns:
            Confidence score (0-100)
        """
        if len(predictions) < 2:
            return 50.0  # Default confidence for single model
        
        # Calculate weighted variance
        mean_pred = sum(predictions.values()) / len(predictions)
        weighted_variance = sum(
            self.weights.get(name, 0) * (pred - mean_pred) ** 2
            for name, pred in predictions.items()
        )
        
        # Convert to confidence (0-100)
        if mean_pred == 0:
            return 50.0
        
        relative_std = (weighted_variance ** 0.5) / abs(mean_pred)
        confidence = max(0, min(100, 100 * (1 - relative_std)))
        
        return confidence
    
    def _calculate_price_range(self, ensemble_prediction: float, 
                             predictions: Dict[str, float]) -> List[float]:
        """
        Calculate price range based on individual predictions.
        
        Args:
            ensemble_prediction: Ensemble prediction
            predictions: Individual model predictions
            
        Returns:
            List with [lower_bound, upper_bound]
        """
        if not predictions:
            return [ensemble_prediction * 0.95, ensemble_prediction * 1.05]
        
        pred_values = list(predictions.values())
        std_dev = np.std(pred_values)
        
        # 95% confidence interval
        margin = 1.96 * std_dev
        lower_bound = ensemble_prediction - margin
        upper_bound = ensemble_prediction + margin
        
        return [max(0, lower_bound), upper_bound]


def predict_for_symbol(symbol: str, horizon: str = '1d', 
                      model_names: Optional[List[str]] = None,
                      max_data_points: Optional[int] = None) -> Dict[str, Any]:
    """
    Main function to generate predictions for a stock symbol.
    
    Args:
        symbol: Stock symbol to predict
        horizon: Prediction horizon (1d, 1w, 1m, 1y, 5y)
        model_names: List of model names to use (None for all)
        max_data_points: Maximum data points to use for training
        
    Returns:
        Dictionary with prediction results
    """
    from datetime import datetime
    import os
    
    # Import model classes
    from .real import (
        LSTMWrapper, RandomForestWrapper, ARIMAWrapper,
        SVRWrapper, LinearModelsWrapper, KNNWrapper
    )
    
    try:
        # Build features
        X_train, X_test, y_train, y_test, scalers, metadata = build_features_for_symbol(
            symbol, max_data_points=max_data_points
        )
        
        # Default models if none specified
        if model_names is None:
            model_names = ['lstm', 'random_forest', 'arima', 'svr', 'linear_ridge', 'knn']
        
        # Initialize models
        models = {}
        model_classes = {
            'lstm': LSTMWrapper,
            'random_forest': RandomForestWrapper,
            'arima': ARIMAWrapper,
            'svr': SVRWrapper,
            'linear_ridge': lambda **kwargs: LinearModelsWrapper(model_type='ridge', **kwargs),
            'linear_lasso': lambda **kwargs: LinearModelsWrapper(model_type='lasso', **kwargs),
            'knn': KNNWrapper
        }
        
        # Train models
        trained_models = {}
        validation_metrics = {}
        
        for model_name in model_names:
            if model_name not in model_classes:
                logger.warning(f"Unknown model: {model_name}")
                continue
            
            try:
                logger.info(f"Training {model_name} for {symbol}")
                model = model_classes[model_name]()
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate on test set
                y_pred = model.predict(X_test)
                metrics = calculate_metrics(y_test, y_pred)
                validation_metrics[model_name] = metrics
                
                trained_models[model_name] = model
                logger.info(f"{model_name} trained successfully. RMSE: {metrics['rmse']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        if not trained_models:
            raise ValueError("No models could be trained successfully")
        
        # Create ensemble
        ensemble = EnsemblePredictor(trained_models)
        ensemble.weights = ensemble.calculate_weights_from_rmse(validation_metrics)
        
        # Make prediction
        result = ensemble.predict_ensemble(X_test[-1:], horizon)
        
        # Prepare final result
        time_frame_days = horizon_to_days(horizon)
        
        return {
            'symbol': symbol,
            'horizon': horizon,
            'predicted_price': result['predicted_price'],
            'confidence': result['confidence'],
            'price_range': result['price_range'],
            'time_frame_days': time_frame_days,
            'model_info': {
                'algorithm': 'Ensemble (weighted)',
                'members': list(trained_models.keys()),
                'weights': result['weights'],
                'ensemble_size': result['ensemble_size']
            },
            'data_points_used': metadata['n_points'],
            'last_updated': datetime.utcnow().isoformat() + 'Z',
            'currency': 'USD',  # Default currency
            'individual_predictions': result['individual_predictions']
        }
        
    except Exception as e:
        logger.error(f"Prediction failed for {symbol}: {e}")
        raise
