"""
Stock Technical Indicators Utility

This module provides standardized technical indicator calculations for all
stock prediction algorithms. All indicators are calculated from OHLC data
only - volume is not used as some stocks may not have volume data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class StockIndicators:
    """
    Utility class for calculating technical indicators from stock OHLC data.
    
    All indicators are calculated from Open, High, Low, Close data only.
    Volume is not used as some stocks may not have volume data.
    """
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for stock prediction.
        
        Args:
            df: DataFrame with columns ['date', 'open', 'high', 'low', 'close']
            
        Returns:
            DataFrame with original data plus technical indicators
        """
        if df is None or len(df) == 0:
            return df
        
        df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate all indicators
        df = StockIndicators._add_moving_averages(df)
        df = StockIndicators._add_exponential_moving_averages(df)
        df = StockIndicators._add_rsi(df)
        df = StockIndicators._add_macd(df)
        df = StockIndicators._add_bollinger_bands(df)
        df = StockIndicators._add_momentum_indicators(df)
        df = StockIndicators._add_volatility_indicators(df)
        df = StockIndicators._add_price_ratios(df)
        df = StockIndicators._add_lagged_features(df)
        df = StockIndicators._add_rolling_statistics(df)
        
        # Final cleanup: replace any remaining infinity values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    @staticmethod
    def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple Moving Averages (SMA)."""
        periods = [5, 10, 20, 50, 200]
        
        for period in periods:
            if len(df) >= period:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                # Handle division by zero in SMA ratios
                sma_ratio = df['close'] / df[f'sma_{period}']
                sma_ratio = sma_ratio.replace([np.inf, -np.inf], np.nan)
                df[f'sma_{period}_ratio'] = sma_ratio.fillna(1.0)  # Neutral ratio when SMA is 0
        
        return df
    
    @staticmethod
    def _add_exponential_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Add Exponential Moving Averages (EMA)."""
        periods = [12, 26]
        
        for period in periods:
            if len(df) >= period:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                # Handle division by zero in EMA ratios
                ema_ratio = df['close'] / df[f'ema_{period}']
                ema_ratio = ema_ratio.replace([np.inf, -np.inf], np.nan)
                df[f'ema_{period}_ratio'] = ema_ratio.fillna(1.0)  # Neutral ratio when EMA is 0
        
        return df
    
    @staticmethod
    def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index (RSI)."""
        if len(df) < period + 1:
            df['rsi'] = 50.0  # Neutral RSI
            return df
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Handle division by zero and infinity
        rs = gain / loss
        rs = rs.replace([np.inf, -np.inf], np.nan)
        rs = rs.fillna(1.0)  # Neutral RS when loss is 0
        
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Fill NaN values with neutral RSI and handle infinity
        df['rsi'] = df['rsi'].replace([np.inf, -np.inf], 50.0)
        df['rsi'] = df['rsi'].fillna(50.0)
        
        return df
    
    @staticmethod
    def _add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD (Moving Average Convergence Divergence)."""
        if len(df) < slow:
            df['macd'] = 0.0
            df['macd_signal'] = 0.0
            df['macd_histogram'] = 0.0
            return df
        
        # Calculate EMAs
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        
        # MACD line
        df['macd'] = ema_fast - ema_slow
        
        # Signal line
        df['macd_signal'] = df['macd'].ewm(span=signal).mean()
        
        # Histogram
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    @staticmethod
    def _add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands."""
        if len(df) < period:
            df['bb_upper'] = df['close']
            df['bb_middle'] = df['close']
            df['bb_lower'] = df['close']
            df['bb_width'] = 0.0
            df['bb_position'] = 0.5
            return df
        
        # Middle band (SMA)
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        
        # Standard deviation
        bb_std = df['close'].rolling(window=period).std()
        
        # Upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (bb_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (bb_std * std_dev)
        
        # Band width - handle division by zero
        bb_width = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        bb_width = bb_width.replace([np.inf, -np.inf], np.nan)
        df['bb_width'] = bb_width.fillna(0.0)
        
        # Position within bands - handle division by zero
        bb_range = df['bb_upper'] - df['bb_lower']
        bb_position = (df['close'] - df['bb_lower']) / bb_range
        bb_position = bb_position.replace([np.inf, -np.inf], np.nan)
        df['bb_position'] = bb_position.fillna(0.5)  # Middle position when range is 0
        
        return df
    
    @staticmethod
    def _add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # Multi-day returns
        for days in [1, 5, 10]:
            df[f'return_{days}d'] = df['close'].pct_change(periods=days)
        
        # Price momentum (rate of change)
        for days in [5, 10, 20]:
            if len(df) >= days:
                df[f'momentum_{days}d'] = df['close'] / df['close'].shift(days) - 1
        
        return df
    
    @staticmethod
    def _add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # Rolling volatility
        for window in [5, 10, 20]:
            if len(df) >= window:
                df[f'volatility_{window}d'] = df['price_change'].rolling(window=window).std()
        
        # Average True Range (ATR)
        if len(df) >= 14:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df['atr'] = true_range.rolling(window=14).mean()
        
        return df
    
    @staticmethod
    def _add_price_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """Add price ratio indicators."""
        # High-Low ratio - handle division by zero
        hl_ratio = df['high'] / df['low']
        hl_ratio = hl_ratio.replace([np.inf, -np.inf], np.nan)
        df['hl_ratio'] = hl_ratio.fillna(1.0)  # Neutral ratio when low is 0
        
        # Open-Close ratio - handle division by zero
        oc_ratio = df['open'] / df['close']
        oc_ratio = oc_ratio.replace([np.inf, -np.inf], np.nan)
        df['oc_ratio'] = oc_ratio.fillna(1.0)  # Neutral ratio when close is 0
        
        # Price position within day range - handle division by zero
        day_range = df['high'] - df['low']
        price_position = (df['close'] - df['low']) / day_range
        price_position = price_position.replace([np.inf, -np.inf], np.nan)
        df['price_position'] = price_position.fillna(0.5)  # Middle position when range is 0
        
        # Body size (Open-Close range) - handle division by zero
        body_size = np.abs(df['close'] - df['open']) / day_range
        body_size = body_size.replace([np.inf, -np.inf], np.nan)
        df['body_size'] = body_size.fillna(0.0)  # No body when range is 0
        
        return df
    
    @staticmethod
    def _add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged price features."""
        # Lagged close prices
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            # Handle division by zero in lagged ratios
            lag_ratio = df['close'] / df[f'close_lag_{lag}']
            lag_ratio = lag_ratio.replace([np.inf, -np.inf], np.nan)
            df[f'close_lag_{lag}_ratio'] = lag_ratio.fillna(1.0)  # Neutral ratio when lagged value is 0
        
        # Lagged high and low
        for lag in [1, 2, 5]:
            df[f'high_lag_{lag}'] = df['high'].shift(lag)
            df[f'low_lag_{lag}'] = df['low'].shift(lag)
        
        return df
    
    @staticmethod
    def _add_rolling_statistics(df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics."""
        for window in [5, 10, 20]:
            if len(df) >= window:
                # Rolling min/max
                df[f'close_min_{window}'] = df['close'].rolling(window=window).min()
                df[f'close_max_{window}'] = df['close'].rolling(window=window).max()
                
                # Rolling standard deviation
                df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
                
                # Position within rolling range - handle division by zero
                rolling_range = df[f'close_max_{window}'] - df[f'close_min_{window}']
                close_pos = (df['close'] - df[f'close_min_{window}']) / rolling_range
                close_pos = close_pos.replace([np.inf, -np.inf], np.nan)
                df[f'close_position_{window}'] = close_pos.fillna(0.5)  # Middle position when range is 0
        
        return df
    
    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> List[str]:
        """
        Get list of technical indicator columns for model training.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            List of feature column names (excludes OHLC, date, and target columns)
        """
        exclude_columns = [
            'date', 'open', 'high', 'low', 'close', 'volume', 
            'adjusted_close', 'currency', 'data_source'
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Filter to numeric columns only
        numeric_columns = []
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col)
        
        return numeric_columns
    
    @staticmethod
    def prepare_training_data(df: pd.DataFrame, target_column: str = 'close', 
                            lookback_days: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for machine learning training.
        
        DEPRECATED: This method returns raw prices in y, which is not used by the training pipeline.
        For model training, use DataLoader.prepare_training_data() which returns percentage changes.
        
        This method is kept for backward compatibility and potential future use in
        prediction pipelines that need raw prices.
        
        Args:
            df: DataFrame with technical indicators (OHLC data)
            target_column: Column to use as target variable
            lookback_days: Number of days to look back for prediction
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        if df is None or len(df) == 0:
            return np.array([]), np.array([])
        
        # Get feature columns
        feature_columns = StockIndicators.get_feature_columns(df)
        
        if not feature_columns:
            return np.array([]), np.array([])
        
        # Create feature matrix
        X = df[feature_columns].values
        
        # Create target vector (shifted for next-day prediction)
        y = df[target_column].shift(-lookback_days).values
        
        # Remove last rows (no target for prediction)
        X = X[:-lookback_days]
        y = y[:-lookback_days]
        
        # Remove any rows with NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        return X, y
