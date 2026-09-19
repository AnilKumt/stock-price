"""
Data Loader for Stock Prediction

This module handles loading and preprocessing of stock data from both
historical (data/past) and latest (data/latest) directories.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings

from .config import config

# Add shared module to path to find currency_converter
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from currency_converter import currency_converter

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader for stock prediction system.
    
    Handles loading, combining, and preprocessing of stock data from
    historical and latest data directories.
    """
    
    def __init__(self):
        self.config = config
        self.cache = {}  # Simple cache for loaded data
        self.currency_converter = currency_converter
        
    def load_stock_data(self, symbol: str, category: str) -> Optional[pd.DataFrame]:
        """
        Load historical data for a stock (training uses ONLY data/past/).
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'RELIANCE')
            category: Stock category ('us_stocks' or 'ind_stocks')
            
        Returns:
            DataFrame with historical data from data/past/ directory only
        """
        cache_key = f"{symbol}_{category}_normalized_v1"
        if cache_key in self.cache:
            logger.debug(f"Using cached NORMALIZED data for {symbol}")
            return self.cache[cache_key]
        
        try:
            logger.info(f"[DEBUG] load_stock_data called for symbol: {symbol}, category: {category}")
            
            # Load ONLY historical data for training
            historical_data = self._load_historical_data(symbol, category)
            
            if historical_data is None or len(historical_data) == 0:
                logger.warning(f"No historical data found for {symbol}")
                return None

            # NORMALIZE CURRENCY BEFORE CACHING
            if category == 'ind_stocks':
                logger.info(f"[DEBUG] Passed 'ind_stocks' check for {symbol}.")
                logger.info(f"[DEBUG] Attempting to fetch exchange rate...")
                
                rate = self.currency_converter.get_live_exchange_rate()
                
                logger.info(f"[DEBUG] Exchange rate fetched: {rate}")
                
                if rate is None or rate <= 0:
                    logger.error(f"Failed to get valid exchange rate for {symbol}. Skipping stock.")
                    return None
                
                logger.info(f"Normalizing INR to USD for {symbol}...")
                price_cols = ['open', 'high', 'low', 'close']
                for col in price_cols:
                    if col in historical_data.columns:
                        historical_data[col] = historical_data[col] / rate
                
                logger.info(f"Normalization complete for {symbol} using rate {rate:.4f}")
            else:
                logger.info(f"[DEBUG] Skipped 'ind_stocks' check for {symbol} (category was: {category})")

            # Cache the result
            self.cache[cache_key] = historical_data
            logger.info(f"Loaded and processed {len(historical_data)} data points for {symbol}")
            return historical_data
                
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            return None
    
    def load_stock_data_with_current_price(self, symbol: str, category: str, 
                                           current_price: Optional[Dict] = None) -> Optional[pd.DataFrame]:
        """
        Load historical data + optionally append current live price for prediction.
        
        This method is used for making predictions with real-time data.
        
        Args:
            symbol: Stock symbol
            category: Stock category ('us_stocks' or 'ind_stocks')
            current_price: Optional dict with keys: 'close', 'open', 'high', 'low', 'date'
                          If None, uses only historical data
        
        Returns:
            DataFrame with historical data + current price (if provided)
        """
        try:
            # Load historical data
            df = self.load_stock_data(symbol, category)
            
            if df is None or len(df) == 0:
                logger.warning(f"No historical data available for {symbol}")
                return None
            
            # Append current price if provided
            if current_price:
                logger.info(f"Appending current live price for {symbol}")
                # Create a local copy to prevent in-place mutation of caller's dictionary
                current_price = current_price.copy()

                # NORMALIZE LIVE PRICE FOR INDIAN STOCKS
                if category == 'ind_stocks':
                    rate = self.currency_converter.get_live_exchange_rate()
                    if rate is None or rate <= 0:
                        logger.error(f"Failed to get exchange rate for {symbol} live price. Skipping append.")
                        return df
                    
                    current_price['close'] = current_price['close'] / rate
                    if 'open' in current_price:
                        current_price['open'] = current_price.get('open', current_price['close']) / rate
                    if 'high' in current_price:
                        current_price['high'] = current_price.get('high', current_price['close']) / rate
                    if 'low' in current_price:
                        current_price['low'] = current_price.get('low', current_price['close']) / rate
                    logger.info(f"Normalized live price for {symbol} using rate {rate:.4f}")
                
                # Create new row with current price
                new_row = pd.DataFrame([{
                    'date': current_price.get('date', pd.Timestamp.now()),
                    'open': current_price.get('open', current_price['close']),
                    'high': current_price.get('high', current_price['close']),
                    'low': current_price.get('low', current_price['close']),
                    'close': current_price['close'],
                    'volume': np.nan,  # Volume not available, set to NaN
                    'data_source': 'live'
                }])
                
                # Standardize column names
                new_row.columns = new_row.columns.str.lower()
                new_row['date'] = pd.to_datetime(new_row['date'], utc=False)
                
                # Append to historical data
                df = pd.concat([df, new_row], ignore_index=True)
                df = df.sort_values('date').reset_index(drop=True)
                
                logger.info(f"Added current price to {symbol}, total points: {len(df)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data with current price for {symbol}: {str(e)}")
            return None
    
    def _load_historical_data(self, symbol: str, category: str) -> Optional[pd.DataFrame]:
        """Load historical data from data/past directory with permanent fallback."""
        try:
            # Try data/past first (updated data takes precedence)
            file_path = os.path.join(
                self.config.PAST_DATA_DIR,
                category,
                'individual_files',
                f'{symbol}.csv'
            )
            
            data_source = 'past'
            
            if not os.path.exists(file_path):
                # Fallback to permanent directory for offline mode
                logger.debug(f"Historical file not found in data/past for {symbol}, trying permanent...")
                file_path = os.path.join(
                    self.config.PERMANENT_DATA_DIR,
                    category,
                    'individual_files',
                    f'{symbol}.csv'
                )
                data_source = 'permanent'
                
                if not os.path.exists(file_path):
                    logger.debug(f"File not found in either data/past or permanent: {symbol}")
                    return None
            
            logger.info(f"Loading {symbol} from {data_source} directory")
            
            df = pd.read_csv(file_path)
            
            # Standardize column names (handle case variations)
            df.columns = df.columns.str.lower()
            
            # Ensure required columns exist
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns in {symbol} historical data: {missing_columns}")
                return None
            
            # Convert date column - ensure timezone-naive for consistency
            df['date'] = pd.to_datetime(df['date'], utc=False)
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            # Add data source
            df['data_source'] = 'historical'
            
            logger.debug(f"Loaded {len(df)} historical records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {str(e)}")
            return None
    
    def _load_latest_data(self, symbol: str, category: str) -> Optional[pd.DataFrame]:
        """
        Load latest data from data/latest directory.
        
        NOTE: This method is kept for potential future use in predictions.
        It is NOT used during model training to ensure consistency.
        """
        try:
            file_path = os.path.join(
                self.config.LATEST_DATA_DIR,
                category,
                'individual_files',
                f'{symbol}.csv'
            )
            
            if not os.path.exists(file_path):
                logger.debug(f"Latest file not found: {file_path}")
                return None
            
            df = pd.read_csv(file_path)
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            
            # Ensure required columns exist
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns in {symbol} latest data: {missing_columns}")
                return None
            
            # Convert date column - ensure timezone-naive for consistency
            df['date'] = pd.to_datetime(df['date'], utc=False)
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            # Add data source
            df['data_source'] = 'latest'
            
            logger.debug(f"Loaded {len(df)} latest records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading latest data for {symbol}: {str(e)}")
            return None
    
    def _combine_data(self, historical_data: Optional[pd.DataFrame], 
                     latest_data: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Combine historical and latest data, removing duplicates.
        
        NOTE: This method is deprecated for training.
        Training now uses ONLY historical data from data/past/.
        Kept for potential future use in prediction pipeline.
        """
        if historical_data is None and latest_data is None:
            return None
        
        if historical_data is None:
            return latest_data
        
        if latest_data is None:
            return historical_data
        
        # Combine dataframes
        combined = pd.concat([historical_data, latest_data], ignore_index=True)
        
        # Remove duplicates based on date
        combined = combined.drop_duplicates(subset=['date'], keep='last')
        
        # Sort by date
        combined = combined.sort_values('date').reset_index(drop=True)
        
        # Remove any rows with missing essential data
        combined = combined.dropna(subset=['date', 'close'])
        
        return combined
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators and features for machine learning.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional features
        """
        if df is None or len(df) == 0:
            return df
        
        df = df.copy()
        
        try:
            # Price-based features
            df['price_change'] = df['close'].pct_change()
            df['price_change_abs'] = df['price_change'].abs()
            
            # Moving averages
            for window in self.config.MOVING_AVERAGES:
                if len(df) >= window:
                    df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
                    df[f'ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
            
            # Volatility features
            df['volatility'] = df['price_change'].rolling(
                window=self.config.VOLATILITY_WINDOW
            ).std()
            
            # RSI (Relative Strength Index)
            df['rsi'] = self._calculate_rsi(df['close'], self.config.RSI_PERIOD)
            
            # High-Low features
            df['hl_ratio'] = df['high'] / df['low']
            df['oc_ratio'] = df['open'] / df['close']
            
            # Price position within day range
            df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Lagged features
            for lag in [1, 2, 3, 5, 10]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                if len(df) >= window:
                    df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
                    df[f'close_min_{window}'] = df['close'].rolling(window=window).min()
                    df[f'close_max_{window}'] = df['close'].rolling(window=window).max()
            
            # Time-based features
            # Ensure date column is datetime type before using .dt accessor
            # Convert to timezone-naive datetime (remove timezone info)
            df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            
            # Remove rows with NaN values created by rolling calculations
            df = df.dropna()
            
            # Ensure consistent feature set
            df = self._standardize_features(df)
            
            logger.debug(f"Created {len(df.columns)} features for prediction")
            return df
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            return df
    
    def _standardize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all stocks have the same feature set by adding missing features with default values.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with standardized feature set
        """
        # Define the expected feature set (38 features total)
        expected_features = [
            'price_change', 'price_change_abs', 'volatility', 'rsi',
            'hl_ratio', 'oc_ratio', 'price_position', 'day_of_week', 'month', 'quarter',
            'market_type'
        ]
        
        # Add moving averages
        for window in self.config.MOVING_AVERAGES:
            expected_features.extend([f'ma_{window}', f'ma_{window}_ratio'])
        
        # Add lagged features
        for lag in [1, 2, 3, 5, 10]:
            expected_features.append(f'close_lag_{lag}')
        
        # Add rolling statistics
        for window in [5, 10, 20]:
            expected_features.extend([f'close_std_{window}', f'close_min_{window}', f'close_max_{window}'])
        
        # Add missing features with default values
        for feature in expected_features:
            if feature not in df.columns:
                if feature in ['day_of_week', 'month', 'quarter']:
                    df[feature] = 0  # Default for time features
                elif 'ratio' in feature or 'position' in feature:
                    df[feature] = 1.0  # Default for ratio features
                else:
                    df[feature] = 0.0  # Default for other features
        
        # Remove any extra features that aren't in the expected set
        columns_to_keep = ['date', 'open', 'high', 'low', 'close'] + expected_features
        df = df[[col for col in columns_to_keep if col in df.columns]]
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def prepare_training_data(self, df: pd.DataFrame, target_column: str = 'close', for_prediction: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for machine learning training.
        
        Args:
            df: DataFrame with features
            target_column: Column to use as target variable
            for_prediction: If True, requires less data (only enough for features)
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        # For prediction with pre-trained models, we only need enough data to create features
        # For training new models, we need MIN_TRAINING_DAYS
        min_required_days = self.config.LOOKBACK_DAYS + 30 if for_prediction else self.config.MIN_TRAINING_DAYS
        
        if df is None or len(df) < min_required_days:
            logger.warning(f"Insufficient data: {len(df) if df is not None else 0} days (need {min_required_days})")
            return np.array([]), np.array([])
        
        try:
            # Ensure consistent feature set before preparing training data
            df = self._standardize_features(df)
            
            # Select feature columns (exclude target and metadata columns)
            exclude_columns = [
                'date', 'data_source', target_column, 'adjusted_close', 'currency', 'volume'
            ]
            
            feature_columns = [col for col in df.columns if col not in exclude_columns]
            
            # Remove any remaining non-numeric columns
            numeric_columns = []
            for col in feature_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_columns.append(col)
                else:
                    logger.debug(f"Skipping non-numeric column: {col}")
            
            if not numeric_columns:
                logger.error("No numeric feature columns found")
                return np.array([]), np.array([])
            
            # Create feature matrix
            X = df[numeric_columns].values
            
            # Create target vector as PERCENTAGE CHANGE (not raw price)
            # This normalizes across different price ranges and is standard in finance
            current_price = df[target_column].values
            next_price = df[target_column].shift(-1).values
            
            # Calculate percentage change: (next_price - current_price) / current_price * 100
            y = ((next_price - current_price) / current_price) * 100
            
            # Remove last row (no target for prediction)
            X = X[:-1]
            y = y[:-1]
            
            # Handle infinity and NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            # For percentage change, typical range is -10% to +10% per day
            # Clip extreme values to -50% to +50% (rare but possible)
            y = np.clip(y, -50, 50)
            y = np.nan_to_num(y, nan=0.0, posinf=50.0, neginf=-50.0)
            
            # Remove any rows with NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) == 0:
                logger.error("No valid training data after preprocessing")
                return np.array([]), np.array([])
            
            logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return np.array([]), np.array([])
    
    def get_stock_symbols(self, category: str) -> List[str]:
        """
        Get list of stock symbols from index file.
        
        Args:
            category: Stock category ('us_stocks' or 'ind_stocks')
            
        Returns:
            List of stock symbols
        """
        try:
            if category == 'us_stocks':
                index_file = self.config.US_INDEX_FILE
            elif category == 'ind_stocks':
                index_file = self.config.IND_INDEX_FILE
            else:
                logger.error(f"Invalid category: {category}")
                return []
            
            if not os.path.exists(index_file):
                logger.error(f"Index file not found: {index_file}")
                return []
            
            df = pd.read_csv(index_file)
            
            if 'symbol' not in df.columns:
                logger.error(f"'symbol' column not found in {index_file}")
                return []
            
            symbols = df['symbol'].tolist()
            logger.info(f"Found {len(symbols)} symbols in {category}")
            return symbols
            
        except Exception as e:
            logger.error(f"Error loading symbols for {category}: {str(e)}")
            return []
    
    def validate_data_quality(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Validate data quality for prediction.
        
        Args:
            df: DataFrame to validate
            symbol: Stock symbol for logging
            
        Returns:
            True if data quality is acceptable
        """
        if df is None or len(df) == 0:
            logger.warning(f"No data to validate for {symbol}")
            return False
        
        try:
            # Check minimum data points
            if len(df) < self.config.MIN_TRAINING_DAYS:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} days")
                return False
            
            # Check for missing values in essential columns
            essential_columns = ['date', 'open', 'high', 'low', 'close']
            missing_data = df[essential_columns].isnull().sum()
            
            if missing_data.any():
                logger.warning(f"Missing data in {symbol}: {missing_data.to_dict()}")
                return False
            
            # Check for reasonable price ranges
            if (df['close'] < self.config.MIN_PRICE).any() or (df['close'] > self.config.MAX_PRICE).any():
                logger.warning(f"Price out of range for {symbol}")
                return False
            
            # Check for extreme volatility
            price_changes = df['close'].pct_change().abs()
            if (price_changes > self.config.MAX_VOLATILITY).any():
                logger.warning(f"Extreme volatility detected for {symbol}")
                return False
            
            # Check date continuity
            date_gaps = df['date'].diff().dt.days
            if (date_gaps > 7).any():  # More than 7 days gap
                logger.warning(f"Large date gaps detected for {symbol}")
                # Don't return False for this, just warn
            
            logger.debug(f"Data quality validation passed for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating data for {symbol}: {str(e)}")
            return False
    
    def clear_cache(self):
        """Clear the data cache."""
        self.cache.clear()
        logger.debug("Data cache cleared")
