"""
Shared Module

This module contains shared utilities and common functionality used across
different modules in the backend. It includes:
- Configuration management
- Logging utilities
- Data validation helpers
- Common data structures
- Utility functions
- Constants and enums

This module ensures consistency and reduces code duplication across the project.
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json

# Global in-memory cache for stock categorization
_CATEGORIZATION_CACHE: Dict[str, str] = {}

# Configuration Management
class Config:
    """
    Centralized configuration management
    """
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data')
        self.permanent_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'permanent')
        self.cache_duration = int(os.getenv('CACHE_DURATION', 60))
        self.min_request_delay = float(os.getenv('MIN_REQUEST_DELAY', 2.0))
        self.port = int(os.getenv('PORT', 5000))
        
        # API Keys
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        self.upstox_api_key = os.getenv('UPSTOX_API_KEY')
        
        # Upstox OAuth Credentials (Static - from .env)
        self.upstox_client_id = os.getenv('UPSTOX_CLIENT_ID')
        self.upstox_client_secret = os.getenv('UPSTOX_CLIENT_SECRET')
        self.upstox_redirect_uri = os.getenv('UPSTOX_REDIRECT_URI', 'http://localhost:3000')
    
    def get_data_path(self, *paths):
        """Get path relative to data directory"""
        return os.path.join(self.data_dir, *paths)
    
    def get_permanent_path(self, *paths):
        """Get path relative to permanent directory"""
        return os.path.join(self.permanent_dir, *paths)

# Logging Utilities
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with consistent formatting
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Data Validation
class DataValidator:
    """
    Common data validation utilities
    """
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Validate stock symbol format"""
        if not symbol or not isinstance(symbol, str):
            return False
        return len(symbol.strip()) > 0 and len(symbol.strip()) <= 10
    
    @staticmethod
    def validate_price(price: Any) -> bool:
        """Validate price value"""
        try:
            price_float = float(price)
            return price_float > 0
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_date_format(date_str: str) -> bool:
        """Validate date format (YYYY-MM-DD)"""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False

# Common Data Structures
class StockData:
    """
    Standardized stock data structure
    """
    def __init__(self, symbol: str, price: float, timestamp: str, 
                 source: str, company_name: str = None):
        self.symbol = symbol.upper()
        self.price = float(price)
        self.timestamp = timestamp
        self.source = source
        self.company_name = company_name or symbol
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'price': self.price,
            'timestamp': self.timestamp,
            'source': self.source,
            'company_name': self.company_name
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())

class PredictionResult:
    """
    Standardized prediction result structure
    """
    def __init__(self, symbol: str, predicted_price: float, 
                 confidence: float, algorithm: str, timestamp: str):
        self.symbol = symbol.upper()
        self.predicted_price = float(predicted_price)
        self.confidence = float(confidence)
        self.algorithm = algorithm
        self.timestamp = timestamp
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'predicted_price': self.predicted_price,
            'confidence': self.confidence,
            'algorithm': self.algorithm,
            'timestamp': self.timestamp
        }

# Utility Functions
def get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()

def categorize_stock(symbol: str) -> str:
    """
    Categorize stock using prioritized resolution hierarchy:
    1. Explicit exchange suffixes (.NS, .BO, .US)
    2. In-memory categorization cache
    3. Local master index files (O(1) disk lookup)
    4. Pattern-based rule matching
    5. Fallback to YFinance scraping (cached)
    """
    if not symbol or not isinstance(symbol, str):
        return 'us_stocks'
        
    symbol_upper = symbol.upper().strip()
    if not symbol_upper:
        return 'us_stocks'
        
    base_symbol = symbol_upper.split('.')[0]
    
    # 1. Check explicit exchange suffixes (highest priority)
    indian_suffixes = ['.NS', '.BO']
    for suffix in indian_suffixes:
        if symbol_upper.endswith(suffix):
            return 'ind_stocks'
    
    us_suffixes = ['.US']
    for suffix in us_suffixes:
        if symbol_upper.endswith(suffix):
            return 'us_stocks'
            
    # 2. Check in-memory cache
    if base_symbol in _CATEGORIZATION_CACHE:
        return _CATEGORIZATION_CACHE[base_symbol]
    
    # 3. Check local index files first (instant disk lookup before network I/O)
    try:
        current_dir = os.path.dirname(os.path.dirname(__file__))
        indian_index = os.path.join(current_dir, '..', 'permanent', 'ind_stocks', 'index_ind_stocks.csv')
        us_index = os.path.join(current_dir, '..', 'permanent', 'us_stocks', 'index_us_stocks.csv')
        
        # Check Indian index
        if os.path.exists(indian_index):
            try:
                import pandas as pd
                df = pd.read_csv(indian_index)
                if not df.empty and 'symbol' in df.columns and base_symbol in df['symbol'].dropna().astype(str).str.upper().values:
                    _CATEGORIZATION_CACHE[base_symbol] = 'ind_stocks'
                    return 'ind_stocks'
            except Exception:
                pass
                
        # Check US index
        if os.path.exists(us_index):
            try:
                import pandas as pd
                df = pd.read_csv(us_index)
                if not df.empty and 'symbol' in df.columns and base_symbol in df['symbol'].dropna().astype(str).str.upper().values:
                    _CATEGORIZATION_CACHE[base_symbol] = 'us_stocks'
                    return 'us_stocks'
            except Exception:
                pass
    except Exception:
        pass
        
    # 4. Pattern matching analysis for known Indian company names/symbols
    if len(base_symbol) >= 3:
        indian_patterns = [
            'RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICI', 'WIPRO', 'MARUTI', 
            'BAJAJ', 'TATA', 'ADANI', 'LT', 'ITC', 'BHARTI', 'ONGC', 'SBI',
            'NTPC', 'POWER', 'COAL', 'GAIL', 'BPCL', 'HPCL', 'IOC', 'BANK'
        ]
        for pattern in indian_patterns:
            if pattern in base_symbol:
                _CATEGORIZATION_CACHE[base_symbol] = 'ind_stocks'
                return 'ind_stocks'

    # 5. YFinance remote scraping fallback (only reached for unknown symbols)
    try:
        import yfinance as yf
        indian_symbol = f"{base_symbol}.NS"
        ticker = yf.Ticker(indian_symbol)
        info = ticker.info
        
        if info and 'symbol' in info and info.get('symbol'):
            exchange = info.get('exchange', '').upper()
            country = info.get('country', '').upper()
            indian_exchanges = ['NSE', 'NSI', 'BSE', 'BOM']
            if any(ex in exchange for ex in indian_exchanges) or country == 'INDIA':
                _CATEGORIZATION_CACHE[base_symbol] = 'ind_stocks'
                return 'ind_stocks'
        
        ticker_us = yf.Ticker(base_symbol)
        info_us = ticker_us.info
        if info_us and 'symbol' in info_us and info_us.get('symbol'):
            exchange = info_us.get('exchange', '').upper()
            country = info_us.get('country', '').upper()
            us_exchanges = ['NASDAQ', 'NYSE', 'NYSEARCA', 'BATS', 'AMEX']
            if any(ex in exchange for ex in us_exchanges) or country == 'UNITED STATES':
                _CATEGORIZATION_CACHE[base_symbol] = 'us_stocks'
                return 'us_stocks'
    except Exception:
        pass
        
    # 6. Fallback pattern matching
    import re
    if re.match(r'^[A-Z]{1,5}$', base_symbol) and '.' not in symbol_upper:
        category = 'us_stocks'
    else:
        category = 'us_stocks'
        
    _CATEGORIZATION_CACHE[base_symbol] = category
    return category

def validate_and_categorize_stock(symbol: str) -> str:
    """
    Validate stock symbol and categorize it efficiently.
    Uses categorize_stock directly to leverage the prioritized resolution hierarchy.
    """
    return categorize_stock(symbol)

def format_price(price: float, currency: str = 'USD') -> str:
    """Format price with currency symbol"""
    if currency == 'USD':
        return f"${price:.2f}"
    elif currency == 'INR':
        return f"₹{price:.2f}"
    else:
        return f"{price:.2f} {currency}"

def ensure_alphabetical_order(df, column: str = 'symbol'):
    """
    Ensure DataFrame is sorted alphabetically by specified column.
    
    Args:
        df: pandas DataFrame
        column: Column name to sort by (default: 'symbol')
    
    Returns:
        Sorted DataFrame
    """
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame) and column in df.columns:
            return df.sort_values(by=column).reset_index(drop=True)
    except ImportError:
        pass
    return df

def standardize_csv_columns(df):
    """
    Standardize DataFrame columns to lowercase format.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        DataFrame with standardized lowercase columns
    """
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            column_mapping = {
                'Date': 'date',
                'Open': 'open', 
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adjusted_close',
                'Adj_Close': 'adjusted_close',
                'Symbol': 'symbol',
                'Company Name': 'company_name',
                'Company_Name': 'company_name',
                'Sector': 'sector',
                'Market Cap': 'market_cap',
                'Market_Cap': 'market_cap',
                'Headquarters': 'headquarters',
                'Exchange': 'exchange'
            }
            
            df = df.rename(columns=column_mapping)
            df.columns = df.columns.str.lower()
            return df
    except ImportError:
        pass
    return df

def get_currency_for_category(category: str) -> str:
    """
    Get currency code for stock category.
    
    Args:
        category: Stock category ('us_stocks', 'ind_stocks', etc.)
    
    Returns:
        Currency code ('USD', 'INR', etc.)
    """
    if category == 'us_stocks':
        return 'USD'
    elif category == 'ind_stocks':
        return 'INR'
    else:
        return 'USD'

def get_live_exchange_rate() -> float:
    """
    Get live USD-INR exchange rate using currency converter.
    
    Returns:
        float: USD to INR exchange rate
    """
    try:
        from .currency_converter import get_live_exchange_rate
        return get_live_exchange_rate()
    except ImportError:
        return 83.5

def convert_usd_to_inr(usd_amount: float) -> float:
    """
    Convert USD amount to INR using live exchange rate.
    
    Args:
        usd_amount: Amount in USD
        
    Returns:
        float: Amount in INR
    """
    try:
        from .currency_converter import convert_usd_to_inr
        return convert_usd_to_inr(usd_amount)
    except ImportError:
        return usd_amount * 83.5

def convert_inr_to_usd(inr_amount: float) -> float:
    """
    Convert INR amount to USD using live exchange rate.
    
    Args:
        inr_amount: Amount in INR
        
    Returns:
        float: Amount in USD
    """
    try:
        from .currency_converter import convert_inr_to_usd
        return convert_inr_to_usd(inr_amount)
    except ImportError:
        return inr_amount / 83.5

# Constants
class Constants:
    """Application constants"""
    
    US_STOCKS = 'us_stocks'
    INDIAN_STOCKS = 'ind_stocks'
    
    YFINANCE = 'yfinance'
    FINNHUB = 'finnhub'
    PERMANENT_DIR = 'permanent_directory'
    
    CSV_EXTENSION = '.csv'
    JSON_EXTENSION = '.json'
    
    DEFAULT_CACHE_DURATION = 60
    DEFAULT_REQUEST_DELAY = 2.0
    DEFAULT_PORT = 5000
    MAX_SYMBOL_LENGTH = 10
    MAX_SEARCH_RESULTS = 20
    EXCHANGE_RATE_CACHE_DURATION = 3600
    
    HISTORICAL_START = "2020-01-01"
    HISTORICAL_END = "2024-12-31"
    LATEST_START = "2025-01-01"
    
    REQUIRED_STOCK_COLUMNS = [
        'date', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close', 'currency'
    ]
    
    REQUIRED_INDEX_COLUMNS = [
        'symbol', 'company_name', 'sector', 'market_cap', 'headquarters', 'exchange'
    ]

# Error Classes
class StockDataError(Exception):
    """Base exception for stock data related errors"""
    pass

class InvalidSymbolError(StockDataError):
    """Exception for invalid stock symbols"""
    pass

class DataFetchError(StockDataError):
    """Exception for data fetching errors"""
    pass

class PredictionError(StockDataError):
    """Exception for prediction related errors"""
    pass

__all__ = [
    'Config',
    'setup_logger',
    'DataValidator',
    'StockData',
    'PredictionResult',
    'get_current_timestamp',
    'categorize_stock',
    'validate_and_categorize_stock',
    'format_price',
    'ensure_alphabetical_order',
    'standardize_csv_columns',
    'get_currency_for_category',
    'get_live_exchange_rate',
    'convert_usd_to_inr',
    'convert_inr_to_usd',
    'Constants',
    'StockDataError',
    'InvalidSymbolError',
    'DataFetchError',
    'PredictionError'
]
