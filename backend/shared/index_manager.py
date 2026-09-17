"""
Dynamic Index Manager
Manages the master dynamic index files for stock metadata
"""

import os
import pandas as pd
import threading
from typing import Dict, Optional, List

class DynamicIndexManager:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.us_index_path = os.path.join(data_dir, 'index_us_stocks_dynamic.csv')
        self.ind_index_path = os.path.join(data_dir, 'index_ind_stocks_dynamic.csv')
        self._lock = threading.Lock()
    
    def get_index_path(self, category: str) -> str:
        """Get path to dynamic index for category"""
        if category == 'us_stocks':
            return self.us_index_path
        elif category == 'ind_stocks':
            return self.ind_index_path
        else:
            raise ValueError(f"Unknown category: {category}")

    def _read_csv_safely(self, category: str) -> pd.DataFrame:
        """Safely read index CSV with automatic corruption recovery."""
        index_path = self.get_index_path(category)
        if not os.path.exists(index_path):
            self._initialize_from_permanent_unlocked(category)
            
        if not os.path.exists(index_path):
            return pd.DataFrame(columns=['symbol', 'company_name', 'sector', 
                                      'market_cap', 'headquarters', 'exchange', 'currency', 'isin'])
        
        try:
            return pd.read_csv(index_path)
        except Exception:
            # Auto-recover from permanent index if file is corrupt or truncated
            if self._initialize_from_permanent_unlocked(category):
                try:
                    return pd.read_csv(index_path)
                except Exception:
                    pass
            return pd.DataFrame(columns=['symbol', 'company_name', 'sector', 
                                      'market_cap', 'headquarters', 'exchange', 'currency', 'isin'])
    
    def stock_exists(self, symbol: str, category: str) -> bool:
        """Check if stock exists in dynamic index"""
        with self._lock:
            df = self._read_csv_safely(category)
            if df.empty or 'symbol' not in df.columns:
                return False
            return symbol.upper() in df['symbol'].dropna().astype(str).str.upper().values
    
    def get_stock_info(self, symbol: str, category: str) -> Optional[Dict]:
        """Get stock metadata from dynamic index"""
        with self._lock:
            df = self._read_csv_safely(category)
            if df.empty or 'symbol' not in df.columns:
                return None
            row = df[df['symbol'].dropna().astype(str).str.upper() == symbol.upper()]
            if row.empty:
                return None
            return row.iloc[0].to_dict()
    
    def add_stock(self, symbol: str, stock_info: Dict, category: str):
        """Add new stock to dynamic index (alphabetically sorted)"""
        with self._lock:
            index_path = self.get_index_path(category)
            df = self._read_csv_safely(category)
            
            # Ensure ISIN column exists for backward compatibility
            if 'isin' not in df.columns:
                df['isin'] = ''
            df['isin'] = df['isin'].astype(str)
            
            # Check if already exists
            if 'symbol' in df.columns and symbol.upper() in df['symbol'].dropna().astype(str).str.upper().values:
                print(f"Stock {symbol} already exists in {category} index")
                return
            
            # Add new stock
            new_row = {
                'symbol': symbol.upper(),
                'company_name': stock_info.get('company_name', symbol),
                'sector': stock_info.get('sector', 'N/A'),
                'market_cap': stock_info.get('market_cap', ''),
                'headquarters': stock_info.get('headquarters', 'N/A'),
                'exchange': stock_info.get('exchange', 'N/A'),
                'currency': 'INR' if category == 'ind_stocks' else 'USD',
                'isin': stock_info.get('isin', '')
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Sort alphabetically by symbol
            df = df.sort_values('symbol').reset_index(drop=True)
            
            # Validate no deletion before saving
            self._validate_no_deletion_unlocked(category, df)
            
            # Save safely
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            df.to_csv(index_path, index=False)
            print(f"Added {symbol} to {category} dynamic index")
    
    def get_all_symbols(self, category: str) -> list:
        """Get all symbols from dynamic index"""
        with self._lock:
            df = self._read_csv_safely(category)
            if df.empty or 'symbol' not in df.columns:
                return []
            return df['symbol'].dropna().tolist()
    
    def get_isin(self, symbol: str, category: str) -> Optional[str]:
        """Get ISIN for a stock from dynamic index"""
        with self._lock:
            df = self._read_csv_safely(category)
            if df.empty or 'isin' not in df.columns or 'symbol' not in df.columns:
                return None
            row = df[df['symbol'].dropna().astype(str).str.upper() == symbol.upper()]
            if row.empty:
                return None
            isin = row.iloc[0]['isin']
            return str(isin) if pd.notna(isin) and str(isin).strip() else None
    
    def _validate_no_deletion_unlocked(self, category: str, new_df: pd.DataFrame):
        """Internal validation helper (expects lock to be held)."""
        index_path = self.get_index_path(category)
        if os.path.exists(index_path):
            try:
                old_df = pd.read_csv(index_path)
                old_count = len(old_df)
                new_count = len(new_df)
                
                if new_count < old_count:
                    print(f"⚠️ SAFETY CHECK: Attempting to reduce {category} from {old_count} to {new_count} stocks!")
                    print(f"   Attempting auto-restore from permanent index...")
                    
                    if self._initialize_from_permanent_unlocked(category):
                        restored_df = pd.read_csv(index_path)
                        if len(restored_df) >= old_count:
                            return
                    
                    raise ValueError(
                        f"SAFETY CHECK FAILED: Attempting to reduce {category} from "
                        f"{old_count} to {new_count} stocks. This would delete stocks!"
                    )
            except Exception as e:
                if isinstance(e, ValueError):
                    raise e
    
    def validate_no_deletion(self, category: str, new_df: pd.DataFrame):
        """Ensure we're not deleting stocks from dynamic index"""
        with self._lock:
            self._validate_no_deletion_unlocked(category, new_df)
    
    def update_stock_isin(self, symbol: str, isin: str, category: str):
        """Update ISIN for existing stock"""
        with self._lock:
            index_path = self.get_index_path(category)
            df = self._read_csv_safely(category)
            if df.empty or 'symbol' not in df.columns:
                print(f"Stock {symbol} not found in {category} index")
                return
            
            if 'isin' not in df.columns:
                df['isin'] = ''
            df['isin'] = df['isin'].astype(str)
            
            mask = df['symbol'].dropna().astype(str).str.upper() == symbol.upper()
            if mask.any():
                df.loc[mask, 'isin'] = isin
                self._validate_no_deletion_unlocked(category, df)
                df.to_csv(index_path, index=False)
                print(f"Updated ISIN for {symbol}: {isin}")
            else:
                print(f"Stock {symbol} not found in {category} index")
    
    def get_stock_count(self, category: str) -> int:
        """Get the number of stocks in the dynamic index"""
        with self._lock:
            df = self._read_csv_safely(category)
            return len(df)
    
    def _initialize_from_permanent_unlocked(self, category: str) -> bool:
        """Internal helper to initialize dynamic index without acquiring lock again."""
        index_path = self.get_index_path(category)
        
        if category == 'us_stocks':
            permanent_path = os.path.join(os.path.dirname(os.path.dirname(self.data_dir)), 
                                        'permanent', 'us_stocks', 'index_us_stocks.csv')
            currency = 'USD'
        elif category == 'ind_stocks':
            permanent_path = os.path.join(os.path.dirname(os.path.dirname(self.data_dir)), 
                                        'permanent', 'ind_stocks', 'index_ind_stocks.csv')
            currency = 'INR'
        else:
            print(f"Unknown category: {category}")
            return False
        
        if not os.path.exists(permanent_path):
            print(f"Permanent {category} index not found: {permanent_path}")
            return False
        
        try:
            df_permanent = pd.read_csv(permanent_path)
            required_cols = ['symbol', 'company_name', 'sector', 'market_cap', 'headquarters', 'exchange']
            for col in required_cols:
                if col not in df_permanent.columns:
                    return False
            
            df_permanent['currency'] = currency
            df_permanent['isin'] = ''
            
            dynamic_columns = ['symbol', 'company_name', 'sector', 'market_cap', 'headquarters', 'exchange', 'currency', 'isin']
            df_dynamic = df_permanent[dynamic_columns]
            df_dynamic = df_dynamic.sort_values('symbol').reset_index(drop=True)
            
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            df_dynamic.to_csv(index_path, index=False)
            print(f"✅ Successfully initialized {category} dynamic index with {len(df_dynamic)} stocks")
            return True
        except Exception as e:
            print(f"Error initializing {category} from permanent: {e}")
            return False
    
    def initialize_from_permanent(self, category: str) -> bool:
        """
        Initialize dynamic index from permanent index if dynamic is empty or corrupt.
        """
        with self._lock:
            return self._initialize_from_permanent_unlocked(category)
