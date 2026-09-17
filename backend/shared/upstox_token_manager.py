"""
Upstox Token Manager

Handles automatic refresh of Upstox OAuth2 access tokens using refresh tokens.
Uses JSON cache for dynamic tokens and .env for static credentials with thread-safe operations.
Supports both proactive (before expiry) and reactive (on 401 errors) refresh strategies.
"""

import os
import json
import time
import threading
import requests
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class UpstoxTokenManager:
    """
    Manages Upstox OAuth2 tokens with automatic refresh capabilities.
    
    Features:
    - JSON cache for dynamic tokens (access_token, refresh_token, expiry)
    - .env for static credentials (client_id, client_secret, redirect_uri)
    - Thread-safe token updates using reentrant lock (RLock)
    - Proactive refresh (15 minutes before expiry)
    - Reactive refresh (on 401 errors)
    - Comprehensive error handling
    """
    
    def __init__(self, env_file_path: str = None):
        """
        Initialize token manager.
        
        Args:
            env_file_path: Path to .env file (default: backend/.env)
        """
        if env_file_path is None:
            # Default to backend/.env
            self.env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
            # Also check parent directory for .env
            if not os.path.exists(self.env_file):
                parent_env = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
                if os.path.exists(parent_env):
                    self.env_file = parent_env
        else:
            self.env_file = env_file_path
            
        # Cache file path
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '_cache')
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, 'upstox_tokens.json')
        
        # Reentrant thread lock for safe token updates across nested method calls
        self._lock = threading.RLock()
        
        # Upstox OAuth endpoints (v2)
        self.refresh_url = "https://api.upstox.com/v2/login/authorization/token"
        
        # Load initial tokens
        self._load_tokens()
        
        # Proactive refresh threshold (15 minutes before expiry)
        self.refresh_threshold_minutes = 15
        
    def _load_tokens(self) -> None:
        """Load static credentials from .env and dynamic tokens from JSON cache."""
        # Load static credentials from .env
        try:
            if os.path.exists(self.env_file):
                from dotenv import load_dotenv
                load_dotenv(self.env_file)
                
                self.client_id = os.getenv('UPSTOX_CLIENT_ID')
                self.client_secret = os.getenv('UPSTOX_CLIENT_SECRET')
                self.redirect_uri = os.getenv('UPSTOX_REDIRECT_URI', 'http://localhost:3000')
                
                logger.info("✓ Loaded static credentials from .env file")
            else:
                logger.warning("No .env file found")
                self.client_id = None
                self.client_secret = None
                self.redirect_uri = 'http://localhost:3000'
                
        except Exception as e:
            logger.warning(f"Failed to load credentials from .env: {e}")
            self.client_id = None
            self.client_secret = None
            self.redirect_uri = 'http://localhost:3000'
        
        # Load dynamic tokens from JSON cache only
        self._load_tokens_from_cache()
    
    def _load_tokens_from_cache(self) -> None:
        """Load dynamic tokens from JSON cache file, with .env fallback for migration."""
        # First try JSON cache
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                self.access_token = cache_data.get('access_token')
                self.refresh_token = cache_data.get('refresh_token')
                
                # Parse token expiry
                expiry_str = cache_data.get('token_expiry')
                self.token_expiry = None
                if expiry_str and expiry_str != 'will_be_generated_by_oauth':
                    try:
                        self.token_expiry = datetime.fromisoformat(expiry_str)
                    except ValueError:
                        logger.warning(f"Invalid token expiry format in cache: {expiry_str}")
                
                logger.info("✓ Loaded dynamic tokens from JSON cache")
                return
                
        except Exception as e:
            logger.warning(f"Failed to load tokens from cache: {e}")
        
        # Fallback to .env for migration (temporary)
        try:
            from dotenv import load_dotenv
            load_dotenv(self.env_file)
            
            access_token = os.getenv('UPSTOX_ACCESS_TOKEN')
            refresh_token = os.getenv('UPSTOX_REFRESH_TOKEN')
            expiry_str = os.getenv('UPSTOX_TOKEN_EXPIRY')
            
            if access_token and access_token != 'will_be_generated_by_oauth':
                self.access_token = access_token
                self.refresh_token = refresh_token if refresh_token != 'will_be_generated_by_oauth' else None
                
                # Parse token expiry
                self.token_expiry = None
                if expiry_str and expiry_str != 'will_be_generated_by_oauth':
                    try:
                        self.token_expiry = datetime.fromisoformat(expiry_str)
                    except ValueError:
                        logger.warning(f"Invalid token expiry format in .env: {expiry_str}")
                
                logger.info("✓ Loaded dynamic tokens from .env (migration fallback)")
                
                # Migrate to JSON cache
                if self.access_token:
                    self._migrate_to_json_cache()
                
        except Exception as e:
            logger.warning(f"Failed to load tokens from .env fallback: {e}")
    
    def _migrate_to_json_cache(self) -> None:
        """Migrate tokens from .env to JSON cache."""
        try:
            # Set expiry to next day 3:30 AM if not set
            if not self.token_expiry:
                now = datetime.now()
                tomorrow_330am = (now + timedelta(days=1)).replace(hour=3, minute=30, second=0, microsecond=0)
                self.token_expiry = tomorrow_330am
                expires_in_seconds = int((tomorrow_330am - now).total_seconds())
            else:
                expires_in_seconds = 3600  # Default 1 hour
            
            # Save to JSON cache
            self._save_tokens(self.access_token, self.refresh_token or '', expires_in_seconds)
            logger.info("✓ Migrated tokens from .env to JSON cache")
            
        except Exception as e:
            logger.warning(f"Failed to migrate tokens to JSON cache: {e}")
        
        # If we reach here, no tokens were found
        if not self.access_token:
            logger.warning("No Upstox tokens found in .env or cache")
            self.access_token = None
            self.refresh_token = None
            self.token_expiry = None
    
    def _save_tokens(self, access_token: str, refresh_token: str, expires_in: int) -> None:
        """
        Save tokens to JSON cache file only.
        
        Args:
            access_token: New access token
            refresh_token: New refresh token (may be same as old)
            expires_in: Token expiry time in seconds
        """
        with self._lock:
            try:
                # Calculate new expiry time
                new_expiry = datetime.now() + timedelta(seconds=expires_in)
                self.access_token = access_token
                self.refresh_token = refresh_token
                self.token_expiry = new_expiry
                
                # Update cache data
                cache_data = {
                    'client_id': self.client_id,
                    'client_secret': self.client_secret,
                    'access_token': access_token,
                    'refresh_token': refresh_token,
                    'redirect_uri': self.redirect_uri,
                    'token_expiry': new_expiry.isoformat(),
                    'last_updated': datetime.now().isoformat()
                }
                
                # Save to cache file only
                with open(self.cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                
                # Set restrictive permissions on cache file (Unix systems)
                try:
                    os.chmod(self.cache_file, 0o600)
                except OSError:
                    pass  # Ignore on Windows
                
                logger.info("✓ Tokens saved to JSON cache")
                
            except Exception as e:
                logger.error(f"Failed to save tokens: {e}")
                raise
    
    
    def is_token_expired(self) -> bool:
        """
        Check if token is expired or near expiry.
        
        Upstox tokens expire daily at 3:30 AM IST, so we check if:
        1. Token expiry time is set and has passed, OR
        2. Current time is past 3:30 AM and we don't have a fresh token
        
        Returns:
            True if token needs refresh (expired or expires within threshold)
        """
        if not self.token_expiry:
            return True  # No expiry info, assume needs refresh
        
        # Check if expires within threshold
        threshold_time = datetime.now() + timedelta(minutes=self.refresh_threshold_minutes)
        is_near_expiry = self.token_expiry <= threshold_time
        
        # Additional check: if it's past 3:30 AM today and we don't have a fresh token
        # (Upstox tokens expire daily at 3:30 AM IST)
        now = datetime.now()
        today_330am = now.replace(hour=3, minute=30, second=0, microsecond=0)
        is_past_daily_expiry = now >= today_330am and self.token_expiry < today_330am
        
        return is_near_expiry or is_past_daily_expiry
    
    def needs_daily_token_refresh(self) -> bool:
        """
        Check if we need a new token due to Upstox's daily 3:30 AM expiration.
        
        Returns:
            True if current time is past 3:30 AM and token is from before today's 3:30 AM
        """
        if not self.token_expiry:
            return True
        
        now = datetime.now()
        today_330am = now.replace(hour=3, minute=30, second=0, microsecond=0)
        return now >= today_330am and self.token_expiry < today_330am
    
    def refresh_access_token(self) -> bool:
        """
        Refresh access token using refresh token.
        
        Returns:
            True if refresh successful, False otherwise
        """
        if not self.refresh_token or not self.client_id or not self.client_secret:
            logger.error("Missing refresh token or OAuth credentials")
            return False
        
        try:
            logger.info("Refreshing Upstox access token...")
            
            # Prepare refresh request
            data = {
                'grant_type': 'refresh_token',
                'refresh_token': self.refresh_token,
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'redirect_uri': self.redirect_uri
            }
            
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            # Make refresh request
            response = requests.post(
                self.refresh_url,
                data=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                token_data = response.json()
            elif response.status_code == 299:
                logger.warning(f"Upstox API returned deprecation warning: {response.text}")
                logger.error("Upstox API v1 is deprecated. Token refresh failed.")
                logger.error("Please migrate to Upstox API v2 or use existing token if still valid.")
                return False
            else:
                logger.error(f"Token refresh failed: HTTP {response.status_code}: {response.text}")
                return False
            
            # Extract new tokens
            logger.info(f"Token data type: {type(token_data)}")
            logger.info(f"Token data content: {token_data}")
            
            if isinstance(token_data, str):
                logger.error("Token data is a string, not a dictionary. Cannot extract tokens.")
                return False
                
            new_access_token = token_data.get('access_token')
            new_refresh_token = token_data.get('refresh_token', self.refresh_token)
            expires_in = token_data.get('expires_in', 3600)  # Default 1 hour
            
            if new_access_token:
                # Save new tokens
                self._save_tokens(new_access_token, new_refresh_token, expires_in)
                logger.info("✓ Access token refreshed successfully")
                return True
            else:
                logger.error("No access token in refresh response")
                return False
                
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return False
    
    def get_valid_token(self) -> Optional[str]:
        """
        Get a valid access token, refreshing if necessary.
        
        Returns:
            Valid access token or None if refresh fails
        """
        with self._lock:
            # Check if proactive refresh needed
            if self.is_token_expired():
                logger.info("Token expired or near expiry, refreshing proactively...")
                if not self.refresh_access_token():
                    logger.error("Proactive token refresh failed")
                    # Even if refresh failed, return the existing token if we have one
                    # This allows the system to work with potentially expired tokens
                    # until the user migrates to API v2
                    if self.access_token:
                        logger.warning("Using existing token despite refresh failure (API v1 deprecated)")
                        return self.access_token
                    return None
            
            return self.access_token
    
    def handle_401_error(self) -> bool:
        """
        Handle 401 Unauthorized error by refreshing token.
        
        Returns:
            True if token was refreshed successfully, False otherwise
        """
        logger.warning("Received 401 error, attempting token refresh...")
        return self.refresh_access_token()
    
    def set_oauth_credentials(self, client_id: str, client_secret: str, redirect_uri: str = None) -> None:
        """
        Set OAuth credentials (for initial setup).
        
        Args:
            client_id: Upstox client ID
            client_secret: Upstox client secret
            redirect_uri: OAuth redirect URI (default: http://localhost:8080)
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri or 'http://localhost:8080'
        
        # Update cache
        try:
            cache_data = {
                'client_id': client_id,
                'client_secret': client_secret,
                'redirect_uri': self.redirect_uri,
                'last_updated': datetime.now().isoformat()
            }
            
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    existing_data = json.load(f)
                cache_data.update(existing_data)
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            logger.info("✓ OAuth credentials saved to cache")
            
        except Exception as e:
            logger.warning(f"Failed to save OAuth credentials to cache: {e}")
    
    def has_valid_credentials(self) -> bool:
        """
        Check if we have all required credentials for token operations.
        
        Returns:
            True if all credentials are available
        """
        return bool(
            self.client_id and 
            self.client_secret and 
            self.redirect_uri
        )
    
    def get_token_info(self) -> Dict:
        """
        Get current token information for debugging.
        
        Returns:
            Dictionary with token status information
        """
        return {
            'has_access_token': bool(self.access_token),
            'has_refresh_token': bool(self.refresh_token),
            'has_client_credentials': bool(self.client_id and self.client_secret),
            'token_expiry': self.token_expiry.isoformat() if self.token_expiry else None,
            'is_expired': self.is_token_expired(),
            'minutes_until_expiry': (
                (self.token_expiry - datetime.now()).total_seconds() / 60
                if self.token_expiry and self.token_expiry > datetime.now()
                else 0
            )
        }
