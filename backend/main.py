"""
Main Coordinator Module

This is the central coordinator for the stock prediction system.
It orchestrates all components:
- Live data fetching (live-data module)
- Company information (company-info module) 
- Algorithm selection and execution (algorithms module)
- Prediction generation (prediction module)
- Shared utilities (shared module)

The main.py serves as the entry point and API coordinator.
"""

import os
import logging
import pandas as pd
import json
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import yfinance as yf

# Import modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_fetching'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_fetching', 'ind_stocks', 'current_fetching'))
from data_manager import LiveFetcher
from ind_current_fetcher import IndianCurrentFetcher
from us_stocks.latest_fetching.yfinance_latest import USLatestFetcher
from ind_stocks.latest_fetching.yfinance_latest import IndianLatestFetcher
from shared.utilities import Config, setup_logger, Constants, StockDataError, get_current_timestamp, categorize_stock, validate_and_categorize_stock
from shared.currency_converter import convert_usd_to_inr, convert_inr_to_usd, get_exchange_rate_info

# Load environment variables
load_dotenv()

# Custom JSON encoder to handle NaN values
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return super().default(obj)

# Setup logging
logger = setup_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.json_encoder = CustomJSONEncoder
CORS(app)

# Initialize configuration
config = Config()

# Initialize components
live_fetcher = LiveFetcher()
indian_fetcher = IndianCurrentFetcher()
us_latest_fetcher = USLatestFetcher()
ind_latest_fetcher = IndianLatestFetcher()

def initialize_dynamic_indexes():
    """Initialize dynamic indexes on startup if needed"""
    try:
        from shared.index_manager import DynamicIndexManager
        
        logger.info("Checking dynamic indexes initialization...")
        index_manager = DynamicIndexManager(config.data_dir)
        
        # Check US stocks
        us_count = index_manager.get_stock_count('us_stocks')
        logger.info(f"US stocks in dynamic index: {us_count}")
        
        if us_count < 400:
            logger.warning(f"US dynamic index has only {us_count} stocks, initializing from permanent...")
            success = index_manager.initialize_from_permanent('us_stocks')
            if success:
                new_count = index_manager.get_stock_count('us_stocks')
                logger.info(f"✅ US dynamic index initialized with {new_count} stocks")
            else:
                logger.error("❌ Failed to initialize US dynamic index")
        
        # Check Indian stocks
        ind_count = index_manager.get_stock_count('ind_stocks')
        logger.info(f"Indian stocks in dynamic index: {ind_count}")
        
        if ind_count < 400:
            logger.warning(f"Indian dynamic index has only {ind_count} stocks, initializing from permanent...")
            success = index_manager.initialize_from_permanent('ind_stocks')
            if success:
                new_count = index_manager.get_stock_count('ind_stocks')
                logger.info(f"✅ Indian dynamic index initialized with {new_count} stocks")
            else:
                logger.error("❌ Failed to initialize Indian dynamic index")
        
        logger.info("Dynamic indexes initialization check completed")
        
    except Exception as e:
        logger.error(f"Error during dynamic indexes initialization: {e}")

# Initialize dynamic indexes on startup
initialize_dynamic_indexes()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Stock Prediction API',
        'version': '1.0.0',
        'timestamp': get_current_timestamp()
    })

@app.route('/live_price', methods=['GET'])
def get_live_price():
    """Get live stock price for a symbol"""
    symbol = request.args.get('symbol')
    
    if not symbol:
        return jsonify({
            'error': 'Symbol parameter is required',
            'message': 'Please provide a stock symbol using ?symbol=SYMBOL'
        }), 400
    
    try:
        logger.info(f"Fetching live price for symbol: {symbol}")
        
        # Use appropriate fetcher based on stock category
        category = validate_and_categorize_stock(symbol)
        if category == 'ind_stocks':
            result = indian_fetcher.fetch_current_price(symbol)
        else:  # Default to US stocks
            result = live_fetcher.fetch_live_price(symbol)
        
        # Add currency conversion information
        try:
            exchange_info = get_exchange_rate_info()
            result['exchange_rate'] = exchange_info['rate']
            result['exchange_source'] = exchange_info['source']
            
            # Add converted prices
            if result.get('currency') == 'USD':
                result['price_inr'] = convert_usd_to_inr(result['price'])
            elif result.get('currency') == 'INR':
                result['price_usd'] = convert_inr_to_usd(result['price'])
                
        except Exception as e:
            logger.warning(f"Currency conversion failed: {e}")
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except ValueError as e:
        logger.warning(f"Invalid symbol {symbol}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Invalid symbol',
            'message': str(e)
        }), 404
        
    except Exception as e:
        logger.error(f"Live price fetch failed for {symbol}: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to fetch price',
            'message': f'Unable to fetch live price for {symbol}.',
            'details': str(e),
            'suggestion': 'Check Upstox token status or permanent directory data'
        }), 500

@app.route('/latest_prices', methods=['GET'])
def get_latest_prices():
    """Get latest prices for all or specific category"""
    category = request.args.get('category')
    
    try:
        data = live_fetcher.get_latest_prices(category)
        
        if not data or (hasattr(data, 'empty') and data.empty):
            return jsonify({
                'success': True,
                'data': [],
                'message': 'No data available'
            })
        
        if hasattr(data, 'to_dict'):
            return jsonify({
                'success': True,
                'data': data.to_dict('records')
            })
        else:
            return jsonify({
                'success': True,
                'data': data
            })
        
    except Exception as e:
        logger.error(f"Error getting latest prices: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get latest prices',
            'message': str(e)
        }), 500

@app.route('/search', methods=['GET'])
def search_stocks():
    """Search for stocks by symbol or company name"""
    query = request.args.get('q', '').strip()
    
    if not query:
        return jsonify({'success': True, 'data': []})
    
    try:
        from shared.index_manager import DynamicIndexManager
        index_manager = DynamicIndexManager(config.data_dir)
        
        results = []
        query_lower = query.lower()
        
        for category in ['us_stocks', 'ind_stocks']:
            index_path = index_manager.get_index_path(category)
            if os.path.exists(index_path):
                df = pd.read_csv(index_path)
                matches = df[
                    (df['symbol'].str.lower().str.contains(query_lower, na=False)) |
                    (df['company_name'].str.lower().str.contains(query_lower, na=False))
                ]
                for _, row in matches.head(10).iterrows():
                    results.append({
                        'symbol': row['symbol'],
                        'name': row.get('company_name', row['symbol'])
                    })
        
        seen = set()
        unique_results = []
        for result in results:
            if result['symbol'] not in seen:
                seen.add(result['symbol'])
                unique_results.append(result)
                if len(unique_results) >= Constants.MAX_SEARCH_RESULTS:
                    break
        
        if not unique_results and query:
            try:
                if query.isupper() and '.' not in query:
                    logger.info(f"Attempting to fetch new Indian stock: {query}")
                    result = indian_fetcher.fetch_current_price(query)
                    if result:
                        unique_results.append({
                            'symbol': result['symbol'],
                            'name': result.get('company_name', result['symbol'])
                        })
                        logger.info(f"Successfully added new stock to search results: {query}")
            except Exception as e:
                logger.info(f"Could not fetch new stock {query}: {e}")
        
        return jsonify({'success': True, 'data': unique_results})
        
    except Exception as e:
        logger.error(f"Error searching stocks: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to search stocks', 'message': str(e)}), 500

@app.route('/symbols', methods=['GET'])
def get_symbols():
    """Get all available symbols by category"""
    category = request.args.get('category')
    
    try:
        from shared.index_manager import DynamicIndexManager
        index_manager = DynamicIndexManager(config.data_dir)
        
        if category:
            symbols = index_manager.get_all_symbols(category)
        else:
            symbols = {
                'us_stocks': index_manager.get_all_symbols('us_stocks'),
                'ind_stocks': index_manager.get_all_symbols('ind_stocks')
            }
        
        return jsonify({'success': True, 'data': symbols})
        
    except Exception as e:
        logger.error(f"Error getting symbols: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to get symbols', 'message': str(e)}), 500

# ML Prediction endpoints
@app.route('/api/predict', methods=['GET'])
def predict_stock_price():
    """Generate stock price prediction using pre-trained ML models"""
    try:
        symbol = request.args.get('symbol')
        horizon_raw = request.args.get('horizon', '1D')
        horizon = horizon_raw.strip().upper() if horizon_raw else '1D'
        model_name = request.args.get('model', 'ensemble')  # Default to ensemble
        
        if not symbol:
            return jsonify({'error': 'Symbol parameter is required'}), 400
        
        # Validate horizon
        valid_horizons = ['1D', '1W', '1M', '1Y', '5Y']
        if horizon not in valid_horizons:
            return jsonify({'error': f'Invalid horizon. Must be one of {valid_horizons}'}), 400
        
        # Import and initialize predictor
        from prediction.predictor import StockPredictor
        from shared.currency_converter import get_live_exchange_rate
        
        predictor = StockPredictor()
        
        # Check if requested model is available
        available_models = list(predictor.models.keys())
        if model_name != 'ensemble' and model_name not in available_models:
            return jsonify({
                'error': f'Model {model_name} not trained yet',
                'available_models': available_models
            }), 404
        
        # Determine stock category
        category = validate_and_categorize_stock(symbol)
        if not category:
            return jsonify({'error': f'Invalid stock symbol: {symbol}'}), 400
        
        current_price_param = request.args.get('current_price')
        data_source = 'live_api'
        data_date = None
        
        if current_price_param:
            current_price = float(current_price_param)
            data_source = 'live_api'
            logger.info(f"Using live price from frontend for {symbol}: ${current_price:.2f}")
        else:
            from prediction.data_loader import DataLoader
            data_loader = DataLoader()
            df = data_loader.load_stock_data(symbol, category)
            if df is None or len(df) == 0:
                return jsonify({'error': 'Could not load stock data for prediction'}), 500
            
            current_price = float(df['close'].iloc[-1])
            data_date = str(df['date'].iloc[-1]) if 'date' in df.columns else None
            data_source = 'stored_data'
            logger.info(f"Using stored price for {symbol}: ${current_price:.2f} from {data_date}")
        
        if model_name == 'ensemble':
            selected_models = None
        else:
            selected_models = [model_name]
        
        result = predictor.predict_single_stock_with_models(
            symbol=symbol,
            category=category,
            horizon=horizon,
            current_price=current_price,
            model_filter=selected_models
        )
        
        if not result:
            logger.error(f"Prediction returned None for {symbol} ({category})")
            return jsonify({
                'error': 'Prediction failed',
                'message': f'Could not generate prediction for {symbol}. Check if model data exists.',
                'details': f'Symbol: {symbol}, Category: {category}, Horizon: {horizon}'
            }), 500
        
        exchange_rate = get_live_exchange_rate()
        
        response = {
            'symbol': symbol,
            'horizon': horizon,
            'predicted_price': result['predicted_price'],
            'current_price': current_price,
            'confidence': result['confidence'],
            'price_range': result.get('price_range'),
            'time_frame_days': result.get('time_frame_days'),
            'model_info': result.get('model_info'),
            'data_points_used': result.get('data_points_used'),
            'last_updated': result.get('last_updated'),
            'currency': 'USD',
            'exchange_rate': exchange_rate,
            'original_category': category,
            'data_source': data_source,
            'data_date': data_date
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return jsonify({'error': 'Prediction failed', 'message': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train_models():
    """Train ML models for a specific symbol"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        models = data.get('models', None)
        force = data.get('force', False)
        max_data_points = data.get('max_data_points', None)
        
        if not symbol:
            return jsonify({
                'error': 'Symbol is required'
            }), 400
        
        from algorithms.utils import predict_for_symbol
        
        result = predict_for_symbol(symbol, '1d', models, max_data_points)
        
        training_results = {
            'symbol': symbol,
            'models_trained': result['model_info']['members'],
            'training_metrics': result.get('individual_predictions', {}),
            'data_points_used': result['data_points_used'],
            'last_updated': result['last_updated']
        }
        
        return jsonify(training_results)
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        return jsonify({
            'error': 'Training failed',
            'message': str(e)
        }), 500


@app.route('/api/models/', methods=['GET'])
def get_models(symbol):
    """Get information about trained models for a symbol"""
    try:
        models_dir = os.path.join('backend', 'models', symbol)
        
        if not os.path.exists(models_dir):
            return jsonify({
                'symbol': symbol,
                'models': [],
                'message': 'No trained models found for this symbol'
            })
        
        model_files = []
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                metadata_file = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        model_files.append({
                            'model_name': item,
                            'saved_at': metadata.get('saved_at'),
                            'training_metrics': metadata.get('training_metrics', {}),
                            'model_params': metadata.get('model_params', {})
                        })
                    except Exception as e:
                        logger.warning(f"Could not read metadata for {item}: {e}")
        
        return jsonify({
            'symbol': symbol,
            'models': model_files,
            'count': len(model_files)
        })
        
    except Exception as e:
        logger.error(f"Error getting models for {symbol}: {str(e)}")
        return jsonify({
            'error': 'Failed to get models',
            'message': str(e)
        }), 500

@app.route('/stock_info', methods=['GET'])
def get_stock_info():
    """Get stock metadata from dynamic index"""
    symbol = request.args.get('symbol')
    
    if not symbol:
        return jsonify({'success': False, 'error': 'Symbol parameter is required'}), 400
    
    try:
        from shared.index_manager import DynamicIndexManager
        
        logger.info(f"Fetching stock info for symbol: {symbol}")
        category = validate_and_categorize_stock(symbol)
        
        index_manager = DynamicIndexManager(config.data_dir)
        stock_info = index_manager.get_stock_info(symbol, category)
        
        if not stock_info:
            return jsonify({
                'success': False,
                'error': 'Symbol not found',
                'message': f'Symbol {symbol} not found in {category} index'
            }), 404
        
        return jsonify({'success': True, 'data': stock_info})
        
    except Exception as e:
        logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch stock info',
            'message': str(e)
        }), 500

@app.route('/company_info', methods=['GET'])
def get_company_info():
    """[FUTURE] Get comprehensive company information"""
    symbol = request.args.get('symbol')
    info_type = request.args.get('info_type', 'all')
    
    if not symbol:
        return jsonify({
            'success': False,
            'error': 'Symbol parameter is required'
        }), 400
    
    return jsonify({
        'success': False,
        'error': 'Company info feature not yet implemented',
        'message': 'This endpoint will be available in a future update'
    }), 501

@app.route('/historical', methods=['GET'])
def get_historical_data():
    """Get historical stock data for chart visualization"""
    symbol = request.args.get('symbol')
    period = request.args.get('period')
    
    if not symbol:
        return jsonify({
            'success': False,
            'error': 'Symbol parameter is required'
        }), 400
    
    if not period or period not in ['week', 'month', 'year', '5year']:
        return jsonify({
            'success': False,
            'error': 'Period parameter is required and must be one of: week, month, year, 5year'
        }), 400
    
    try:
        logger.info(f"Fetching historical data for {symbol} ({period})")
        category = validate_and_categorize_stock(symbol)
        
        from datetime import datetime, timedelta
        today = datetime.now().date()
        
        if period == 'week':
            start_date = today - timedelta(days=7)
        elif period == 'month':
            start_date = today - timedelta(days=30)
        elif period == 'year':
            start_date = datetime(2024, 1, 1).date()
        elif period == '5year':
            start_date = None
        
        from shared.index_manager import DynamicIndexManager
        index_manager = DynamicIndexManager(config.data_dir)
        stock_info = index_manager.get_stock_info(symbol, category)
        default_currency = stock_info.get('currency', 'USD') if stock_info else 'USD'
        
        past_file = os.path.join(config.data_dir, 'past', category, 'individual_files', f'{symbol}.csv')
        latest_file = os.path.join(config.data_dir, 'latest', category, 'individual_files', f'{symbol}.csv')
        
        historical_data = []
        
        if os.path.exists(past_file):
            try:
                import pandas as pd
                df_past = pd.read_csv(past_file)
                try:
                    df_past['date'] = pd.to_datetime(df_past['date'], utc=True).dt.tz_localize(None).dt.date
                except:
                    df_past['date'] = pd.to_datetime(df_past['date']).dt.date
                historical_data.append(df_past)
                logger.info(f"Loaded {len(df_past)} records from past data")
            except Exception as e:
                logger.warning(f"Could not read past data for {symbol}: {e}")
        
        if os.path.exists(latest_file):
            try:
                import pandas as pd
                df_latest = pd.read_csv(latest_file)
                df_latest['date'] = pd.to_datetime(df_latest['date']).dt.date
                historical_data.append(df_latest)
                logger.info(f"Loaded {len(df_latest)} records from latest data")
            except Exception as e:
                logger.warning(f"Could not read latest data for {symbol}: {e}")
        
        if not historical_data:
            logger.info(f"Trying permanent directory for {symbol}")
            permanent_file = os.path.join(config.permanent_dir, category, 'individual_files', f'{symbol}.csv')
            
            if os.path.exists(permanent_file):
                try:
                    import pandas as pd
                    df_perm = pd.read_csv(permanent_file)
                    try:
                        df_perm['date'] = pd.to_datetime(df_perm['date'], utc=True).dt.tz_localize(None).dt.date
                    except:
                        df_perm['date'] = pd.to_datetime(df_perm['date']).dt.date
                    historical_data.append(df_perm)
                    logger.info(f"✅ Loaded {len(df_perm)} records from permanent directory (offline mode)")
                except Exception as e:
                    logger.warning(f"Could not read permanent data for {symbol}: {e}")
        
        if not historical_data:
            return jsonify({
                'success': False,
                'error': 'No data files found',
                'message': f'No historical data found for symbol {symbol}'
            }), 404
        
        try:
            import pandas as pd
            combined_df = pd.concat(historical_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['date']).sort_values('date')
            
            if len(combined_df) > 0:
                logger.info(f"Combined data date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
                logger.info(f"Total combined data points: {len(combined_df)}")
            
            if period == 'year':
                logger.info(f"Getting 1 year historical data for {symbol} from 2024-01-01")
                filtered_df = combined_df[combined_df['date'] >= start_date]
                logger.info(f"Selected {len(filtered_df)} data points for 1-year chart")
                if len(filtered_df) > 0:
                    logger.info(f"Filtered data date range: {filtered_df['date'].min()} to {filtered_df['date'].max()}")
            elif period == '5year':
                logger.info(f"Getting all available historical data for {symbol} (5-year chart)")
                filtered_df = combined_df
                logger.info(f"Selected {len(filtered_df)} data points for 5-year chart")
            else:
                filtered_df = combined_df[combined_df['date'] >= start_date]
            
            price_points = []
            for _, row in filtered_df.iterrows():
                price_points.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'open': float(row['open']) if pd.notna(row['open']) else None,
                    'high': float(row['high']) if pd.notna(row['high']) else None,
                    'low': float(row['low']) if pd.notna(row['low']) else None,
                    'close': float(row['close']) if pd.notna(row['close']) else None,
                    'volume': int(row['volume']) if pd.notna(row['volume']) else 0,
                    'currency': row['currency'] if pd.notna(row.get('currency')) else default_currency
                })
            
            if period not in ['year', '5year']:
                logger.info(f"Checking if additional data needed for {symbol} {period}: {len(price_points)} existing points")
                
                if category == 'us_stocks':
                    additional_data = us_latest_fetcher.fetch_recent_data_on_demand(symbol, period, price_points)
                elif category == 'ind_stocks':
                    additional_data = ind_latest_fetcher.fetch_recent_data_on_demand(symbol, period, price_points)
                else:
                    additional_data = []
                
                logger.info(f"Additional data fetched: {len(additional_data)} points")
                
                if additional_data:
                    all_data = price_points + additional_data
                    seen_dates = set()
                    unique_data = []
                    for point in sorted(all_data, key=lambda x: x['date']):
                        if point['date'] not in seen_dates:
                            seen_dates.add(point['date'])
                            unique_data.append(point)
                    price_points = unique_data
                    logger.info(f"Combined data: {len(price_points)} total points for {symbol} ({period})")
            
            if not price_points:
                return jsonify({
                    'success': False,
                    'error': 'No data in date range',
                    'message': f'No data available for {symbol} in the specified {period} period'
                }), 404
            
            logger.info(f"Returning {len(price_points)} price points for {symbol} ({period})")
            
            return jsonify({
                'success': True,
                'data': price_points
            })
            
        except ImportError:
            import csv
            from datetime import datetime
            
            all_records = []
            for file_path in [past_file, latest_file]:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', newline='', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                try:
                                    date_obj = datetime.strptime(row['date'][:10], '%Y-%m-%d').date()
                                    if date_obj >= start_date:
                                        all_records.append({
                                            'date': row['date'][:10],
                                            'open': float(row['open']) if row['open'] else None,
                                            'high': float(row['high']) if row['high'] else None,
                                            'low': float(row['low']) if row['low'] else None,
                                            'close': float(row['close']) if row['close'] else None,
                                            'volume': int(float(row['volume'])) if row['volume'] else 0,
                                            'currency': default_currency
                                        })
                                except (ValueError, KeyError) as e:
                                    continue
                    except Exception as e:
                        logger.warning(f"Error reading {file_path}: {e}")
            
            if not all_records:
                return jsonify({
                    'success': False,
                    'error': 'No data in date range',
                    'message': f'No data available for {symbol} in the specified {period} period'
                }), 404
            
            all_records.sort(key=lambda x: x['date'])
            
            return jsonify({
                'success': True,
                'data': all_records
            })
        
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch historical data',
            'message': f'Unable to fetch historical data for {symbol}. Please try again later.',
            'details': str(e)
        }), 500

@app.route('/algorithms', methods=['GET'])
def get_available_algorithms():
    """[FUTURE] Get list of available prediction algorithms"""
    return jsonify({
        'success': True,
        'data': {
            'available_algorithms': [
                'Random Forest',
                'LSTM',
                'ARIMA',
                'Linear Regression',
                'Support Vector Machine',
                'Gradient Boosting',
                'Neural Network',
                'Technical Analysis',
                'Sentiment Analysis',
                'Ensemble Method'
            ],
            'status': 'Placeholder - algorithms not yet implemented'
        }
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Not found',
        'message': 'The requested endpoint was not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

def find_free_port(start_port=5000):
    """Find a free port starting from start_port"""
    import socket
    port = start_port
    while port < start_port + 100:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            port += 1
    raise RuntimeError("Could not find a free port")

def start_server(port=None):
    """Start the Flask server"""
    if port is None:
        port = config.port
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except OSError:
        free_port = find_free_port(port)
        logger.info(f"Port {port} is busy, using port {free_port}")
        app.run(host='0.0.0.0', port=free_port, debug=False)

if __name__ == '__main__':
    logger.info(f"Starting Stock Prediction API server on port {config.port}")
    start_server(config.port)
