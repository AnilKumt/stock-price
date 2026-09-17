#!/usr/bin/env python3
"""
Simple Model Training Status Checker

This is the ONLY status checking script for the project.
Shows current training status of all ML models in a clean format.

Usage:
    python status.py                    # Table format (default)
    python status.py --json             # JSON format
    python status.py --simple           # Simple one-line format
    python status.py --help             # Show help
"""

import sys
import os
import json
import argparse
from datetime import datetime


def get_training_summary():
    """Get training status summary for all ML models by inspecting trained model artifacts and metadata."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, 'backend', 'models')
    status_file = os.path.join(models_dir, 'model_status.json')
    
    saved_status = {}
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                saved_status = json.load(f)
        except Exception:
            pass
            
    model_list = [
        ('linear_regression', 'Linear Regression', 'linear_regression_model.pkl'),
        ('decision_tree', 'Decision Tree', 'decision_tree_model.pkl'),
        ('random_forest', 'Random Forest', 'random_forest_model.pkl'),
        ('svm', 'SVM', 'svm_model.pkl'),
        ('knn', 'KNN', 'knn_model.pkl'),
        ('arima', 'ARIMA', 'arima_model.pkl'),
        ('autoencoder', 'Autoencoder', 'autoencoder_model.pkl_metadata.pkl'),
    ]
    
    summary_models = {}
    completed_count = 0
    failed_count = 0
    pending_count = 0
    
    for key, display_name, file_name in model_list:
        model_info = saved_status.get(key, {})
        model_dir = os.path.join(models_dir, key)
        model_file_path = os.path.join(model_dir, file_name)
        
        # Check if model artifact exists on disk
        file_exists = False
        if key == 'arima':
            file_exists = os.path.exists(model_file_path) or (os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0)
        else:
            file_exists = os.path.exists(model_file_path)
            
        is_trained = model_info.get('trained', file_exists)
        status_str = model_info.get('status', 'completed' if file_exists else 'pending')
        
        last_updated = model_info.get('last_updated', model_info.get('trained_date', ''))
        if file_exists and not last_updated:
            target_path = model_file_path if os.path.exists(model_file_path) else model_dir
            try:
                mtime = os.path.getmtime(target_path)
                last_updated = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                pass
                
        error = model_info.get('error_message', model_info.get('error', ''))
        stocks_trained = model_info.get('stocks_trained', 936 if is_trained else 0)
        
        # Extract R² score
        r2_score = model_info.get('r2_score')
        if r2_score is None and 'validation_metrics' in model_info:
            r2_score = model_info['validation_metrics'].get('avg_r2_score')
            
        if error:
            failed_count += 1
        elif is_trained or status_str == 'completed':
            completed_count += 1
        else:
            pending_count += 1
            
        summary_models[display_name] = {
            'details': {
                'status': status_str,
                'trained': is_trained,
                'stocks_trained': stocks_trained,
                'r2_score': r2_score,
                'trained_date': last_updated,
                'last_updated': last_updated,
                'error': error,
                'error_message': error,
                'validation_metrics': {'avg_r2_score': r2_score} if r2_score is not None else {}
            }
        }
        
    return {
        'total_models': len(model_list),
        'completed': completed_count,
        'failed': failed_count,
        'pending': pending_count,
        'models': summary_models
    }


def format_r2_score(r2_score):
    """Format R² score for display."""
    if r2_score is None:
        return "N/A"
    elif r2_score > 1000:
        return f"{r2_score:.0f}"
    elif r2_score > 0:
        return f"{r2_score:.3f}"
    else:
        return f"{r2_score:.1f}"


def format_date(date_str):
    """Format date for display."""
    if date_str:
        return date_str[:10]  # Just the date part
    return "N/A"


def format_error(error_str):
    """Format error message for display."""
    if not error_str:
        return ""
    return error_str[:50] + "..." if len(error_str) > 50 else error_str


def print_status_table(summary):
    """Print status in table format."""
    print("="*100)
    print("ML MODEL TRAINING STATUS")
    print("="*100)
    print(f"Total Models: {summary['total_models']}")
    print(f"Completed: {summary['completed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pending: {summary['pending']}")
    print("")
    
    print("Model Details:")
    print("-" * 100)
    print(f"{'Model Name':<20} | {'Trained':<8} | {'Stocks':<8} | {'R² Score':<10} | {'Date':<12} | {'Error'}")
    print("-" * 100)
    
    for name, details in summary['models'].items():
        model_details = details['details']
        
        status = model_details.get('status', 'pending')
        trained = (status == 'completed') or model_details.get('trained', False)
        stocks_trained = model_details.get('stocks_trained', 0)
        
        validation_metrics = model_details.get('validation_metrics', {})
        r2_score = validation_metrics.get('avg_r2_score') if validation_metrics else model_details.get('r2_score')
        
        trained_date = model_details.get('last_updated', model_details.get('trained_date', ''))
        error = model_details.get('error_message', model_details.get('error', ''))
        
        trained_str = "Yes" if trained else "No"
        r2_str = format_r2_score(r2_score)
        date_str = format_date(trained_date)
        error_str = format_error(error)
        
        print(f"{name:<20} | {trained_str:<8} | {stocks_trained:<8} | {r2_str:<10} | {date_str:<12} | {error_str}")
    
    print("="*100)


def print_status_json(summary):
    """Print status as JSON."""
    models = {}
    for name, details in summary['models'].items():
        model_details = details['details']
        status = model_details.get('status', 'pending')
        trained = (status == 'completed') or model_details.get('trained', False)
        validation_metrics = model_details.get('validation_metrics', {})
        r2_score = validation_metrics.get('avg_r2_score') if validation_metrics else model_details.get('r2_score')
        trained_date = model_details.get('last_updated', model_details.get('trained_date', ''))
        error = model_details.get('error_message', model_details.get('error', ''))
        
        models[name] = {
            'trained': trained,
            'stocks_trained': model_details.get('stocks_trained', 0),
            'r2_score': r2_score,
            'trained_date': trained_date,
            'error': error
        }
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_models': summary['total_models'],
            'completed': summary['completed'],
            'failed': summary['failed'],
            'pending': summary['pending']
        },
        'models': models
    }
    
    print(json.dumps(output, indent=2))


def print_status_simple(summary):
    """Print status in simple one-line format."""
    print(f"Models: {summary['completed']}/{summary['total_models']} completed, {summary['failed']} failed, {summary['pending']} pending")
    
    for name, details in summary['models'].items():
        model_details = details['details']
        status = model_details.get('status', 'pending')
        trained = (status == 'completed') or model_details.get('trained', False)
        stocks_trained = model_details.get('stocks_trained', 0)
        validation_metrics = model_details.get('validation_metrics', {})
        r2_score = validation_metrics.get('avg_r2_score') if validation_metrics else model_details.get('r2_score')
        error = model_details.get('error_message', model_details.get('error', ''))
        
        status_icon = "✅" if trained and not error else "❌" if error else "⏳"
        r2_str = format_r2_score(r2_score)
        
        print(f"{status_icon} {name}: {stocks_trained} stocks, R²={r2_str}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Check ML model training status')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--simple', action='store_true', help='Simple one-line format')
    
    args = parser.parse_args()
    
    try:
        summary = get_training_summary()
        
        if args.json:
            print_status_json(summary)
        elif args.simple:
            print_status_simple(summary)
        else:
            print_status_table(summary)
            
    except Exception as e:
        print(f"Error checking status: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
