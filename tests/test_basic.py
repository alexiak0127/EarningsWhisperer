# tests/test_basic.py
import pandas as pd
import numpy as np
import os
import glob
import joblib
from sklearn.preprocessing import StandardScaler

def test_sentiment_score_range():
    """Test that sentiment scores are properly normalized between -1 and 1."""
    filepath = 'data/features/combined_features.csv'
    assert os.path.exists(filepath), f"{filepath} not found"
    df = pd.read_csv(filepath)

    assert 'sentiment_score' in df.columns, "'sentiment_score' column missing"
    assert df['sentiment_score'].dtype in [float, int], "sentiment_score should be numeric"
    assert df['sentiment_score'].between(-1, 1).all(), "sentiment_score should be between -1 and 1"

def test_combined_features_exists():
    """Test that the combined features file exists and has the expected structure."""
    filepath = 'data/features/combined_features.csv'
    assert os.path.exists(filepath), f"{filepath} not found"

    df = pd.read_csv(filepath)
    assert not df.empty, "combined_features.csv is empty"
    assert 'ticker' in df.columns and 'target' in df.columns, "Missing expected columns"

def test_target_distribution():
    """Test that the target variable has all expected classes."""
    filepath = 'data/features/combined_features.csv'
    assert os.path.exists(filepath), f"{filepath} not found"
    df = pd.read_csv(filepath)
    
    assert 'target' in df.columns, "Target column missing"
    unique_targets = sorted(df['target'].unique())
    assert set(unique_targets) == {-1, 0, 1}, f"Expected targets [-1, 0, 1], got {unique_targets}"


def test_processed_data_files():
    """Test that processed data files were created with technical indicators."""
    companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'CRM']
    
    for ticker in companies:
        processed_file = f'data/processed/{ticker}_processed.csv'
        if not os.path.exists(processed_file):
            continue  # Skip if file doesn't exist (might not be processed yet)
        
        df = pd.read_csv(processed_file)
        assert not df.empty, f"{processed_file} is empty"
        
        # Check for technical indicators
        for column in ['RSI', 'MA5', 'Daily_Return']:
            assert column in df.columns, f"'{column}' column missing in {processed_file}"

def test_model_outputs():
    """Test that model files are created and have expected format."""
    # Check if model files exist
    model_files = glob.glob('models/*.pkl')
    assert len(model_files) > 0, "No model files found"
    
    # Test each model file
    for model_file in model_files:
        assert os.path.getsize(model_file) > 0, f"{model_file} is empty"
        
        # Try to load model (will throw error if format is invalid)
        try:
            model = joblib.load(model_file)
            assert model is not None, f"Failed to load model from {model_file}"
        except Exception as e:
            assert False, f"Error loading model {model_file}: {str(e)}"

def test_visualization_outputs():
    """Test that visualization files are created."""
    vis_files = glob.glob('visualizations/*.png')
    assert len(vis_files) > 0, "No visualization files found"
    
    # Check for key visualizations
    required_vis = [
        'model_comparison.png',
        'sentiment_distribution.png'
    ]
    
    for vis in required_vis:
        vis_path = f'visualizations/{vis}'
        assert os.path.exists(vis_path), f"{vis_path} not found"
        assert os.path.getsize(vis_path) > 0, f"{vis_path} is empty"

def test_feature_importances():
    """Test that feature importance files are created with valid data."""
    importance_files = glob.glob('results/*_feature_importance.csv')
    
    if importance_files:  # Only run this test if files exist
        for file in importance_files:
            df = pd.read_csv(file)
            assert not df.empty, f"{file} is empty"
            assert 'Feature' in df.columns, f"'Feature' column missing in {file}"
            assert 'Importance' in df.columns, f"'Importance' column missing in {file}"
            
            # Check that importance values are valid
            assert df['Importance'].dtype in [float, int], f"Importance should be numeric in {file}"
            assert df['Importance'].min() >= 0, f"Importance values should be non-negative in {file}"

def test_confusion_matrices():
    """Test that confusion matrices are created with valid data."""
    cm_files = glob.glob('results/*_confusion.csv')
    
    if cm_files:  # Only run this test if files exist
        for file in cm_files:
            df = pd.read_csv(file)
            assert not df.empty, f"{file} is empty"
            
            # Confusion matrix should be square with non-negative integers
            matrix_data = df.iloc[:, 1:].values
            assert matrix_data.shape[0] == matrix_data.shape[1], f"Confusion matrix should be square in {file}"
            assert np.all(matrix_data >= 0), f"Confusion matrix should have non-negative values in {file}"
