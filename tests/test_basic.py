# tests/test_basic.py
import pandas as pd
import os

def test_sentiment_score_range():
    filepath = 'data/features/combined_features.csv'
    assert os.path.exists(filepath), f"{filepath} not found"
    df = pd.read_csv(filepath)

    assert 'sentiment_score' in df.columns, "'sentiment_score' column missing"
    assert df['sentiment_score'].dtype in [float, int], "sentiment_score should be numeric"
    assert df['sentiment_score'].between(-1, 1).all(), "sentiment_score should be between -1 and 1"

def test_combined_features_exists():
    filepath = 'data/features/combined_features.csv'
    assert os.path.exists(filepath), f"{filepath} not found"

    df = pd.read_csv(filepath)
    assert not df.empty, "combined_features.csv is empty"
    assert 'ticker' in df.columns and 'target' in df.columns, "Missing expected columns"

