import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.makedirs('results', exist_ok=True)


# loads the feature dataset
def load_features():
    features_file = 'data/features/combined_features.csv'
    
    # error - fixed
    if not os.path.exists(features_file):
        print(f"Features file not found: {features_file}")
        return None
    
    df = pd.read_csv(features_file)
    
    print(f"Loaded {len(df)} samples from {features_file}")
    
    return df

#  -- prepare data for modeling - handles preprocessing --
def prepare_data(df):

    print("Preparing data for modeling...")
    
    # drop rows with missing values
    original_len = len(df)
    df = df.dropna()
    print(f"Removed {original_len - len(df)} rows with missing values")
    
    # text -> numeric dummy
    if 'sentiment' in df.columns:
        df = pd.get_dummies(df, columns=['sentiment'])
    
    # Feature columns to use for modeling
    potential_features = [
        'sentiment_score', 'pre_return', 'pre_volume_change', 'pre_rsi'
    ]
    
    # Check if all columns exist
    features = [col for col in potential_features if col in df.columns]
    
    # Add one-hot encoded features if they exist
    sentiment_cols = [col for col in df.columns if col.startswith('sentiment_')]
    features.extend(sentiment_cols)

    print(f"Using these features: {features}")
    
    # Ensure the target exists
    if 'target' not in df.columns:
        print("Error: No target column found!")
        return None, None, None, None, None
    
    # Split features and target
    X = df[features]
    y = df['target']
    
    # Split into training and testing sets 
    # - 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, features, scaler


# -- train logistic regression model --

def train_logistic_regression(X_train, y_train):
    """Train a simple logistic regression model"""
    
    print("Training logistic regression model...")

    model = LogisticRegression(max_iter=1000,  random_state=42)
    model.fit(X_train, y_train)
    
    return model


# -- Train Random Forest model --

def train_random_forest(X_train, y_train, features):
    """Train a random forest classifier"""
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model


# -- Evaluate the model --

def evaluate_model(model, X_test, y_test, model_name, features):
    """Evaluate model performance"""
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)

    # Save confusion matrix for visualization
    cm_df = pd.DataFrame(
        cm, 
        columns=['Pred_Down', 'Pred_Stable', 'Pred_Up'],
        index=['True_Down', 'True_Stable', 'True_Up']
    )
    cm_df.to_csv(f'results/{model_name.lower().replace(" ", "_")}_confusion.csv')
    
  # Save predictions for later analysis
    pred_df = pd.DataFrame({
        'true': y_test,
        'pred': y_pred
    })
    pred_df.to_csv(f'results/{model_name.lower().replace(" ", "_")}_predictions.csv')
    
    # For logistic regression, save coefficients
    if isinstance(model, LogisticRegression):
        if hasattr(model, 'coef_'):
            for i, target_class in enumerate(['Down', 'Stable', 'Up']):
                coef_df = pd.DataFrame({
                    'Feature': features,
                    'Coefficient': model.coef_[i]
                })
                coef_df.to_csv(f'results/logistic_regression_coefficients_{target_class.lower()}.csv', index=False)
    
    return {
        'accuracy': accuracy,
        'report': report
    }

def main():
    print("Starting modeling...")
    
    df = load_features()
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    X_train, X_test, y_train, y_test, features, scaler = prepare_data(df)
    if X_train is None:
        print("Data preparation failed. Exiting.")
        return
    
    model_results = {}
    
    print("\n--- Logistic Regression ---")
    lr_model = train_logistic_regression(X_train, y_train)
    model_results['Logistic Regression'] = evaluate_model(lr_model, X_test, y_test, 'Logistic Regression', features)
    
    print("\n--- Random Forest ---")
    rf_model = train_random_forest(X_train, y_train, features)
    model_results['Random Forest'] = evaluate_model(rf_model, X_test, y_test, 'Random Forest', features)
    
    results_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'Accuracy': [r['accuracy'] for r in model_results.values()]
    })
    results_df.to_csv('results/model_comparison.csv', index=False)
    
    print("\nModeling complete! Run visualization.py to create charts.")


if __name__ == "__main__":
    main()