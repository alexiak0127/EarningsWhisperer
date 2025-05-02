import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
import joblib
from xgboost import XGBClassifier 

# Create directories
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)


# Load the feature dataset
def load_features():
    features_file = 'data/features/combined_features.csv'
    
    if not os.path.exists(features_file):
        print(f"Features file not found: {features_file}")
        return None
    
    df = pd.read_csv(features_file)
    print(f"Loaded {len(df)} samples from {features_file}")
    
    return df


# Prepare data for modeling
def prepare_data(df, test_size=0.3):
    print("\nPreparing data for modeling...")
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Create dummy variables for sentiment
    if 'sentiment' in df.columns:
        df = pd.get_dummies(df, columns=['sentiment'], drop_first=False)
    
    # Select features
    potential_features = [
        'sentiment_score', 'pre_return', 'pre_volume_change', 'pre_rsi'
    ]
    features = [col for col in potential_features if col in df.columns]
    sentiment_cols = [col for col in df.columns if col.startswith('sentiment_')]
    features.extend(sentiment_cols)
    
    print(f"Using features: {features}")
    
    # Split features and target
    X = df[features]
    y = df['target']
    
    # Check class distribution
    class_counts = pd.Series(y).value_counts().sort_index()
    print(f"Class distribution: {class_counts.to_dict()}")
    
    # Increase test size for small datasets to ensure at least 3 samples per class
    if min(class_counts) < 10:
        print("Small dataset detected. Adjusting test size to ensure enough samples per class.")
        test_size = 0.5
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Check class balance in test set
    test_class_counts = pd.Series(y_test).value_counts().sort_index()
    print(f"Test set class distribution: {test_class_counts.to_dict()}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, features, scaler


# Random baseline model
def random_baseline_model(X_test, y_test):
    print("\nEvaluating Random Baseline...")

    np.random.seed(42)
    y_pred = np.random.choice([-1, 0, 1], size=len(y_test))

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nRandom Baseline Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)

    cm_df = pd.DataFrame(
        cm,
        columns=['Pred_Down', 'Pred_Stable', 'Pred_Up'],
        index=['True_Down', 'True_Stable', 'True_Up']
    )
    cm_df.to_csv('results/random_baseline_confusion.csv')

    pred_df = pd.DataFrame({
        'true': y_test,
        'pred': y_pred
    })
    pred_df.to_csv('results/random_baseline_predictions.csv')

    return {
        'accuracy': accuracy,
        'report': report
    }


# Train logistic regression model with hyperparameter tuning
def train_logistic_regression(X_train, y_train, cv=5):
    print("\nTraining logistic regression model with hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['lbfgs'],  # Use only lbfgs for multinomial
        'max_iter': [1000]
    }
    
    # Create base model
    base_model = LogisticRegression(
        multi_class='multinomial',
        random_state=42
    )
    
    # Create StratifiedKFold for cross-validation
    cv_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Train model
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Return best model
    return grid_search.best_estimator_


# Train random forest model with hyperparameter tuning
def train_random_forest(X_train, y_train, cv=5):
    print("\nTraining random forest model with hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create base model
    base_model = RandomForestClassifier(random_state=42)
    
    # Create StratifiedKFold for cross-validation
    cv_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Train model
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Return best model
    return grid_search.best_estimator_


# Train AdaBoost model with hyperparameter tuning
def train_adaboost(X_train, y_train, cv=5):
    print("\nTraining AdaBoost model with hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }
    
    # Create base model
    base_model = AdaBoostClassifier(random_state=42)
    
    # Create StratifiedKFold for cross-validation
    cv_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Train model
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Return best model
    return grid_search.best_estimator_


# Train XGBoost model with hyperparameter tuning
def train_xgboost(X_train, y_train, cv=5):
    
    print("\nTraining XGBoost model with hyperparameter tuning...")
    
    # Map target values for XGBoost (from -1,0,1 to 0,1,2)
    target_mapping = {-1: 0, 0: 1, 1: 2}
    y_train_mapped = np.array([target_mapping.get(label, 0) for label in y_train])
    
    # Define parameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Create base model
    base_model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        random_state=42
    )
    
    # Create StratifiedKFold for cross-validation
    cv_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Train model
    grid_search.fit(X_train, y_train_mapped)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Save mapping for later use
    joblib.dump(target_mapping, 'models/xgboost_mapping.pkl')
    
    # Return best model
    return grid_search.best_estimator_


# Evaluate model
def evaluate_model(model, X_test, y_test, model_name, features):
    print(f"\nEvaluating {model_name}...")
    
    # Skip if model is None (could happen with XGBoost)
    if model is None:
        print(f"No {model_name} model to evaluate.")
        return None
    
    # Handle XGBoost predictions specially (need to map values back)
    if model_name == 'XGBoost':
        # Load mapping
        target_mapping = joblib.load('models/xgboost_mapping.pkl')
        reverse_mapping = {0: -1, 1: 0, 2: 1}
        
        # Get mapped predictions
        y_pred_mapped = model.predict(X_test)
        
        # Convert back to original values
        y_pred = np.array([reverse_mapping.get(label, 0) for label in y_pred_mapped])
    else:
        # Regular prediction
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)
    
    # Define class labels for confusion matrix
    class_labels = {-1: 'Down', 0: 'Stable', 1: 'Up'}
    
    # Get unique classes present in y_test and y_pred
    unique_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
    
    # Create labels for confusion matrix
    col_labels = [f"Pred_{class_labels.get(c, str(c))}" for c in unique_classes]
    row_labels = [f"True_{class_labels.get(c, str(c))}" for c in unique_classes]
    
    # Save confusion matrix
    cm_df = pd.DataFrame(cm, columns=col_labels, index=row_labels)
    cm_df.to_csv(f'results/{model_name.lower().replace(" ", "_")}_confusion.csv')
    
    # Save predictions
    pred_df = pd.DataFrame({'true': y_test, 'pred': y_pred})
    pred_df.to_csv(f'results/{model_name.lower().replace(" ", "_")}_predictions.csv')
    
    # Save model-specific outputs
    if model_name == 'Logistic Regression':
        if hasattr(model, 'coef_'):
            # Save coefficients for each class
            for i, c in enumerate(model.classes_):
                class_name = class_labels.get(c, str(c)).lower()
                coef_df = pd.DataFrame({
                    'Feature': features,
                    'Coefficient': model.coef_[i]
                })
                coef_df.to_csv(f'results/logistic_regression_coefficients_{class_name}.csv', index=False)
    
    elif model_name in ['Random Forest', 'AdaBoost', 'XGBoost']:
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            importance_df.to_csv(f'results/{model_name.lower().replace(" ", "_")}_feature_importance.csv', index=False)
    
    return {
        'accuracy': accuracy,
        'report': report
    }


# Main function
def main():
    print("Starting modeling process...")
    
    # Load dataset
    df = load_features()
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test, features, scaler = prepare_data(df)
    
    # Save scaler for future use
    joblib.dump(scaler, 'models/standard_scaler.pkl')
    
    # Train and evaluate models
    model_results = {}
    
    # Random baseline
    model_results['Random Baseline'] = random_baseline_model(X_test, y_test)
    
    # Number of cross-validation folds
    # Adjust automatically for small datasets
    n_classes = len(np.unique(y_train))
    min_samples_per_class = min(pd.Series(y_train).value_counts())
    cv_folds = min(5, min_samples_per_class // 2)  # Ensure at least 2 samples per class per fold
    cv_folds = max(2, cv_folds)  # Minimum 2 folds
    print(f"\nUsing {cv_folds} cross-validation folds based on dataset size")
    
    # Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train, cv=cv_folds)
    joblib.dump(lr_model, 'models/logistic_regression.pkl')
    model_results['Logistic Regression'] = evaluate_model(lr_model, X_test, y_test, 'Logistic Regression', features)
    
    # Random Forest
    rf_model = train_random_forest(X_train, y_train, cv=cv_folds)
    joblib.dump(rf_model, 'models/random_forest.pkl')
    model_results['Random Forest'] = evaluate_model(rf_model, X_test, y_test, 'Random Forest', features)
    
    # AdaBoost
    ada_model = train_adaboost(X_train, y_train, cv=cv_folds)
    joblib.dump(ada_model, 'models/adaboost.pkl')
    model_results['AdaBoost'] = evaluate_model(ada_model, X_test, y_test, 'AdaBoost', features)
    
    # XGBoost
    xgb_model = train_xgboost(X_train, y_train, cv=cv_folds)
    if xgb_model is not None:
        joblib.dump(xgb_model, 'models/xgboost.pkl')
        model_results['XGBoost'] = evaluate_model(xgb_model, X_test, y_test, 'XGBoost', features)
    
    # Save model comparison
    results_df = pd.DataFrame({
        'Model': [],
        'Accuracy': []
    })
    
    for model_name, result in model_results.items():
        if result is not None:
            results_df = pd.concat([results_df, pd.DataFrame({
                'Model': [model_name],
                'Accuracy': [result['accuracy']]
            })], ignore_index=True)
    
    # Sort by accuracy
    results_df = results_df.sort_values('Accuracy', ascending=False)
    results_df.to_csv('results/model_comparison.csv', index=False)
    
    print("\nModeling complete!")
    print("Run visualization.py to create charts.")


if __name__ == "__main__":
    main()