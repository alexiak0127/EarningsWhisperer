import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer
import time
import joblib

os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)


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

# -- Hyperparameter tuning for logistic regression --
def tune_logistic_regression(X_train, y_train, cv=5):
    """Tune hyperparameters for logistic regression model"""
    
    print("Tuning logistic regression hyperparameters...")
    
    # Define parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [1000, 2000]
    }
    
    # Remove invalid combinations
    valid_params = []
    for p in param_grid['penalty']:
        for s in param_grid['solver']:
            # l1 only works with liblinear and saga
            if p == 'l1' and s not in ['liblinear', 'saga']:
                continue
            # elasticnet only works with saga
            if p == 'elasticnet' and s != 'saga':
                continue
            # None penalty works with all solvers
            valid_params.append({'penalty': p, 'solver': s})
    
    # Create base model
    lr = LogisticRegression(random_state=42)
    
    # Use RandomizedSearchCV to find best parameters
    random_search = RandomizedSearchCV(
        estimator=lr,
        param_distributions={
            'C': param_grid['C'],
            'max_iter': param_grid['max_iter']
        },
        n_iter=10,
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Iterate through valid penalty/solver combinations
    best_score = 0
    best_params = None
    best_model = None
    
    for params in valid_params:
        try:
            print(f"Trying parameters: {params}")
            lr.set_params(**params)
            random_search.estimator = lr
            random_search.fit(X_train, y_train)
            
            if random_search.best_score_ > best_score:
                best_score = random_search.best_score_
                best_params = {**params, **random_search.best_params_}
                best_model = random_search.best_estimator_
        except Exception as e:
            print(f"Error with parameters {params}: {e}")
    
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")
    
    # Retrain with best parameters
    final_model = LogisticRegression(**best_params, random_state=42)
    final_model.fit(X_train, y_train)
    
    # Save the model and parameters
    joblib.dump(final_model, 'models/logistic_regression_tuned.pkl')
    pd.DataFrame([best_params]).to_csv('results/logistic_regression_best_params.csv', index=False)
    
    return final_model


# -- Hyperparameter tuning for Random Forest --
def tune_random_forest(X_train, y_train, features, cv=5):
    """Tune hyperparameters for random forest classifier"""
    
    print("Tuning Random Forest hyperparameters...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create base model
    rf = RandomForestClassifier(random_state=42)
    
    # Use RandomizedSearchCV to find best parameters
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=20,
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    # Retrain with best parameters
    final_model = RandomForestClassifier(**best_params, random_state=42)
    final_model.fit(X_train, y_train)
    
    # Save the model and parameters
    joblib.dump(final_model, 'models/random_forest_tuned.pkl')
    pd.DataFrame([best_params]).to_csv('results/random_forest_best_params.csv', index=False)
    
    return final_model


# -- Hyperparameter tuning for AdaBoost --
def tune_adaboost(X_train, y_train, features, cv=5):
    """Tune hyperparameters for AdaBoost classifier"""
    
    print("Tuning AdaBoost hyperparameters...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'algorithm': ['SAMME', 'SAMME.R']
    }
    
    # Create base model
    adaboost = AdaBoostClassifier(random_state=42)
    
    # Use GridSearchCV to find best parameters
    grid_search = GridSearchCV(
        estimator=adaboost,
        param_grid=param_grid,
        cv=cv,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Retrain with best parameters
    final_model = AdaBoostClassifier(**best_params, random_state=42)
    final_model.fit(X_train, y_train)
    
    # Save the model and parameters
    joblib.dump(final_model, 'models/adaboost_tuned.pkl')
    pd.DataFrame([best_params]).to_csv('results/adaboost_best_params.csv', index=False)
    
    return final_model


# -- Hyperparameter tuning for XGBoost --
def tune_xgboost(X_train, y_train, features, cv=5):
    """Tune hyperparameters for XGBoost classifier"""
    
    print("Tuning XGBoost hyperparameters...")
    
    # Map target values from {-1, 0, 1} to {0, 1, 2}
    target_mapping = {-1: 0, 0: 1, 1: 2}
    y_train_mapped = np.array([target_mapping.get(label, 0) for label in y_train])
    
    # Define parameter grid
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200, 300],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5]
    }
    
    # Fixed parameters
    fixed_params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'seed': 42
    }
    
    # Create base model
    xgb_model = xgb.XGBClassifier(**fixed_params)
    
    # Use RandomizedSearchCV to find best parameters
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=20,
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train_mapped)
    
    best_params = random_search.best_params_
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    # Combine best parameters with fixed parameters
    final_params = {**best_params, **fixed_params}
    
    # Retrain with best parameters
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_train, y_train_mapped)
    
    # Save the model and parameters
    joblib.dump(final_model, 'models/xgboost_tuned.pkl')
    pd.DataFrame([best_params]).to_csv('results/xgboost_best_params.csv', index=False)
    
    return final_model, target_mapping


# -- Evaluate the model --
def evaluate_model(model, X_test, y_test, model_name, features, target_mapping=None):
    """Evaluate model performance"""

    print(f"\nEvaluating {model_name}...")
    
    # Predict
    if model_name == 'XGBoost':
        y_pred_mapped = model.predict(X_test).astype(int)
        
        # Convert back from mapped values to original values
        mapping_reverse = {0: -1, 1: 0, 2: 1}
        y_pred = np.array([mapping_reverse.get(label, 0) for label in y_pred_mapped])
    else:
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
    cm_df.to_csv(f'results/{model_name.lower().replace(" ", "_")}_tuned_confusion.csv')
    
    # Save predictions for later analysis
    pred_df = pd.DataFrame({
        'true': y_test,
        'pred': y_pred
    })
    pred_df.to_csv(f'results/{model_name.lower().replace(" ", "_")}_tuned_predictions.csv')
    
    # For logistic regression, save coefficients
    if isinstance(model, LogisticRegression):
        if hasattr(model, 'coef_'):
            coef_df = pd.DataFrame({
                'Feature': features,
                'Coefficient': model.coef_[0] if model.coef_.shape[0] == 1 else model.coef_[0]
            })
            coef_df.to_csv(f'results/logistic_regression_tuned_coefficients.csv', index=False)
    
    # For Random Forest, save feature importance
    if model_name == 'Random Forest':
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        importance_df.to_csv(f'results/random_forest_tuned_feature_importance.csv', index=False)
    
    # For AdaBoost, save feature importance
    elif model_name == 'AdaBoost':
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        importance_df.to_csv(f'results/adaboost_tuned_feature_importance.csv', index=False)

    # For XGBoost, save feature importance
    elif model_name == 'XGBoost':
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        importance_df.to_csv(f'results/xgboost_tuned_feature_importance.csv', index=False)

    return {
        'accuracy': accuracy,
        'report': report
    }


def main():
    print("Starting modeling with hyperparameter tuning...")
    start_time = time.time()
    
    df = load_features()
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    X_train, X_test, y_train, y_test, features, scaler = prepare_data(df)
    if X_train is None:
        print("Data preparation failed. Exiting.")
        return
    
    # Save the scaler for future use
    joblib.dump(scaler, 'models/standard_scaler.pkl')
    
    model_results = {}
    
    print("\n--- Random Baseline ---")
    model_results['Random Baseline'] = random_baseline_model(y_test)
    
    print("\n--- Logistic Regression with Hyperparameter Tuning ---")
    lr_model = tune_logistic_regression(X_train, y_train)
    model_results['Logistic Regression'] = evaluate_model(lr_model, X_test, y_test, 'Logistic Regression', features)
    
    print("\n--- Random Forest with Hyperparameter Tuning ---")
    rf_model = tune_random_forest(X_train, y_train, features)
    model_results['Random Forest'] = evaluate_model(rf_model, X_test, y_test, 'Random Forest', features)
    
    print("\n--- AdaBoost with Hyperparameter Tuning ---")
    adaboost_model = tune_adaboost(X_train, y_train, features)
    model_results['AdaBoost'] = evaluate_model(adaboost_model, X_test, y_test, 'AdaBoost', features)

    print("\n--- XGBoost with Hyperparameter Tuning ---")
    xgb_model, target_mapping = tune_xgboost(X_train, y_train, features)
    model_results['XGBoost'] = evaluate_model(xgb_model, X_test, y_test, 'XGBoost', features, target_mapping)
    
    # Compare all models
    results_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'Accuracy': [r['accuracy'] for r in model_results.values()]
    })
    results_df.to_csv('results/tuned_model_comparison.csv', index=False)
    
    end_time = time.time()
    print(f"\nModeling complete! Total time: {(end_time - start_time)/60:.2f} minutes")


if __name__ == "__main__":
    main()