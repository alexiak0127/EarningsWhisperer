# neural_network_model.py
# Enhances the modeling.py file with neural network implementations for FFNN and CNN

import pandas as pd
import numpy as np
import os
import json
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from modeling import load_features, prepare_data as prepare_base_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("neural_network.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import neural network libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
except ImportError as e:
    logger.error(f"Failed to import TensorFlow: {e}")
    raise

# Create results and model directories
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results/neural_networks', exist_ok=True)

# GPU memory config - only apply if GPUs are available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU memory growth set to True for {len(gpus)} GPUs")
    except RuntimeError as e:
        logger.warning(f"Error setting GPU memory growth: {e}")

# Prepare data specifically for neural networks
def prepare_data_for_nn(df):
    
    logger.info("Preparing data for neural networks")
    
    if df is None or df.empty:
        logger.error("Input DataFrame is None or empty")
        return None, None, None, None, None, None
    
    # First use the standard data preparation from modeling.py
    try:
        X_train, X_test, y_train, y_test, features, scaler = prepare_base_data(df)
    except Exception as e:
        logger.error(f"Error in base data preparation: {e}")
        return None, None, None, None, None, None
    
    if X_train is None or y_train is None:
        logger.error("Base data preparation returned None values")
        return None, None, None, None, None, None
    
    # Check the distribution of target values
    unique_values = np.unique(y_train)
    logger.info(f"Unique target values in training set: {unique_values}")
    
    # Determine target mapping based on unique values
    # Default is assuming values are -1, 0, 1 for down, stable, up
    if set(unique_values) == {-1, 0, 1}:
        target_mapping = {-1: 0, 0: 1, 1: 2}
        logger.info("Using default target mapping: {-1: 0, 0: 1, 1: 2}")
    else:
        # Create a mapping from existing values to sequential integers
        target_mapping = {val: idx for idx, val in enumerate(sorted(unique_values))}
        logger.info(f"Created custom target mapping: {target_mapping}")
    
    # Convert target to categorical for neural networks
    try:
        y_train_categorical = np.array([target_mapping[y] for y in y_train])
        y_test_categorical = np.array([target_mapping[y] for y in y_test])
        logger.info("Target values converted to categorical")
    except KeyError as e:
        logger.error(f"Error converting targets to categorical: {e}")
        logger.error(f"Target values not in mapping: {unique_values}")
        logger.error(f"Target mapping: {target_mapping}")
        return None, None, None, None, None, None
    
    return X_train, X_test, y_train_categorical, y_test_categorical, features, target_mapping


#  Build a Feed-Forward Neural Network model for stock movement classification
def build_ffnn_model(input_shape, num_classes=3, dropout_rate=0.3):
    
    logger.info(f"Building FFNN model with input shape {input_shape}")
    
    model = Sequential([
        # Input layer
        Dense(64, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Hidden layers
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Output layer - num_classes classes
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"FFNN model compiled with {model.count_params()} total parameters")
    return model


#  Build a 1D CNN model for stock movement classification
def build_cnn_model(input_shape, num_classes=3, dropout_rate=0.3):
    
    # Reshape input for CNN
    input_reshape = (input_shape, 1)
    logger.info(f"Building CNN model with input shape {input_reshape}")
    
    model = Sequential([
        # CNN layers
        Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=input_reshape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=64, kernel_size=2, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # Flatten before dense layers
        Flatten(),
        
        # Dense layers
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Output layer - num_classes classes
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"CNN model compiled with {model.count_params()} total parameters")
    return model


# Train a neural network model
def train_neural_network(X_train, y_train, X_test, y_test, model_type='ffnn', batch_size=32, epochs=50, num_classes=3):

    logger.info(f"Training {model_type} model with batch size {batch_size} and max {epochs} epochs")
    
    # Set up callbacks for training
    model_save_path = f'models/{model_type}_best_model.h5'
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(model_save_path, save_best_only=True)
    ]
    
    # Build the appropriate model based on type
    if model_type == 'ffnn':
        model = build_ffnn_model(X_train.shape[1], num_classes=num_classes)
        X_train_input = X_train
        X_test_input = X_test
    
    elif model_type == 'cnn':
        # Reshape input for CNN
        X_train_input = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_input = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        model = build_cnn_model(X_train.shape[1], num_classes=num_classes)
    
    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type} - must be 'ffnn' or 'cnn'")
    
    # Train the model
    logger.info(f"Starting training for {model_type} model...")
    history = model.fit(
        X_train_input, y_train,
        validation_data=(X_test_input, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_input, y_test)
    logger.info(f"{model_type} Test Loss: {loss:.4f}")
    logger.info(f"{model_type} Test Accuracy: {accuracy:.4f}")
    
    # Save the model history for visualization
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],  # Convert np.float32 to Python float
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']], 
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    
    with open(f'results/neural_networks/{model_type}_history.json', 'w') as f:
        json.dump(history_dict, f)
    
    # Save model architecture as JSON
    model_json = model.to_json()
    with open(f'models/{model_type}_architecture.json', 'w') as f:
        f.write(model_json)
    
    # Save entire model
    model.save(f'models/{model_type}_full_model.h5')
    
    return model, history


def evaluate_neural_network(model, X_test, y_test, model_type='ffnn', inverse_mapping=None):
    
    logger.info(f"Evaluating {model_type} model")
    
    # Prepare input based on model type
    if model_type == 'ffnn':
        X_test_input = X_test
    elif model_type == 'cnn':
        X_test_input = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type} - must be 'ffnn' or 'cnn'")
    
    # Default inverse mapping if none provided
    if inverse_mapping is None:
        inverse_mapping = {0: -1, 1: 0, 2: 1}
        logger.warning("No inverse mapping provided, using default {0: -1, 1: 0, 2: 1}")
    
    # Make predictions
    try:
        y_pred_prob = model.predict(X_test_input)
        y_pred = np.argmax(y_pred_prob, axis=1)
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return None
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get class names based on inverse mapping
    sorted_classes = sorted(inverse_mapping.keys())
    class_names = [str(inverse_mapping[c]) for c in sorted_classes]
    
    # Generate detailed classification report
    try:
        report = classification_report(y_test, y_pred, target_names=class_names)
        report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    except Exception as e:
        logger.error(f"Error generating classification report: {e}")
        report = "Error generating report"
        report_dict = {}
    
    # Create confusion matrix
    try:
        cm = confusion_matrix(y_test, y_pred)
    except Exception as e:
        logger.error(f"Error generating confusion matrix: {e}")
        cm = np.array([])
    
    logger.info(f"\n{model_type} Neural Network Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(report)
    
    # Save confusion matrix
    try:
        cm_df = pd.DataFrame(
            cm,
            columns=[f'Pred_{c}' for c in class_names],
            index=[f'True_{c}' for c in class_names]
        )
        cm_df.to_csv(f'results/neural_networks/{model_type}_confusion.csv')
    except Exception as e:
        logger.error(f"Error saving confusion matrix: {e}")
    
    # Convert numerical predictions back to original categories
    try:
        y_pred_original = np.array([inverse_mapping.get(p, p) for p in y_pred])
        y_test_original = np.array([inverse_mapping.get(t, t) for t in y_test])
        
        # Save predictions
        pred_df = pd.DataFrame({
            'true': y_test_original,
            'pred': y_pred_original
        })
        pred_df.to_csv(f'results/neural_networks/{model_type}_predictions.csv')
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
    
    # Save prediction probabilities
    try:
        prob_df = pd.DataFrame(
            y_pred_prob,
            columns=[f'Prob_{c}' for c in class_names]
        )
        prob_df.to_csv(f'results/neural_networks/{model_type}_probabilities.csv')
    except Exception as e:
        logger.error(f"Error saving probabilities: {e}")
    
    # Save classification report
    with open(f'results/neural_networks/{model_type}_classification_report.json', 'w') as f:
        json.dump(report_dict, f)
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm
    }

def calculate_feature_importance(model, X_test, y_test, features, model_type='ffnn', n_repeats=5):
   
    logger.info(f"Calculating feature importance for {model_type} model")
    
    if model is None or X_test is None or y_test is None or features is None:
        logger.error("Cannot calculate feature importance: model, data, or features are None")
        return None
    
    try:
        from sklearn.inspection import permutation_importance
    except ImportError as e:
        logger.error(f"Failed to import permutation_importance: {e}")
        return None
    
    # Prepare a wrapper function for sklearn permutation_importance
    def model_predict_proba(X):
        try:
            # Reshape if necessary
            if model_type == 'cnn':
                X = X.reshape(X.shape[0], X.shape[1], 1)
            return model.predict(X)
        except Exception as e:
            logger.error(f"Error in prediction for permutation importance: {e}")
            return None
    
    # Use a smaller subset for large datasets to reduce computation time
    max_samples = 1000
    if len(X_test) > max_samples:
        logger.info(f"Using {max_samples} samples for feature importance to reduce computation time")
        indices = np.random.choice(len(X_test), max_samples, replace=False)
        X_subset = X_test[indices]
        y_subset = y_test[indices]
    else:
        X_subset = X_test
        y_subset = y_test
    
    # Calculate permutation importance with fewer repeats for efficiency
    try:
        logger.info(f"Running permutation importance with {n_repeats} repeats")
        result = permutation_importance(
            model_predict_proba, X_subset, y_subset,
            n_repeats=n_repeats,
            random_state=42,
            scoring='accuracy',
            n_jobs=-1  # Use all available cores
        )
    except Exception as e:
        logger.error(f"Error calculating permutation importance: {e}")
        return None
    
    # Handle features list length mismatch
    if len(features) != X_test.shape[1]:
        logger.error(f"Feature list length ({len(features)}) doesn't match X_test shape ({X_test.shape[1]})")
        features = [f"feature_{i}" for i in range(X_test.shape[1])]
    
    # Save feature importance data
    try:
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': result.importances_mean,
            'Importance_Std': result.importances_std
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        importance_df.to_csv(f'results/neural_networks/{model_type}_feature_importance.csv', index=False)
        logger.info(f"Feature importance saved to results/neural_networks/{model_type}_feature_importance.csv")
    except Exception as e:
        logger.error(f"Error saving feature importance: {e}")
        return None
    
    return importance_df

# neural_network_model.py
# Enhances the modeling.py file with neural network implementations for FFNN and CNN

import pandas as pd
import numpy as np
import os
import json
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from modeling import load_features, prepare_data as prepare_base_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("neural_network.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import neural network libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
except ImportError as e:
    logger.error(f"Failed to import TensorFlow: {e}")
    raise

# Create results and model directories
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results/neural_networks', exist_ok=True)

# GPU memory config - only apply if GPUs are available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU memory growth set to True for {len(gpus)} GPUs")
    except RuntimeError as e:
        logger.warning(f"Error setting GPU memory growth: {e}")

# Prepare data specifically for neural networks
def prepare_data_for_nn(df):
    
    logger.info("Preparing data for neural networks")
    
    if df is None or df.empty:
        logger.error("Input DataFrame is None or empty")
        return None, None, None, None, None, None
    
    # First use the standard data preparation from modeling.py
    try:
        X_train, X_test, y_train, y_test, features, scaler = prepare_base_data(df)
    except Exception as e:
        logger.error(f"Error in base data preparation: {e}")
        return None, None, None, None, None, None
    
    if X_train is None or y_train is None:
        logger.error("Base data preparation returned None values")
        return None, None, None, None, None, None
    
    # Check the distribution of target values
    unique_values = np.unique(y_train)
    logger.info(f"Unique target values in training set: {unique_values}")
    
    # Determine target mapping based on unique values
    # Default is assuming values are -1, 0, 1 for down, stable, up
    if set(unique_values) == {-1, 0, 1}:
        target_mapping = {-1: 0, 0: 1, 1: 2}
        logger.info("Using default target mapping: {-1: 0, 0: 1, 1: 2}")
    else:
        # Create a mapping from existing values to sequential integers
        target_mapping = {val: idx for idx, val in enumerate(sorted(unique_values))}
        logger.info(f"Created custom target mapping: {target_mapping}")
    
    # Convert target to categorical for neural networks
    try:
        y_train_categorical = np.array([target_mapping[y] for y in y_train])
        y_test_categorical = np.array([target_mapping[y] for y in y_test])
        logger.info("Target values converted to categorical")
    except KeyError as e:
        logger.error(f"Error converting targets to categorical: {e}")
        logger.error(f"Target values not in mapping: {unique_values}")
        logger.error(f"Target mapping: {target_mapping}")
        return None, None, None, None, None, None
    
    return X_train, X_test, y_train_categorical, y_test_categorical, features, target_mapping


#  Build a Feed-Forward Neural Network model for stock movement classification
def build_ffnn_model(input_shape, num_classes=3, dropout_rate=0.3):
    
    logger.info(f"Building FFNN model with input shape {input_shape}")
    
    model = Sequential([
        # Input layer
        Dense(64, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Hidden layers
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Output layer - num_classes classes
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"FFNN model compiled with {model.count_params()} total parameters")
    return model


#  Build a 1D CNN model for stock movement classification
def build_cnn_model(input_shape, num_classes=3, dropout_rate=0.3):
    
    # Reshape input for CNN
    input_reshape = (input_shape, 1)
    logger.info(f"Building CNN model with input shape {input_reshape}")
    
    model = Sequential([
        # CNN layers
        Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=input_reshape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=64, kernel_size=2, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # Flatten before dense layers
        Flatten(),
        
        # Dense layers
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Output layer - num_classes classes
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"CNN model compiled with {model.count_params()} total parameters")
    return model


# Train a neural network model
def train_neural_network(X_train, y_train, X_test, y_test, model_type='ffnn', batch_size=32, epochs=50, num_classes=3):

    logger.info(f"Training {model_type} model with batch size {batch_size} and max {epochs} epochs")
    
    # Set up callbacks for training
    model_save_path = f'models/{model_type}_best_model.h5'
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(model_save_path, save_best_only=True)
    ]
    
    # Build the appropriate model based on type
    if model_type == 'ffnn':
        model = build_ffnn_model(X_train.shape[1], num_classes=num_classes)
        X_train_input = X_train
        X_test_input = X_test
    
    elif model_type == 'cnn':
        # Reshape input for CNN
        X_train_input = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_input = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        model = build_cnn_model(X_train.shape[1], num_classes=num_classes)
    
    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type} - must be 'ffnn' or 'cnn'")
    
    # Train the model
    logger.info(f"Starting training for {model_type} model...")
    history = model.fit(
        X_train_input, y_train,
        validation_data=(X_test_input, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_input, y_test)
    logger.info(f"{model_type} Test Loss: {loss:.4f}")
    logger.info(f"{model_type} Test Accuracy: {accuracy:.4f}")
    
    # Save the model history for visualization
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],  # Convert np.float32 to Python float
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']], 
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    
    with open(f'results/neural_networks/{model_type}_history.json', 'w') as f:
        json.dump(history_dict, f)
    
    # Save model architecture as JSON
    model_json = model.to_json()
    with open(f'models/{model_type}_architecture.json', 'w') as f:
        f.write(model_json)
    
    # Save entire model
    model.save(f'models/{model_type}_full_model.h5')
    
    return model, history


def evaluate_neural_network(model, X_test, y_test, model_type='ffnn', inverse_mapping=None):
    
    logger.info(f"Evaluating {model_type} model")
    
    # Prepare input based on model type
    if model_type == 'ffnn':
        X_test_input = X_test
    elif model_type == 'cnn':
        X_test_input = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type} - must be 'ffnn' or 'cnn'")
    
    # Default inverse mapping if none provided
    if inverse_mapping is None:
        inverse_mapping = {0: -1, 1: 0, 2: 1}
        logger.warning("No inverse mapping provided, using default {0: -1, 1: 0, 2: 1}")
    
    # Make predictions
    try:
        y_pred_prob = model.predict(X_test_input)
        y_pred = np.argmax(y_pred_prob, axis=1)
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return None
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get class names based on inverse mapping
    sorted_classes = sorted(inverse_mapping.keys())
    class_names = [str(inverse_mapping[c]) for c in sorted_classes]
    
    # Generate detailed classification report
    try:
        report = classification_report(y_test, y_pred, target_names=class_names)
        report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    except Exception as e:
        logger.error(f"Error generating classification report: {e}")
        report = "Error generating report"
        report_dict = {}
    
    # Create confusion matrix
    try:
        cm = confusion_matrix(y_test, y_pred)
    except Exception as e:
        logger.error(f"Error generating confusion matrix: {e}")
        cm = np.array([])
    
    logger.info(f"\n{model_type} Neural Network Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(report)
    
    # Save confusion matrix
    try:
        cm_df = pd.DataFrame(
            cm,
            columns=[f'Pred_{c}' for c in class_names],
            index=[f'True_{c}' for c in class_names]
        )
        cm_df.to_csv(f'results/neural_networks/{model_type}_confusion.csv')
    except Exception as e:
        logger.error(f"Error saving confusion matrix: {e}")
    
    # Convert numerical predictions back to original categories
    try:
        y_pred_original = np.array([inverse_mapping.get(p, p) for p in y_pred])
        y_test_original = np.array([inverse_mapping.get(t, t) for t in y_test])
        
        # Save predictions
        pred_df = pd.DataFrame({
            'true': y_test_original,
            'pred': y_pred_original
        })
        pred_df.to_csv(f'results/neural_networks/{model_type}_predictions.csv')
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
    
    # Save prediction probabilities
    try:
        prob_df = pd.DataFrame(
            y_pred_prob,
            columns=[f'Prob_{c}' for c in class_names]
        )
        prob_df.to_csv(f'results/neural_networks/{model_type}_probabilities.csv')
    except Exception as e:
        logger.error(f"Error saving probabilities: {e}")
    
    # Save classification report
    with open(f'results/neural_networks/{model_type}_classification_report.json', 'w') as f:
        json.dump(report_dict, f)
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm
    }

def calculate_feature_importance(model, X_test, y_test, features, model_type='ffnn', n_repeats=5):
   
    logger.info(f"Calculating feature importance for {model_type} model")
    
    if model is None or X_test is None or y_test is None or features is None:
        logger.error("Cannot calculate feature importance: model, data, or features are None")
        return None
    
    try:
        from sklearn.inspection import permutation_importance
    except ImportError as e:
        logger.error(f"Failed to import permutation_importance: {e}")
        return None
    
    # Prepare a wrapper function for sklearn permutation_importance
    def model_predict_proba(X):
        try:
            # Reshape if necessary
            if model_type == 'cnn':
                X = X.reshape(X.shape[0], X.shape[1], 1)
            return model.predict(X)
        except Exception as e:
            logger.error(f"Error in prediction for permutation importance: {e}")
            return None
    
    # Use a smaller subset for large datasets to reduce computation time
    max_samples = 1000
    if len(X_test) > max_samples:
        logger.info(f"Using {max_samples} samples for feature importance to reduce computation time")
        indices = np.random.choice(len(X_test), max_samples, replace=False)
        X_subset = X_test[indices]
        y_subset = y_test[indices]
    else:
        X_subset = X_test
        y_subset = y_test
    
    # Calculate permutation importance with fewer repeats for efficiency
    try:
        logger.info(f"Running permutation importance with {n_repeats} repeats")
        result = permutation_importance(
            model_predict_proba, X_subset, y_subset,
            n_repeats=n_repeats,
            random_state=42,
            scoring='accuracy',
            n_jobs=-1  # Use all available cores
        )
    except Exception as e:
        logger.error(f"Error calculating permutation importance: {e}")
        return None
    
    # Handle features list length mismatch
    if len(features) != X_test.shape[1]:
        logger.error(f"Feature list length ({len(features)}) doesn't match X_test shape ({X_test.shape[1]})")
        features = [f"feature_{i}" for i in range(X_test.shape[1])]
    
    # Save feature importance data
    try:
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': result.importances_mean,
            'Importance_Std': result.importances_std
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        importance_df.to_csv(f'results/neural_networks/{model_type}_feature_importance.csv', index=False)
        logger.info(f"Feature importance saved to results/neural_networks/{model_type}_feature_importance.csv")
    except Exception as e:
        logger.error(f"Error saving feature importance: {e}")
        return None
    
    return importance_df

if __name__ == "__main__":
    logger.info("Starting neural network model training")
    
    # Load features
    df = load_features()
    
    if df is not None:
        # Prepare data
        X_train, X_test, y_train, y_test, features, target_mapping = prepare_data_for_nn(df)
        
        if X_train is not None and y_train is not None:
            # Create inverse mapping for evaluation
            inverse_mapping = {v: k for k, v in target_mapping.items()}
            
            # Set common training parameters
            batch_size = 32
            epochs = 50
            
            # 1. Train and evaluate FFNN model
            logger.info("=== Feed-Forward Neural Network (FFNN) ===")
            ffnn_model, ffnn_history = train_neural_network(
                X_train, y_train, X_test, y_test, 
                model_type='ffnn', 
                batch_size=batch_size, 
                epochs=epochs
            )
            
            ffnn_results = evaluate_neural_network(
                ffnn_model, X_test, y_test, 
                model_type='ffnn', 
                inverse_mapping=inverse_mapping
            )
            
            ffnn_importance = calculate_feature_importance(
                ffnn_model, X_test, y_test, 
                features, 
                model_type='ffnn'
            )
            
            # 2. Train and evaluate CNN model
            logger.info("=== Convolutional Neural Network (CNN) ===")
            cnn_model, cnn_history = train_neural_network(
                X_train, y_train, X_test, y_test, 
                model_type='cnn', 
                batch_size=batch_size, 
                epochs=epochs
            )
            
            cnn_results = evaluate_neural_network(
                cnn_model, X_test, y_test, 
                model_type='cnn', 
                inverse_mapping=inverse_mapping
            )
            
            cnn_importance = calculate_feature_importance(
                cnn_model, X_test, y_test, 
                features, 
                model_type='cnn'
            )
            
            # 3. Compare model performance
            logger.info("=== Model Comparison ===")
            if ffnn_results and cnn_results:
                ffnn_acc = ffnn_results.get('accuracy', 0)
                cnn_acc = cnn_results.get('accuracy', 0)
                
                logger.info(f"FFNN Accuracy: {ffnn_acc:.4f}")
                logger.info(f"CNN Accuracy: {cnn_acc:.4f}")
                
                if ffnn_acc > cnn_acc:
                    logger.info("FFNN model performed better")
                elif cnn_acc > ffnn_acc:
                    logger.info("CNN model performed better")
                else:
                    logger.info("Both models performed equally")
            
            logger.info("Neural network model training and evaluation completed")
        else:
            logger.error("Failed to prepare data for neural networks")
    else:
        logger.error("Failed to load features data")
