import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import joblib

def train_models(X_train, y_train, models_to_train):
    """
    Train selected machine learning models on the training data.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training target
        models_to_train (list): List of model names to train
        
    Returns:
        dict: Dictionary of trained models
    """
    trained_models = {}
    
    # Ensure target is always integer type for classification
    y_train = np.asarray(y_train, dtype=int)
    
    for model_name in models_to_train:
        if model_name == "decision_tree":
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, y_train)
            trained_models["decision_tree"] = model
            
        elif model_name == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            trained_models["random_forest"] = model
            
        elif model_name == "logistic_regression":
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            trained_models["logistic_regression"] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test, feature_names):
    """
    Evaluate trained models on the test data.
    
    Args:
        models (dict): Dictionary of trained models
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test target
        feature_names (list): List of feature names
        
    Returns:
        dict: Dictionary of model evaluation results
    """
    results = {}
    
    # Ensure test target is integer type for evaluation metrics
    y_test = np.asarray(y_test, dtype=int)
    
    for model_name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Check if we need to use multiclass metrics
        unique_classes = np.unique(y_test)
        if len(unique_classes) > 2:
            # Use weighted average for multiclass problems
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        else:
            # Binary classification metrics
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC curve and AUC - adapted for multiclass if needed
        try:
            # Try binary ROC calculation first
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
        except (ValueError, IndexError):
            # Fallback for multiclass: use first class vs rest approach
            # Create a binarized version for ROC curve
            binary_y_test = np.zeros_like(y_test)
            binary_y_test[y_test == unique_classes[0]] = 1
            
            # Use the probability of the first class
            fpr, tpr, thresholds = roc_curve(binary_y_test, y_pred_proba[:, 0])
            roc_auc = auc(fpr, tpr)
        
        # Get feature importance
        try:
            if model_name in ["decision_tree", "random_forest"]:
                feature_importance = model.feature_importances_
            elif model_name == "logistic_regression":
                # Check if logistic regression is multi-class
                if len(model.coef_) > 1:
                    # For multi-class, take the mean absolute value of coefficients across all classes
                    feature_importance = np.mean(np.abs(model.coef_), axis=0)
                else:
                    feature_importance = np.abs(model.coef_[0])
            else:
                # For models that don't have built-in feature importance,
                # create a placeholder with equal importance
                feature_importance = np.ones(X_test.shape[1]) / X_test.shape[1]
        except (AttributeError, TypeError):
            # Fallback for any model without feature importance attribute
            feature_importance = np.ones(X_test.shape[1]) / X_test.shape[1]
        
        # Store results
        results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc,
            "feature_importance": feature_importance,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba
        }
    
    return results

def predict_health_status(model, input_data):
    """
    Predict health status using a trained model.
    
    Args:
        model: Trained machine learning model
        input_data (numpy.ndarray): Input features for prediction
        
    Returns:
        tuple: Prediction and probability
    """
    # Make prediction
    prediction = model.predict(input_data)
    
    # Get prediction probability
    probability = model.predict_proba(input_data)
    
    return prediction, probability

def save_model(model, model_path):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained machine learning model
        model_path (str): Path to save the model
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        joblib.dump(model, model_path)
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def load_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        object: Loaded model
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
