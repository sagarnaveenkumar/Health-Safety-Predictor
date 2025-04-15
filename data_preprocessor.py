import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df, handle_missing=True, encode_categorical=True, normalize_data=True):
    """
    Preprocess the input dataframe for machine learning.
    
    Args:
        df (pandas.DataFrame): The input dataframe to preprocess
        handle_missing (bool): Whether to handle missing values
        encode_categorical (bool): Whether to encode categorical variables
        normalize_data (bool): Whether to normalize numerical features
        
    Returns:
        pandas.DataFrame: The preprocessed dataframe
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Identify numerical and categorical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle missing values
    if handle_missing:
        # For numerical columns, fill with median
        if numerical_cols:
            for col in numerical_cols:
                if data[col].isnull().sum() > 0:
                    data[col] = data[col].fillna(data[col].median())
        
        # For categorical columns, fill with mode
        if categorical_cols:
            for col in categorical_cols:
                if data[col].isnull().sum() > 0:
                    data[col] = data[col].fillna(data[col].mode()[0])
    
    # Encode categorical variables
    if encode_categorical and categorical_cols:
        for col in categorical_cols:
            # For binary categorical features
            if data[col].nunique() == 2:
                # Use Label Encoder for binary features
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
            else:
                # Use One-Hot Encoding for categorical features with more than 2 categories
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                data = pd.concat([data, dummies], axis=1)
                data = data.drop(col, axis=1)
    
    # Normalize numerical features
    if normalize_data and numerical_cols:
        scaler = StandardScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data

def get_feature_target_split(df, target_column):
    """
    Split the dataframe into features and target.
    
    Args:
        df (pandas.DataFrame): The input dataframe
        target_column (str): The name of the target column
        
    Returns:
        tuple: X (features), y (target), feature_names (list of feature names)
    """
    # Make sure target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataframe")
    
    # Get feature matrix and target vector
    X = df.drop(target_column, axis=1)
    
    # Explicitly convert target to integer for classification
    y = df[target_column].astype(int)
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    return X.values, y.values, feature_names
