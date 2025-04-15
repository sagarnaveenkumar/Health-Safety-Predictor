import pandas as pd
import numpy as np
import streamlit as st
from sklearn.datasets import fetch_openml
import os
import io
import requests
import zipfile
import time
from pathlib import Path

# Create a data directory if it doesn't exist
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

def get_available_datasets():
    """
    Returns a list of available healthcare datasets.
    
    Returns:
        list: List of available dataset names
    """
    return [
        "Heart Disease UCI",
        "Diabetes",
        "Stroke Prediction",
        "Cardiovascular Disease",
        "Body Signal of Smoking"
    ]

def download_dataset(url, file_path, sep=',', header='infer', names=None, na_values=None, show_warning=True):
    """
    Downloads a dataset from a URL and saves it locally.
    
    Args:
        url (str): URL to download dataset from
        file_path (str): Path to save the dataset
        sep (str): Separator for CSV file
        header (str or int): Header row for CSV file
        names (list): Column names
        na_values: Values to treat as NaN
        show_warning (bool): Whether to show a warning if download fails
        
    Returns:
        pandas.DataFrame or None: Downloaded dataset or None if failed
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Save the raw data to file
        with open(file_path, 'wb') as f:
            f.write(response.content)
            
        # Read the data into a DataFrame with proper handling of parameters
        # Create a dictionary of parameters and filter out None values
        params = {}
        
        # Always include the separator
        params['sep'] = sep
        
        # Add other parameters if they're not None
        if names is not None:
            params['names'] = names
            
        if na_values is not None:
            params['na_values'] = na_values
            
        # Special handling for header parameter since it can be an integer or string
        # Try to use the provided header value, defaulting to 'infer' if there's an issue
        try:
            if header == 'infer':
                params['header'] = 'infer'
            elif header is None:  # No header in file
                params['header'] = None
            else:  # Try to use as int
                params['header'] = int(header)
        except:
            params['header'] = 'infer'  # Default to infer if conversion fails
            
        df = pd.read_csv(file_path, **params)
        return df
        
    except Exception as e:
        if show_warning:
            st.warning(f"Failed to download dataset from {url}: {e}")
        return None

@st.cache_data
def load_dataset(dataset_name):
    """
    Loads a healthcare dataset based on the provided name.
    
    Args:
        dataset_name (str): Name of the dataset to load
    
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    # Standardize dataset naming for filenames
    filename_map = {
        "Heart Disease UCI": "heart_disease",
        "Diabetes": "diabetes",
        "Stroke Prediction": "stroke",
        "Cardiovascular Disease": "cardiovascular",
        "Body Signal of Smoking": "smoking"
    }
    
    

    # Check for preloaded datasets first
    if dataset_name in filename_map:
        # Try loading the preprocessed dataset if it exists
        preprocessed_path = DATA_DIR / f"{filename_map[dataset_name]}_preprocessed.csv"
        if preprocessed_path.exists():
            try:
                st.info(f"Loading preprocessed {dataset_name} dataset from local storage...")
                df = pd.read_csv(preprocessed_path)
                # Ensure we don't have any string or object columns for Arrow compatibility
                for col in df.select_dtypes(include=['object']).columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                    except:
                        # If can't convert to numeric, convert to categorical codes
                        df[col] = df[col].astype('category').cat.codes.astype(int)
                return df
            except Exception as e:
                st.warning(f"Error loading preprocessed dataset: {e}. Downloading fresh data...")
                
    # Load dataset based on name
    if dataset_name == "Heart Disease UCI":
        return load_heart_disease_dataset()
    elif dataset_name == "Diabetes":
        return load_diabetes_dataset()
    elif dataset_name == "Stroke Prediction":
        return load_stroke_dataset()
    elif dataset_name == "Cardiovascular Disease":
        return load_cardiovascular_dataset()
    elif dataset_name == "Body Signal of Smoking":
        return load_smoking_dataset()
    else:
        raise ValueError(f"Dataset '{dataset_name}' not recognized")
    

def filter_patients_by_range(df, column, min_val, max_val):
    """
    Filter patients within a specific range for a given column.

    Args:
        df (pandas.DataFrame): The patient dataset.
        column (str): Column name to filter on.
        min_val (float): Minimum value of the range.
        max_val (float): Maximum value of the range.

    Returns:
        pandas.DataFrame: Filtered patient records.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")

    filtered_df = df[(df[column] >= min_val) & (df[column] <= max_val)]
    return filtered_df

def load_heart_disease_dataset():
    """
    Loads the Heart Disease UCI dataset.
    
    Returns:
        pandas.DataFrame: Heart disease dataset
    """
    # Setup file paths
    raw_file_path = DATA_DIR / "heart_disease_raw.csv"
    preprocessed_file_path = DATA_DIR / "heart_disease_preprocessed.csv"
    
    # Define column names
    column_names = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 
                    'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 'exercise_induced_angina', 
                    'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'heart_disease']
    
    # First, try to load preprocessed data
    if preprocessed_file_path.exists():
        try:
            df = pd.read_csv(preprocessed_file_path)
            st.success("✅ Heart Disease dataset loaded from local storage")
            return df
        except Exception as e:
            st.warning(f"Could not load preprocessed Heart Disease dataset: {e}")
    
    # Try to download data if needed
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "https://raw.githubusercontent.com/rohan-paul/MachineLearning-DeepLearning-Code-for-my-YouTube-Channel/master/Heart-Disease-UCI/processed.cleveland.data"
    ]
    
    df = None
    for i, url in enumerate(urls):
        try:
            # Try to download the dataset
            st.info(f"Downloading Heart Disease dataset from source {i+1}...")
            df = download_dataset(url, raw_file_path, header=0, names=column_names, na_values='?')
            if df is None:
                continue
                
            # Preprocess data
            # Handle missing values (? is already converted to NaN during loading)
            for col in df.select_dtypes(include=['float']).columns:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())
                    
            # Convert categorical variables to integers
            for col in ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'resting_ecg', 
                        'exercise_induced_angina', 'st_slope', 'thalassemia']:
                if col in df.columns:
                    df[col] = df[col].fillna(0).astype(int)
            
            # Convert target to binary (0 = no disease, 1 = disease)
            df['heart_disease'] = df['heart_disease'].apply(lambda x: 0 if x == 0 else 1)
            
            # Save preprocessed data for future use
            df.to_csv(preprocessed_file_path, index=False)
            st.success("✅ Heart Disease dataset downloaded and preprocessed successfully")
            
            return df
            
        except Exception as e:
            st.warning(f"Could not load Heart Disease dataset from source {i+1}: {e}")
            continue
    
    # If we get here, all sources failed
    st.error("❌ Failed to load Heart Disease dataset from all sources")
    # Return a minimal dataset with the correct columns so the app doesn't crash
    return pd.DataFrame(columns=column_names)

def load_diabetes_dataset():
    """
    Loads the Pima Indians Diabetes dataset.
    
    Returns:
        pandas.DataFrame: Diabetes dataset
    """
    # Setup file paths
    raw_file_path = DATA_DIR / "diabetes_raw.csv"
    preprocessed_file_path = DATA_DIR / "diabetes_preprocessed.csv"
    
    # Define column names
    column_names = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 
                    'bmi', 'diabetes_pedigree', 'age', 'diabetes']
                    
    # First, try to load preprocessed data
    if preprocessed_file_path.exists():
        try:
            df = pd.read_csv(preprocessed_file_path)
            st.success("✅ Diabetes dataset loaded from local storage")
            return df
        except Exception as e:
            st.warning(f"Could not load preprocessed Diabetes dataset: {e}")
    
    # Try to download data if needed
    urls = [
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
    ]
    
    df = None
    for i, url in enumerate(urls):
        try:
            # Try to download the dataset
            st.info(f"Downloading Diabetes dataset from source {i+1}...")
            if i == 0:  # First URL has no header
                df = download_dataset(url, raw_file_path, header=0, names=column_names)
            else:  # Second URL has header
                df = download_dataset(url, raw_file_path)
                
            if df is None:
                continue
                
            # Preprocess data
            # Handle missing or zero values in important columns
            # For diabetes dataset, zeros in certain medical measurements are likely missing values
            zero_columns = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
            for col in zero_columns:
                if col in df.columns:
                    # Replace zeros with median for these columns (zeros are physiologically implausible)
                    df[col] = df[col].replace(0, np.nan)
                    df[col] = df[col].fillna(df[col].median())
            
            # Ensure all columns are numeric
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        df[col] = df[col].fillna(df[col].median())
                    except:
                        # If can't convert to numeric, use categorical encoding
                        df[col] = df[col].astype('category').cat.codes
            
            # Ensure diabetes is treated as a binary classification target
            df['diabetes'] = df['diabetes'].astype(int)
            
            # Save preprocessed data for future use
            df.to_csv(preprocessed_file_path, index=False)
            st.success("✅ Diabetes dataset downloaded and preprocessed successfully")
            
            return df
            
        except Exception as e:
            st.warning(f"Could not load Diabetes dataset from source {i+1}: {e}")
            continue
    
    # If all sources fail, show error
    st.error("❌ Failed to load Diabetes dataset from all sources")
    return pd.DataFrame(columns=column_names)

def load_stroke_dataset():
    """
    Loads the Stroke Prediction dataset.
    
    Returns:
        pandas.DataFrame: Stroke dataset
    """
    # Setup file paths
    raw_file_path = DATA_DIR / "stroke_raw.csv"
    preprocessed_file_path = DATA_DIR / "stroke_preprocessed.csv"
    
    # First, try to load preprocessed data
    if preprocessed_file_path.exists():
        try:
            df = pd.read_csv(preprocessed_file_path)
            st.success("✅ Stroke dataset loaded from local storage")
            return df
        except Exception as e:
            st.warning(f"Could not load preprocessed Stroke dataset: {e}")
    
    # Try to download data if needed
    urls = [
        "https://raw.githubusercontent.com/fedesoriano/stroke-prediction-dataset/main/healthcare-dataset-stroke-data.csv",
        "https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/healthcare-dataset-stroke-data.csv"
    ]
    
    df = None
    for i, url in enumerate(urls):
        try:
            # Try to download the dataset
            st.info(f"Downloading Stroke dataset from source {i+1}...")
            df = download_dataset(url, raw_file_path)
            if df is None:
                continue
                
            # Preprocess data
            # Handle missing values in bmi column
            if 'bmi' in df.columns and df['bmi'].isna().any():
                # Fill missing BMI values with the median
                df['bmi'] = df['bmi'].fillna(df['bmi'].median())
                
            # Convert categorical variables to numeric for ML
            if 'gender' in df.columns:
                df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 2}).fillna(1).astype(int)
                
            if 'smoking_status' in df.columns:
                smoking_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3}
                df['smoking_status'] = df['smoking_status'].map(smoking_map).fillna(3).astype(int)
                
            if 'ever_married' in df.columns:
                df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
                
            if 'work_type' in df.columns:
                work_map = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
                df['work_type'] = df['work_type'].map(work_map).fillna(0).astype(int)
                
            if 'Residence_type' in df.columns:
                df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0}).fillna(0).astype(int)
            
            # Handle any remaining object columns
            for col in df.select_dtypes(include=['object']).columns:
                # Convert any remaining object columns to categorical codes
                df[col] = df[col].astype('category').cat.codes.astype(int)
            
            # Make sure numeric columns don't have NaN
            numeric_cols = df.select_dtypes(include=['float', 'int']).columns
            for col in numeric_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())
            
            # Ensure stroke is treated as an integer for classification
            df['stroke'] = df['stroke'].astype(int)
            
            # Save preprocessed data for future use
            df.to_csv(preprocessed_file_path, index=False)
            st.success("✅ Stroke dataset downloaded and preprocessed successfully")
            
            return df
            
        except Exception as e:
            st.warning(f"Could not load Stroke dataset from source {i+1}: {e}")
            continue
            
    # If all sources fail, provide error
    st.error("❌ Failed to load Stroke dataset from all sources")
    columns = ['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
              'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']
    return pd.DataFrame(columns=columns)

def load_cardiovascular_dataset():
    """
    Loads the Cardiovascular Disease dataset.
    
    Returns:
        pandas.DataFrame: Cardiovascular disease dataset
    """
    # Setup file paths
    raw_file_path = DATA_DIR / "cardiovascular_raw.csv"
    preprocessed_file_path = DATA_DIR / "cardiovascular_preprocessed.csv"
    
    # First, try to load preprocessed data
    if preprocessed_file_path.exists():
        try:
            df = pd.read_csv(preprocessed_file_path)
            st.success("✅ Cardiovascular dataset loaded from local storage")
            return df
        except Exception as e:
            st.warning(f"Could not load preprocessed Cardiovascular dataset: {e}")
    
    # Try to download data if needed
    urls = [
        "https://raw.githubusercontent.com/arunbalas/CardioVascularDisease/master/cardio_train.csv",
        "https://raw.githubusercontent.com/suhasghorp/CardioApp/master/cardio_train.csv"
    ]
    
    df = None
    for i, url in enumerate(urls):
        try:
            # Try to download the dataset
            st.info(f"Downloading Cardiovascular dataset from source {i+1}...")
            if i == 0:  # First source uses semicolon separator
                df = download_dataset(url, raw_file_path, sep=';')
            else:  # Try default comma separator for second source
                df = download_dataset(url, raw_file_path)  # Use default sep=','
                if df is not None:
                    # Try reading again with auto-detect separator if needed
                    try:
                        df = pd.read_csv(raw_file_path, engine='python')  # Python engine can auto-detect separators
                    except:
                        # If that fails, keep the original dataframe
                        pass
            
            if df is None:
                continue
                
            # Preprocess data
            # Rename columns for clarity
            rename_dict = {
                'age': 'age_days', 
                'gender': 'gender',
                'height': 'height_cm',
                'weight': 'weight_kg',
                'ap_hi': 'systolic_bp',
                'ap_lo': 'diastolic_bp',
                'cholesterol': 'cholesterol',
                'gluc': 'glucose',
                'smoke': 'smoker',
                'alco': 'alcohol',
                'active': 'physical_activity',
                'cardio': 'cardiovascular_disease'
            }
            
            # Only rename columns that exist
            rename_dict = {k: v for k, v in rename_dict.items() if k in df.columns}
            df = df.rename(columns=rename_dict)
            
            # Convert age from days to years if it exists
            if 'age_days' in df.columns:
                df['age_years'] = (df['age_days'] / 365).round(1)
            
            # Calculate BMI if height and weight exist
            if 'height_cm' in df.columns and 'weight_kg' in df.columns:
                # Replace zeros with NaN to avoid division by zero
                df['height_cm'] = df['height_cm'].replace(0, np.nan)
                df['bmi'] = (df['weight_kg'] / ((df['height_cm']/100) ** 2)).round(1)
                # Fill NaN values with median
                df['bmi'] = df['bmi'].fillna(df['bmi'].median())
                
            # Remove outliers if needed
            if 'systolic_bp' in df.columns:
                df = df[(df['systolic_bp'] > 50) & (df['systolic_bp'] < 250)]
            
            if 'diastolic_bp' in df.columns:
                df = df[(df['diastolic_bp'] > 30) & (df['diastolic_bp'] < 200)]
                
            # Ensure all columns are numeric
            for col in df.select_dtypes(include=['object']).columns:
                # Try to convert to numeric, if not possible use categorical encoding
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].median())
                except:
                    # If can't convert to numeric, use categorical encoding
                    df[col] = df[col].astype('category').cat.codes.astype(int)
                    
            # Make sure numeric columns don't have NaN
            numeric_cols = df.select_dtypes(include=['float', 'int']).columns
            for col in numeric_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())
                
            # Ensure cardiovascular_disease is treated as an integer for classification
            if 'cardiovascular_disease' in df.columns:
                df['cardiovascular_disease'] = df['cardiovascular_disease'].astype(int)
            
            # Save preprocessed data for future use
            df.to_csv(preprocessed_file_path, index=False)
            st.success("✅ Cardiovascular dataset downloaded and preprocessed successfully")
            
            return df
            
        except Exception as e:
            st.warning(f"Could not load Cardiovascular dataset from source {i+1}: {e}")
            continue
    
    # If all sources fail, provide error
    st.error("❌ Failed to load Cardiovascular dataset from all sources")
    columns = ['id', 'age_days', 'gender', 'height_cm', 'weight_kg', 'systolic_bp', 
              'diastolic_bp', 'cholesterol', 'glucose', 'smoker', 'alcohol', 
              'physical_activity', 'cardiovascular_disease', 'age_years', 'bmi']
    return pd.DataFrame(columns=columns)

def load_smoking_dataset():
    """
    Loads the Body Signal of Smoking dataset.
    
    Returns:
        pandas.DataFrame: Smoking dataset
    """
    # Setup file paths
    raw_file_path = DATA_DIR / "smoking_raw.csv"
    preprocessed_file_path = DATA_DIR / "smoking_preprocessed.csv"
    
    # First, try to load preprocessed data
    if preprocessed_file_path.exists():
        try:
            df = pd.read_csv(preprocessed_file_path)
            st.success("✅ Smoking dataset loaded from local storage")
            return df
        except Exception as e:
            st.warning(f"Could not load preprocessed Smoking dataset: {e}")
    
    # Try to download data if needed
    urls = [
        "https://raw.githubusercontent.com/KAIST-CodingStudy/10-team10/main/data/smoking.csv",
        "https://raw.githubusercontent.com/MahdiMottahedi/Smoking-detection-with-machine-learning/main/smoking.csv",
        "https://raw.githubusercontent.com/krishnaik06/AQI-Project/master/smoking.csv"
    ]
    
    df = None
    for i, url in enumerate(urls):
        try:
            # Try to download the dataset
            st.info(f"Downloading Smoking dataset from source {i+1}...")
            df = download_dataset(url, raw_file_path)
            if df is None:
                continue
            
            # Preprocess data
            # Convert gender to numeric if it's not
            if 'gender' in df.columns and df['gender'].dtype == 'object':
                gender_map = {'F': 0, 'M': 1, 'Female': 0, 'Male': 1}
                df['gender'] = df['gender'].map(gender_map).fillna(1).astype(int)
                
            # Clean up potential issues in numerical columns
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numerical_cols:
                # Replace unrealistic values (negative values, extreme outliers)
                df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)
                
                # Fill NaNs with median of the column
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())
            
            # Handle any remaining object columns
            for col in df.select_dtypes(include=['object']).columns:
                # Convert any remaining object columns to categorical codes
                df[col] = df[col].astype('category').cat.codes.astype(int)
            
            # Ensure smoking is treated as an integer for classification
            df['smoking'] = df['smoking'].astype(int)
            
            # Additional preprocessing for specific columns if needed
            if df is not None and 'height' in df.columns and 'weight' in df.columns:
                # Add BMI if not present
                if 'bmi' not in df.columns:
                    df['bmi'] = (df['weight'] / ((df['height']/100) ** 2)).round(1)
                    # Fix any invalid BMI values
                    try:
                        # Calculate median, with fallback to a reasonable default value if needed
                        if df['bmi'].notna().any():
                            median_bmi = df['bmi'].median()
                        else:
                            median_bmi = 25.0  # Reasonable default BMI value
                            
                        # Apply the correction for outliers
                        df['bmi'] = df['bmi'].apply(lambda x: median_bmi if pd.isna(x) or x < 10 or x > 60 else x)
                    except Exception as e:
                        # In case of any other error, just use standard BMI ranges
                        st.warning(f"Error processing BMI values: {e}")
                        df['bmi'] = df['bmi'].fillna(25.0)  # Fill with reasonable default
            
            # Save preprocessed data for future use
            df.to_csv(preprocessed_file_path, index=False)
            st.success("✅ Smoking dataset downloaded and preprocessed successfully")
            
            return df
            
        except Exception as e:
            st.warning(f"Could not load Smoking dataset from source {i+1}: {e}")
            continue
    
    # If all sources fail, provide error
    st.error("❌ Failed to load Smoking dataset from all sources")
    columns = ['gender', 'age', 'height', 'weight', 'waist', 'eyesight_left', 'eyesight_right',
               'hearing_left', 'hearing_right', 'systolic', 'relaxation', 'fasting_blood_sugar',
               'cholesterol', 'triglyceride', 'hdl', 'ldl', 'hemoglobin', 'urine_protein',
               'serum_creatinine', 'ast', 'alt', 'gtp', 'dental_caries', 'smoking']
    return pd.DataFrame(columns=columns)
