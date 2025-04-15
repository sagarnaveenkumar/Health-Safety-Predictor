import pandas as pd
import numpy as np

def display_dataset_info(dataset_name):
    """
    Returns information about the selected dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        str: Markdown formatted information about the dataset
    """
    if dataset_name == "Heart Disease UCI":
        return """
        **Heart Disease UCI Dataset**
        
        This dataset contains 14 attributes related to heart disease diagnosis. The target variable indicates the presence of heart disease (1) or absence (0).
        
        **Features:**
        - age: Age in years
        - sex: Gender (1 = male, 0 = female)
        - chest_pain_type: Type of chest pain (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)
        - resting_blood_pressure: Resting blood pressure in mm Hg
        - cholesterol: Serum cholesterol in mg/dl
        - fasting_blood_sugar: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
        - resting_ecg: Resting electrocardiographic results
        - max_heart_rate: Maximum heart rate achieved
        - exercise_induced_angina: Exercise induced angina (1 = yes, 0 = no)
        - st_depression: ST depression induced by exercise relative to rest
        - st_slope: Slope of the peak exercise ST segment
        - num_major_vessels: Number of major vessels colored by fluoroscopy (0-3)
        - thalassemia: Thalassemia types
        - heart_disease: Target variable (1 = heart disease present, 0 = no heart disease)
        
        **Source:** UCI Machine Learning Repository
        """
    
    elif dataset_name == "Diabetes":
        return """
        **Pima Indians Diabetes Dataset**
        
        This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.
        
        **Features:**
        - pregnancies: Number of times pregnant
        - glucose: Plasma glucose concentration (mg/dL)
        - blood_pressure: Diastolic blood pressure (mm Hg)
        - skin_thickness: Triceps skin fold thickness (mm)
        - insulin: 2-Hour serum insulin (mu U/ml)
        - bmi: Body mass index (weight in kg/(height in m)^2)
        - diabetes_pedigree: Diabetes pedigree function
        - age: Age in years
        - diabetes: Target variable (1 = has diabetes, 0 = no diabetes)
        
        **Source:** UCI Machine Learning Repository
        """
    
    elif dataset_name == "Stroke Prediction":
        return """
        **Stroke Prediction Dataset**
        
        This dataset is used to predict whether a patient is likely to get a stroke based on parameters like gender, age, various diseases, and smoking status.
        
        **Features:**
        - id: Unique identifier
        - gender: Gender of the patient
        - age: Age of the patient
        - hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
        - heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
        - ever_married: 'Yes' or 'No'
        - work_type: Type of work ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked')
        - Residence_type: 'Rural' or 'Urban'
        - avg_glucose_level: Average glucose level in blood
        - bmi: Body mass index
        - smoking_status: 'formerly smoked', 'never smoked', 'smokes', or 'Unknown'
        - stroke: Target variable (1 if the patient had a stroke, 0 if not)
        
        **Source:** Kaggle
        """
    
    elif dataset_name == "Cardiovascular Disease":
        return """
        **Cardiovascular Disease Dataset**
        
        This dataset contains parameters that can be used to predict whether a person has cardiovascular disease.
        
        **Features:**
        - age_days: Age in days (converted to age_years)
        - gender: Gender (1 = male, 2 = female)
        - height_cm: Height in centimeters
        - weight_kg: Weight in kilograms
        - systolic_bp: Systolic blood pressure
        - diastolic_bp: Diastolic blood pressure
        - cholesterol: 1: normal, 2: above normal, 3: well above normal
        - glucose: 1: normal, 2: above normal, 3: well above normal
        - smoker: Whether the patient smokes (0 = no, 1 = yes)
        - alcohol: Whether the patient drinks alcohol (0 = no, 1 = yes)
        - physical_activity: Whether the patient is physically active (0 = no, 1 = yes)
        - cardiovascular_disease: Target variable (0 = no disease, 1 = disease)
        - bmi: Body mass index (calculated from height and weight)
        
        **Source:** Kaggle
        """
    
    elif dataset_name == "Body Signal of Smoking":
        return """
        **Body Signal of Smoking Dataset**
        
        This dataset contains various health indicators and whether a person is a smoker or not.
        
        **Features:**
        - gender: Gender (male, female)
        - age: Age in years
        - height: Height in cm
        - weight: Weight in kg
        - waist: Waist circumference in cm
        - eyesight_left/right: Eyesight measurements
        - hearing_left/right: Hearing test results
        - systolic: Systolic blood pressure
        - relaxation: Diastolic blood pressure
        - fasting_blood_sugar: Fasting blood sugar level
        - cholesterol: Total cholesterol level
        - triglyceride: Triglyceride level
        - hdl: HDL (good) cholesterol
        - ldl: LDL (bad) cholesterol
        - hemoglobin: Hemoglobin level
        - urine_protein: Protein in urine
        - serum_creatinine: Serum creatinine level
        - ast, alt, gtp: Liver enzymes
        - dental_caries: Dental caries presence
        - smoking: Target variable (0 = non-smoker, 1 = smoker)
        
        **Source:** Kaggle
        """
    
    else:
        return f"No information available for {dataset_name}"

def calculate_health_stats(df):
    """
    Calculate summary statistics for key health indicators.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with health statistics
    """
    # Identify key health indicators
    health_indicators = []
    
    # Common health indicator patterns
    indicator_patterns = [
        'blood_pressure', 'bp', 'systolic', 'diastolic',
        'cholesterol', 'hdl', 'ldl', 'triglyceride',
        'bmi', 'weight', 'height',
        'glucose', 'sugar',
        'heart_rate', 'pulse',
        'age'
    ]
    
    # Find columns matching health indicator patterns
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in indicator_patterns):
            # Only include numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                health_indicators.append(col)
    
    # If no health indicators found, return empty dataframe
    if not health_indicators:
        return pd.DataFrame()
    
    # Calculate statistics for each health indicator
    stats = {
        'Indicator': [],
        'Mean': [],
        'Median': [],
        'Std Dev': [],
        'Min': [],
        'Max': []
    }
    
    for indicator in health_indicators:
        # Skip if all values are missing
        if df[indicator].isnull().all():
            continue
        
        # Add statistics
        stats['Indicator'].append(indicator)
        stats['Mean'].append(df[indicator].mean())
        stats['Median'].append(df[indicator].median())
        stats['Std Dev'].append(df[indicator].std())
        stats['Min'].append(df[indicator].min())
        stats['Max'].append(df[indicator].max())
    
    # Create dataframe
    stats_df = pd.DataFrame(stats)
    
    return stats_df
