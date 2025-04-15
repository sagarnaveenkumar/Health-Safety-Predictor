import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

from data_loader import load_dataset, get_available_datasets
from data_preprocessor import preprocess_data, get_feature_target_split
from model_trainer import train_models, evaluate_models, predict_health_status
from visualization import (
    plot_correlation_heatmap, 
    plot_feature_importance, 
    plot_health_indicators_distribution,
    plot_model_comparison,
    plot_roc_curves,
    plot_confusion_matrices
)
from utils import display_dataset_info, calculate_health_stats

# Set page configuration
st.set_page_config(
    page_title="Healthcare Data Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'target_name' not in st.session_state:
    st.session_state.target_name = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

# Main header
st.title("Healthcare Data Analysis & Prediction")
st.markdown("### Data Science for Analyzing Healthcare Trends Using Big Data Analytics")
st.markdown("Analyze health indicators and predict health safety status using machine learning models.")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Data Loading & Exploration", "Data Preprocessing", "Model Training & Evaluation", "Health Prediction", "Visualizations"]
)

# Get available datasets
available_datasets = get_available_datasets()

# Data Loading & Exploration page
if page == "Data Loading & Exploration":
    st.header("Data Loading & Exploration")
    
    # Dataset selection
    dataset_option = st.selectbox(
        "Select a healthcare dataset:",
        available_datasets,
        index=0
    )
    
    if st.button("Load Dataset"):
        with st.spinner("Loading dataset..."):
            df = load_dataset(dataset_option)
            st.session_state.raw_data = df
            st.session_state.data_loaded = True
            st.success(f"Dataset '{dataset_option}' loaded successfully!")
    
    if st.session_state.data_loaded:
        st.subheader("Dataset Overview")
        
        # Display basic information about the dataset
        st.write("Number of records:", len(st.session_state.raw_data))
        st.write("Number of features:", len(st.session_state.raw_data.columns))
        
        # Display first few rows of the dataset
        st.subheader("Sample Data")
        st.dataframe(st.session_state.raw_data.head())
        
        # Display data types and statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Types")
            st.dataframe(pd.DataFrame(st.session_state.raw_data.dtypes, columns=["Data Type"]))
        
        with col2:
            st.subheader("Summary Statistics")
            st.dataframe(st.session_state.raw_data.describe())
        
        # Display missing values
        st.subheader("Missing Values")
        missing_values = pd.DataFrame(st.session_state.raw_data.isnull().sum(), columns=["Count"])
        missing_values["Percentage"] = missing_values["Count"] / len(st.session_state.raw_data) * 100
        st.dataframe(missing_values)
        
        # Display dataset information
        st.subheader("Dataset Information")
        dataset_info = display_dataset_info(dataset_option)
        st.markdown(dataset_info)
        
        # Basic distribution plots
        st.subheader("Feature Distributions")
        numeric_cols = st.session_state.raw_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_cols:
            selected_feature = st.selectbox("Select a feature to visualize:", numeric_cols)
            fig = px.histogram(st.session_state.raw_data, x=selected_feature, marginal="box", 
                              title=f"Distribution of {selected_feature}")
            st.plotly_chart(fig, use_container_width=True)            

# Data Preprocessing page
elif page == "Data Preprocessing":
    st.header("Data Preprocessing")
    
    if not st.session_state.data_loaded:
        st.warning("Please load a dataset first!")
    else:
        st.subheader("Select Target Variable")
        target_options = st.session_state.raw_data.columns.tolist()
        target_variable = st.selectbox("Select the target variable for prediction:", target_options)
        
        st.subheader("Data Preprocessing Options")
        handle_missing = st.checkbox("Handle Missing Values", value=True)
        encode_categorical = st.checkbox("Encode Categorical Variables", value=True)
        normalize_data = st.checkbox("Normalize Numerical Features", value=True)
        
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                preprocessed_data = preprocess_data(
                    st.session_state.raw_data,
                    handle_missing=handle_missing,
                    encode_categorical=encode_categorical,
                    normalize_data=normalize_data
                )
                
                X, y, feature_names = get_feature_target_split(preprocessed_data, target_variable)
                
                # Ensure target is integer type for classification
                y = y.astype(int)
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Store in session state
                st.session_state.preprocessed_data = preprocessed_data
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.feature_names = feature_names
                st.session_state.target_name = target_variable
                
                # Additional check to ensure target variable is properly prepared for classification
                st.write(f"Target variable unique values: {np.unique(y)}")
                
                st.success("Data preprocessing completed!")
        
        if st.session_state.preprocessed_data is not None:
            st.subheader("Preprocessed Data")
            st.dataframe(st.session_state.preprocessed_data.head())
            
            st.subheader("Feature Correlation Matrix")
            correlation_fig = plot_correlation_heatmap(st.session_state.preprocessed_data)
            st.pyplot(correlation_fig)
            
            # Key health indicators summary
            st.subheader("Key Health Indicators Summary")
            health_stats = calculate_health_stats(st.session_state.preprocessed_data)
            st.dataframe(health_stats)

# Model Training & Evaluation page
elif page == "Model Training & Evaluation":
    st.header("Model Training & Evaluation")
    
    if st.session_state.preprocessed_data is None:
        st.warning("Please preprocess the data first!")
    else:
        st.subheader("Select Models to Train")
        use_decision_tree = st.checkbox("Decision Tree", value=True)
        use_random_forest = st.checkbox("Random Forest", value=True)
        use_logistic_regression = st.checkbox("Logistic Regression", value=True)
        
        if st.button("Train Models"):
            with st.spinner("Training models..."):
                models = []
                if use_decision_tree:
                    models.append("decision_tree")
                if use_random_forest:
                    models.append("random_forest")
                if use_logistic_regression:
                    models.append("logistic_regression")
                
                if not models:
                    st.error("Please select at least one model to train!")
                else:
                    # Train the selected models
                    trained_models = train_models(
                        st.session_state.X_train, 
                        st.session_state.y_train,
                        models
                    )
                    
                    # Evaluate the models
                    model_results = evaluate_models(
                        trained_models,
                        st.session_state.X_test,
                        st.session_state.y_test,
                        st.session_state.feature_names
                    )
                    
                    # Store in session state
                    st.session_state.models = trained_models
                    st.session_state.model_results = model_results
                    st.session_state.models_trained = True
                    
                    st.success("Models trained and evaluated successfully!")
        
        if st.session_state.models_trained:
            st.subheader("Model Performance")
            
            # Display model metrics - with null check
            if st.session_state.model_results is not None and len(st.session_state.model_results) > 0:
                metrics_df = pd.DataFrame({
                    "Model": list(st.session_state.model_results.keys()),
                    "Accuracy": [results["accuracy"] for results in st.session_state.model_results.values()],
                    "Precision": [results["precision"] for results in st.session_state.model_results.values()],
                    "Recall": [results["recall"] for results in st.session_state.model_results.values()],
                    "F1 Score": [results["f1"] for results in st.session_state.model_results.values()]
                })
            else:
                # Create empty dataframe with appropriate columns if no models are available
                metrics_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
            
            st.dataframe(metrics_df)
            
            # Model comparison visualization with null checks
            if st.session_state.model_results is not None and len(st.session_state.model_results) > 0:
                st.subheader("Model Comparison")
                model_comparison_fig = plot_model_comparison(st.session_state.model_results)
                st.plotly_chart(model_comparison_fig, use_container_width=True)
                
                # ROC Curves
                st.subheader("ROC Curves")
                roc_fig = plot_roc_curves(st.session_state.model_results)
                st.plotly_chart(roc_fig, use_container_width=True)
                
                # Confusion Matrices
                st.subheader("Confusion Matrices")
                cm_fig = plot_confusion_matrices(st.session_state.model_results)
                st.pyplot(cm_fig)
            else:
                st.info("Train models first to view performance metrics.")
            
            # Feature Importance - with null checks
            if st.session_state.model_results is not None and len(st.session_state.model_results) > 0:
                st.subheader("Feature Importance")
                
                # Safely determine which model to use for feature importance
                model_name = None
                
                # Try to find a tree-based model first (these have better feature importance)
                if st.session_state.model_results is not None:
                    if "random_forest" in st.session_state.model_results:
                        model_name = "random_forest"
                    elif "decision_tree" in st.session_state.model_results:
                        model_name = "decision_tree"
                    elif len(st.session_state.model_results) > 0:
                        # Fall back to first model if no tree-based models available
                        model_name = list(st.session_state.model_results.keys())[0]
                
                # Only proceed if we have a valid model and feature names
                if (model_name is not None and 
                    st.session_state.feature_names is not None and 
                    "feature_importance" in st.session_state.model_results[model_name]):
                    
                    importance_fig = plot_feature_importance(
                        st.session_state.model_results[model_name]["feature_importance"],
                        st.session_state.feature_names
                    )
                    st.plotly_chart(importance_fig, use_container_width=True)
                else:
                    st.info("Feature importance visualization not available for the selected model.")

# Health Prediction page
elif page == "Health Prediction":
    st.header("Health Safety Prediction")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first!")
    else:
        st.subheader("Input Your Health Indicators")
        st.write("Please enter your health information below for prediction. All fields are required.")
        
        # Initialize form for user input
        with st.form("health_prediction_form"):
            # Create columns to make the form more compact
            cols = st.columns(3)
            
            # Initialize an empty dictionary to store user inputs
            user_inputs = {}
            
            # Predefined standard ranges for common health metrics
            health_ranges = {
                'age': {'min': 18, 'max': 100, 'default': 35, 'step': 1},
                'height': {'min': 120, 'max': 220, 'default': 170, 'step': 1}, # cm
                'weight': {'min': 30, 'max': 200, 'default': 70, 'step': 0.5}, # kg
                'bmi': {'min': 15.0, 'max': 45.0, 'default': 24.5, 'step': 0.1},
                'heart_rate': {'min': 40, 'max': 200, 'default': 75, 'step': 1},
                'systolic': {'min': 80, 'max': 200, 'default': 120, 'step': 1},
                'diastolic': {'min': 40, 'max': 120, 'default': 80, 'step': 1},
                'cholesterol': {'min': 100, 'max': 300, 'default': 180, 'step': 1},
                'glucose': {'min': 70, 'max': 200, 'default': 100, 'step': 1},
            }
            
            # For each feature, create an appropriate input field
            feature_names = st.session_state.feature_names if st.session_state.feature_names is not None else []
            for i, feature in enumerate(feature_names):
                col_idx = i % 3
                with cols[col_idx]:
                    # Format feature name for display
                    feature_display = feature.replace('_', ' ').title()
                    
                    # Determine appropriate input type based on feature name and add helpful description
                    feature_lower = feature.lower()
                    
                    # Set a reasonable range and input type for common health metrics
                    if 'sex' in feature_lower or 'gender' in feature_lower:
                        # Map sex/gender: 0=Female, 1=Male in most datasets
                        labels = {0: "Female", 1: "Male"}
                        selected_option = st.radio(
                            f"{feature_display}",
                            options=list(labels.keys()),
                            format_func=lambda x: labels[x],
                            horizontal=True,
                            help="Select your biological sex"
                        )
                        user_inputs[feature] = selected_option
                        
                    elif 'age' in feature_lower:
                        range_info = health_ranges.get('age', {'min': 18, 'max': 100, 'default': 35, 'step': 1})
                        user_inputs[feature] = st.number_input(
                            f"{feature_display} (years)",
                            min_value=range_info['min'], 
                            max_value=range_info['max'],
                            value=range_info['default'],
                            step=range_info['step'],
                            help="Enter your age in years"
                        )
                    
                    elif 'height' in feature_lower:
                        range_info = health_ranges.get('height', {'min': 120, 'max': 220, 'default': 170, 'step': 1})
                        user_inputs[feature] = st.number_input(
                            f"{feature_display} (cm)",
                            min_value=range_info['min'], 
                            max_value=range_info['max'],
                            value=range_info['default'],
                            step=range_info['step'],
                            help="Enter your height in centimeters"
                        )
                    
                    elif 'weight' in feature_lower:
                        range_info = health_ranges.get('weight', {'min': 30, 'max': 200, 'default': 70, 'step': 0.5})
                        user_inputs[feature] = st.number_input(
                            f"{feature_display} (kg)",
                            min_value=range_info['min'], 
                            max_value=range_info['max'],
                            value=range_info['default'],
                            step=range_info['step'],
                            help="Enter your weight in kilograms"
                        )
                    
                    elif 'bmi' in feature_lower:
                        range_info = health_ranges.get('bmi', {'min': 15.0, 'max': 45.0, 'default': 24.5, 'step': 0.1})
                        user_inputs[feature] = st.number_input(
                            f"{feature_display}",
                            min_value=range_info['min'], 
                            max_value=range_info['max'],
                            value=range_info['default'],
                            step=range_info['step'],
                            help="Body Mass Index (weight in kg / height in m²)"
                        )
                    
                    elif 'heart_rate' in feature_lower or 'pulse' in feature_lower:
                        range_info = health_ranges.get('heart_rate', {'min': 40, 'max': 200, 'default': 75, 'step': 1})
                        user_inputs[feature] = st.number_input(
                            f"{feature_display} (bpm)",
                            min_value=range_info['min'], 
                            max_value=range_info['max'],
                            value=range_info['default'],
                            step=range_info['step'],
                            help="Enter your resting heart rate in beats per minute"
                        )
                    
                    elif 'blood_pressure' in feature_lower or 'systolic' in feature_lower:
                        range_info = health_ranges.get('systolic', {'min': 80, 'max': 200, 'default': 120, 'step': 1})
                        user_inputs[feature] = st.number_input(
                            f"{feature_display} (mmHg)",
                            min_value=range_info['min'], 
                            max_value=range_info['max'],
                            value=range_info['default'],
                            step=range_info['step'],
                            help="Enter your systolic blood pressure (the top number)"
                        )
                    
                    elif 'diastolic' in feature_lower or 'relaxation' in feature_lower:
                        range_info = health_ranges.get('diastolic', {'min': 40, 'max': 120, 'default': 80, 'step': 1})
                        user_inputs[feature] = st.number_input(
                            f"{feature_display} (mmHg)",
                            min_value=range_info['min'], 
                            max_value=range_info['max'],
                            value=range_info['default'],
                            step=range_info['step'],
                            help="Enter your diastolic blood pressure (the bottom number)"
                        )
                    
                    elif 'cholesterol' in feature_lower:
                        range_info = health_ranges.get('cholesterol', {'min': 100, 'max': 300, 'default': 180, 'step': 1})
                        user_inputs[feature] = st.number_input(
                            f"{feature_display} (mg/dL)",
                            min_value=range_info['min'], 
                            max_value=range_info['max'],
                            value=range_info['default'],
                            step=range_info['step'],
                            help="Enter your total cholesterol level"
                        )
                    
                    elif 'glucose' in feature_lower or 'blood_sugar' in feature_lower:
                        range_info = health_ranges.get('glucose', {'min': 70, 'max': 200, 'default': 100, 'step': 1})
                        user_inputs[feature] = st.number_input(
                            f"{feature_display} (mg/dL)",
                            min_value=range_info['min'], 
                            max_value=range_info['max'],
                            value=range_info['default'],
                            step=range_info['step'],
                            help="Enter your fasting blood glucose level"
                        )
                    
                    # Handle heart disease specific features
                    elif 'chest_pain' in feature_lower or 'cp' in feature_lower:
                        pain_types = {
                            0: "Typical Angina",
                            1: "Atypical Angina",
                            2: "Non-anginal Pain",
                            3: "Asymptomatic"
                        }
                        user_inputs[feature] = st.selectbox(
                            f"{feature_display}",
                            options=list(pain_types.keys()),
                            format_func=lambda x: pain_types[x],
                            help="Select the type of chest pain experienced"
                        )
                    
                    elif 'st_slope' in feature_lower:
                        slope_types = {
                            0: "Upsloping",
                            1: "Flat",
                            2: "Downsloping"
                        }
                        user_inputs[feature] = st.selectbox(
                            f"{feature_display}",
                            options=list(slope_types.keys()),
                            format_func=lambda x: slope_types[x],
                            help="ST segment slope during exercise ECG"
                        )
                    
                    elif 'resting_ecg' in feature_lower or 'rest_ecg' in feature_lower:
                        ecg_types = {
                            0: "Normal", 
                            1: "ST-T Wave Abnormality", 
                            2: "Left Ventricular Hypertrophy"
                        }
                        user_inputs[feature] = st.selectbox(
                            f"{feature_display}",
                            options=list(ecg_types.keys()),
                            format_func=lambda x: ecg_types[x],
                            help="Resting electrocardiographic results"
                        )
                    
                    elif 'thalassemia' in feature_lower or 'thal' in feature_lower:
                        thal_types = {
                            0: "Normal",
                            1: "Fixed Defect",
                            2: "Reversible Defect",
                            3: "Unknown"
                        }
                        user_inputs[feature] = st.selectbox(
                            f"{feature_display}", 
                            options=list(thal_types.keys()),
                            format_func=lambda x: thal_types[x],
                            help="Thalassemia is a blood disorder that affects hemoglobin production"
                        )
                        
                    # Handle binary categorical variables
                    elif any(word in feature_lower for word in ['smoke', 'smoking', 'tobacco']):
                        user_inputs[feature] = st.selectbox(
                            f"{feature_display}",
                            options=[0, 1],
                            format_func=lambda x: "Yes" if x == 1 else "No",
                            help="Do you smoke or use tobacco products?"
                        )
                    
                    elif 'alcohol' in feature_lower:
                        user_inputs[feature] = st.selectbox(
                            f"{feature_display}",
                            options=[0, 1],
                            format_func=lambda x: "Yes" if x == 1 else "No",
                            help="Do you consume alcohol regularly?"
                        )
                    
                    elif 'exercise' in feature_lower or 'activity' in feature_lower:
                        user_inputs[feature] = st.selectbox(
                            f"{feature_display}",
                            options=[0, 1],
                            format_func=lambda x: "Yes" if x == 1 else "No",
                            help="Do you engage in regular physical activity?"
                        )
                    
                    # Special handling for exercise-induced angina
                    elif 'angina' in feature_lower or 'exang' in feature_lower:
                        user_inputs[feature] = st.selectbox(
                            f"{feature_display}",
                            options=[0, 1],
                            format_func=lambda x: "Yes" if x == 1 else "No",
                            help="Chest pain occurring during physical exertion"
                        )
                        
                    # Special handling for number of major vessels
                    elif 'vessels' in feature_lower or 'ca' in feature_lower:
                        vessel_options = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}
                        user_inputs[feature] = st.selectbox(
                            f"{feature_display}",
                            options=list(vessel_options.keys()),
                            format_func=lambda x: vessel_options[x],
                            help="Number of major vessels colored by fluoroscopy (0-4)"
                        )
                    
                    # For other likely binary variables
                    elif feature.startswith(('is_', 'has_')) or feature.endswith(('_flag', '_status')):
                        user_inputs[feature] = st.selectbox(
                            f"{feature_display}",
                            options=[0, 1],
                            format_func=lambda x: "Yes" if x == 1 else "No"
                        )
                    
                    # For all other features, provide a number input with reasonable defaults
                    else:
                        # Try to determine if the feature should be an integer or float
                        is_likely_integer = any(word in feature_lower for word in ['count', 'number', 'index', 'id', '_id'])
                        
                        if is_likely_integer:
                            user_inputs[feature] = st.number_input(
                                f"{feature_display}",
                                min_value=0,
                                max_value=1000,
                                value=0,
                                step=1
                            )
                        else:
                            user_inputs[feature] = st.number_input(
                                f"{feature_display}",
                                min_value=0.0,
                                max_value=1000.0,
                                value=0.0,
                                step=0.1
                            )
            
            # Add model selection inside the form
            st.subheader("Select Model for Prediction")
            # Handle case when no models are available
            if st.session_state.models is not None and len(st.session_state.models) > 0:
                model_options = list(st.session_state.models.keys())
                selected_model = st.selectbox("Choose a model:", model_options)
            else:
                st.warning("No trained models available. Please train models first.")
                model_options = ["No models available"]
                selected_model = model_options[0]
            
            # Submit button for the form
            submitted = st.form_submit_button("Predict Health Status")
            
        # Process form submission
        if submitted:
            with st.spinner("Analyzing health data..."):
                # Only proceed if we have models available
                if selected_model == "No models available":
                    st.error("Cannot make prediction without trained models. Please train models first.")
                else:
                    try:
                        # Convert user inputs to the format expected by the model
                        input_data = np.array([list(user_inputs.values())])
                        
                        # Make prediction with null check
                        if st.session_state.models is not None and selected_model in st.session_state.models:
                            prediction, probability = predict_health_status(
                                st.session_state.models[selected_model],
                                input_data
                            )
                        else:
                            st.error("Selected model not available. Please train models first.")
                            prediction, probability = None, None
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        prediction, probability = None, None
                    
                    # Only display results if we have valid predictions
                    if prediction is not None and probability is not None:
                        # Display prediction results
                        st.subheader("Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Get number of classes from prediction probabilities
                            num_classes = probability[0].shape[0]
                            
                            if num_classes == 2:
                                # Binary classification
                                if prediction[0] == 1:
                                    st.error("⚠️ Predicted Health Status: At Risk")
                                else:
                                    st.success("✅ Predicted Health Status: Safe")
                            else:
                                # Multiclass classification
                                risk_level = prediction[0]
                                if risk_level == 0:
                                    st.success("✅ Predicted Health Status: Low Risk")
                                elif risk_level == 1:
                                    st.warning("⚠️ Predicted Health Status: Moderate Risk")
                                else:
                                    st.error("🚨 Predicted Health Status: High Risk")
                                    
                                st.write(f"Predicted Risk Class: {prediction[0]}")
                            
                        with col2:
                            # Display probability gauge - adaptable to multiclass
                            if num_classes == 2:
                                # Binary classification case
                                risk_prob = probability[0][1] * 100
                                gauge_title = "Risk Probability (%)"
                            else:
                                # For multiclass, use the predicted class probability
                                risk_prob = probability[0][prediction[0]] * 100
                                gauge_title = f"Confidence in Prediction (%)"
                            
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = risk_prob,
                                title = {'text': gauge_title},
                                gauge = {
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "darkred"},
                                    'steps': [
                                        {'range': [0, 30], 'color': "green"},
                                        {'range': [30, 70], 'color': "yellow"},
                                        {'range': [70, 100], 'color': "red"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "black", 'width': 4},
                                        'thickness': 0.75,
                                        'value': risk_prob
                                    }
                                }
                            ))
                            st.plotly_chart(fig, use_container_width=True)
                            
                        # Check if model results are available
                        if (st.session_state.model_results is not None and 
                            selected_model in st.session_state.model_results and
                            "feature_importance" in st.session_state.model_results[selected_model] and
                            st.session_state.feature_names is not None):
                            
                            # Display health insights based on prediction
                            st.subheader("Health Insights")
                            
                            # Get feature importance for the selected model
                            feature_importance = st.session_state.model_results[selected_model]["feature_importance"]
                            
                            # Sort features by importance
                            sorted_features = [x for _, x in sorted(zip(feature_importance, st.session_state.feature_names), reverse=True)]
                            sorted_importance = sorted(feature_importance, reverse=True)
                            
                            # Display top factors affecting health
                            st.write("Top factors affecting health prediction:")
                            
                            for i in range(min(5, len(sorted_features))):
                                feature = sorted_features[i]
                                importance = sorted_importance[i]
                                user_value = user_inputs[feature]
                                
                                # Provide insights based on feature
                                feature_display = feature.replace('_', ' ').title()
                                st.write(f"**{i+1}. {feature_display}** (Importance: {importance:.2f})")
                                
                                # Add specific guidance for common health metrics
                                if 'bmi' in feature.lower():
                                    if user_value < 18.5:
                                        st.write("   - Your BMI indicates you may be underweight.")
                                    elif user_value < 25:
                                        st.write("   - Your BMI is within the healthy range.")
                                    elif user_value < 30:
                                        st.write("   - Your BMI indicates you may be overweight.")
                                    else:
                                        st.write("   - Your BMI indicates potential obesity, which increases health risks.")
                                
                                elif 'blood_pressure' in feature.lower() or 'systolic' in feature.lower():
                                    if user_value < 120:
                                        st.write("   - Your systolic blood pressure is in the normal range.")
                                    elif user_value < 130:
                                        st.write("   - Your systolic blood pressure is elevated.")
                                    elif user_value < 140:
                                        st.write("   - Your systolic blood pressure indicates Stage 1 hypertension.")
                                    else:
                                        st.write("   - Your systolic blood pressure indicates Stage 2 hypertension or higher.")
                                
                                elif 'cholesterol' in feature.lower():
                                    if user_value < 200:
                                        st.write("   - Your cholesterol is in the desirable range.")
                                    elif user_value < 240:
                                        st.write("   - Your cholesterol is borderline high.")
                                    else:
                                        st.write("   - Your cholesterol is high, which increases risk for heart disease.")
                            
                        # Provide general health recommendations
                        st.subheader("General Health Recommendations")
                        
                        recommendations = [
                            "Maintain a balanced diet rich in fruits, vegetables, and whole grains.",
                            "Engage in regular physical activity (aim for at least 150 minutes per week).",
                            "Maintain a healthy weight and BMI.",
                            "Limit alcohol consumption and avoid smoking.",
                            "Manage stress through relaxation techniques or mindfulness.",
                            "Get regular health check-ups and screenings.",
                            "Ensure adequate sleep (7-9 hours for adults)."
                        ]
                        
                        for rec in recommendations:
                            st.write(f"• {rec}")
                        
                        st.warning("Note: This prediction is based on a machine learning model and should not replace professional medical advice. Please consult with a healthcare provider for accurate health assessment.")

# Visualizations page
elif page == "Visualizations":
    st.header("Health Data Visualizations")
    
    if not st.session_state.data_loaded:
        st.warning("Please load a dataset first!")
    else:
        visualization_type = st.selectbox(
            "Select Visualization Type:",
            ["Health Indicators Distribution", "Correlation Analysis", "Target Variable Analysis"]
        )
        
        if visualization_type == "Health Indicators Distribution":
            st.subheader("Distribution of Health Indicators")
            
            # Create a distribution plot for health indicators
            numeric_cols = st.session_state.raw_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            # Allow selection of multiple health indicators
            selected_indicators = st.multiselect(
                "Select health indicators to visualize:",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if selected_indicators:
                dist_fig = plot_health_indicators_distribution(st.session_state.raw_data, selected_indicators)
                st.plotly_chart(dist_fig, use_container_width=True)
                
                # Show descriptive statistics
                st.subheader("Descriptive Statistics")
                st.dataframe(st.session_state.raw_data[selected_indicators].describe())
        
        elif visualization_type == "Correlation Analysis":
            st.subheader("Correlation Between Health Indicators")
            
            if st.session_state.preprocessed_data is None:
                st.warning("Please preprocess the data first to see correlations!")
            else:
                # Correlation heatmap
                corr_fig = plot_correlation_heatmap(st.session_state.preprocessed_data)
                st.pyplot(corr_fig)
                
                # Scatter plot for exploring relationships between two variables
                numeric_cols = st.session_state.preprocessed_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    x_variable = st.selectbox("Select X-axis variable:", numeric_cols, index=0)
                with col2:
                    y_variable = st.selectbox("Select Y-axis variable:", numeric_cols, index=min(1, len(numeric_cols)-1))
                
                # Add color by target if available
                if st.session_state.target_name in st.session_state.preprocessed_data.columns:
                    color_by_target = st.checkbox("Color by target variable", value=True)
                    
                    if color_by_target:
                        fig = px.scatter(
                            st.session_state.preprocessed_data, 
                            x=x_variable, 
                            y=y_variable,
                            color=st.session_state.target_name,
                            title=f"Relationship between {x_variable} and {y_variable}"
                        )
                    else:
                        fig = px.scatter(
                            st.session_state.preprocessed_data, 
                            x=x_variable, 
                            y=y_variable,
                            title=f"Relationship between {x_variable} and {y_variable}"
                        )
                else:
                    fig = px.scatter(
                        st.session_state.preprocessed_data, 
                        x=x_variable, 
                        y=y_variable,
                        title=f"Relationship between {x_variable} and {y_variable}"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif visualization_type == "Target Variable Analysis":
            st.subheader("Target Variable Analysis")
            
            if st.session_state.target_name is None:
                st.warning("Please preprocess the data and select a target variable first!")
            else:
                # Get target variable data
                target_data = st.session_state.raw_data[st.session_state.target_name]
                
                # Display target variable distribution
                if pd.api.types.is_numeric_dtype(target_data):
                    if len(target_data.unique()) <= 10:  # If few unique values, likely categorical
                        fig = px.histogram(
                            st.session_state.raw_data,
                            x=st.session_state.target_name,
                            title=f"Distribution of {st.session_state.target_name}",
                            color_discrete_sequence=["#3366CC"]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show counts and percentages
                        st.subheader("Target Value Counts")
                        counts = target_data.value_counts().reset_index()
                        counts.columns = [st.session_state.target_name, "Count"]
                        counts["Percentage"] = counts["Count"] / counts["Count"].sum() * 100
                        st.dataframe(counts)
                    else:
                        fig = px.histogram(
                            st.session_state.raw_data,
                            x=st.session_state.target_name,
                            title=f"Distribution of {st.session_state.target_name}",
                            color_discrete_sequence=["#3366CC"]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.pie(
                        values=target_data.value_counts().values,
                        names=target_data.value_counts().index,
                        title=f"Distribution of {st.session_state.target_name}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature relationship with target
                st.subheader("Feature Relationship with Target")
                
                numeric_cols = st.session_state.raw_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
                if st.session_state.target_name in numeric_cols:
                    numeric_cols.remove(st.session_state.target_name)
                
                selected_feature = st.selectbox(
                    "Select a feature to analyze against the target:",
                    numeric_cols
                )
                
                if pd.api.types.is_numeric_dtype(target_data) and len(target_data.unique()) <= 10:
                    # Boxplot for categorical target
                    fig = px.box(
                        st.session_state.raw_data,
                        x=st.session_state.target_name,
                        y=selected_feature,
                        title=f"{selected_feature} by {st.session_state.target_name}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif pd.api.types.is_numeric_dtype(target_data):
                    # Scatter plot for continuous target
                    fig = px.scatter(
                        st.session_state.raw_data,
                        x=selected_feature,
                        y=st.session_state.target_name,
                        title=f"Relationship between {selected_feature} and {st.session_state.target_name}",
                        trendline="ols"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Boxplot for categorical target
                    fig = px.box(
                        st.session_state.raw_data,
                        x=st.session_state.target_name,
                        y=selected_feature,
                        title=f"{selected_feature} by {st.session_state.target_name}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Healthcare Data Analysis & Prediction Tool** | Developed for health data analysis and prediction")
