# Healthcare Data Analysis & Prediction

A Streamlit-based healthcare data analysis application that predicts health safety status using machine learning models and visualizes key health indicators.

## Overview

This project analyzes healthcare data to identify trends and predict whether an individual's health is safe or at risk using machine learning techniques. The application includes:

- Data loading and exploration from reliable healthcare datasets
- Data preprocessing and cleaning
- Machine learning model training and evaluation
- Health risk prediction
- Visualization of key health indicators
- Interactive dashboard for exploring results

## Features

- **Data Exploration**: Load and explore various healthcare datasets
- **Data Preprocessing**: Handle missing values, encode categorical variables, normalize data
- **Model Training**: Train and evaluate Decision Trees, Random Forest, and Logistic Regression models
- **Health Prediction**: Predict health safety status using trained models
- **Visualizations**: View correlation matrices, feature importance, health indicator distributions, and model performance charts

## Data Sources

The application uses the following datasets:

1. **Heart Disease UCI**: Dataset from UCI Machine Learning Repository with heart disease indicators
2. **Pima Indians Diabetes**: Dataset for diabetes prediction based on diagnostic measurements
3. **Stroke Prediction**: Dataset to predict stroke likelihood based on health indicators
4. **Cardiovascular Disease**: Dataset with cardiovascular disease indicators
5. **Body Signal of Smoking**: Dataset with health indicators related to smoking habits

## Models

The application implements the following machine learning models:

- **Decision Tree**: A simple classification model that makes decisions based on feature values
- **Random Forest**: An ensemble method that combines multiple decision trees for improved performance
- **Logistic Regression**: A statistical model for binary classification problems

## Usage

1. Select a dataset from the dropdown menu
2. Explore the data to understand its structure and characteristics
3. Preprocess the data by selecting appropriate options
4. Train machine learning models to predict health status
5. Use the prediction tool to assess health risk based on input values
6. Explore visualizations to understand relationships between health indicators

## Requirements

- Python 3.11
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly

## check the data folder that contains csv files of the diseases raw data and processed data in that folder if it doesn't contain then download and place it in data folder for the to run the project 
- the data sets are available on Kaggle online platform

## ensure install the modules using this below command:
 1.this command automatically install all the required modules to run the health care data project->
- python requirements.txt

## Running the Application

 ## To run the application, use the following command:

- streamlit run app.py --server.address 127.0.0.1 --server.port 5000 

- (or)

- streamlit run app.py

