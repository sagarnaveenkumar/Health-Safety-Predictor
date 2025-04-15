import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_correlation_heatmap(df):
    """
    Create a correlation heatmap for the numerical features in the dataframe.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        matplotlib.figure.Figure: Correlation heatmap figure
    """
    # Get only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Health Indicators', fontsize=16)
    plt.tight_layout()
    
    return plt.gcf()

def plot_feature_importance(feature_importance, feature_names):
    """
    Create a bar chart of feature importance.
    
    Args:
        feature_importance (numpy.ndarray): Array of feature importance values
        feature_names (list): List of feature names
        
    Returns:
        plotly.graph_objects.Figure: Feature importance figure
    """
    # Sort feature importances in descending order
    indices = np.argsort(feature_importance)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_feature_importance = feature_importance[indices]
    
    # Create figure
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_feature_names,
            y=sorted_feature_importance,
            marker_color='rgb(55, 83, 109)'
        )
    ])
    
    # Update layout
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Features',
        yaxis_title='Importance',
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=40, b=20),
        height=500
    )
    
    return fig

def plot_health_indicators_distribution(df, selected_indicators):
    """
    Create distribution plots for selected health indicators.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        selected_indicators (list): List of indicators to plot
        
    Returns:
        plotly.graph_objects.Figure: Distribution figure
    """
    # Create subplots
    n_indicators = len(selected_indicators)
    n_cols = 2
    n_rows = (n_indicators + 1) // 2
    
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=selected_indicators
    )
    
    # Add histograms for each indicator
    for i, indicator in enumerate(selected_indicators):
        row = i // 2 + 1
        col = i % 2 + 1
        
        # Create histogram
        hist_values = df[indicator].dropna()
        
        fig.add_trace(
            go.Histogram(
                x=hist_values,
                nbinsx=30,
                marker_color='rgb(55, 83, 109)'
            ),
            row=row,
            col=col
        )
        
        # Add mean line
        mean_val = hist_values.mean()
        fig.add_vline(
            x=mean_val,
            line_width=2,
            line_dash="dash",
            line_color="red",
            row=row,
            col=col
        )
    
    # Update layout
    fig.update_layout(
        title_text='Distribution of Health Indicators',
        showlegend=False,
        height=300 * n_rows,
        width=1000
    )
    
    fig.update_xaxes(title_text='Value')
    fig.update_yaxes(title_text='Count')
    
    return fig

def plot_model_comparison(model_results):
    """
    Create a bar chart comparing the performance of different models.
    
    Args:
        model_results (dict): Dictionary of model evaluation results
        
    Returns:
        plotly.graph_objects.Figure: Model comparison figure
    """
    # Prepare data
    models = list(model_results.keys())
    accuracy = [results["accuracy"] for results in model_results.values()]
    precision = [results["precision"] for results in model_results.values()]
    recall = [results["recall"] for results in model_results.values()]
    f1 = [results["f1"] for results in model_results.values()]
    
    # Create figure
    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=models, y=accuracy),
        go.Bar(name='Precision', x=models, y=precision),
        go.Bar(name='Recall', x=models, y=recall),
        go.Bar(name='F1 Score', x=models, y=f1)
    ])
    
    # Update layout
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        legend_title='Metrics',
        yaxis=dict(range=[0, 1]),
        barmode='group'
    )
    
    return fig

def plot_roc_curves(model_results):
    """
    Create ROC curves for all models.
    
    Args:
        model_results (dict): Dictionary of model evaluation results
        
    Returns:
        plotly.graph_objects.Figure: ROC curves figure
    """
    # Create figure
    fig = go.Figure()
    
    # Add ROC curve for each model
    for model_name, results in model_results.items():
        fpr = results["fpr"]
        tpr = results["tpr"]
        roc_auc = results["roc_auc"]
        
        fig.add_trace(
            go.Scatter(
                x=fpr, 
                y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {roc_auc:.3f})'
            )
        )
    
    # Add diagonal line representing random classifier
    fig.add_trace(
        go.Scatter(
            x=[0, 1], 
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='grey')
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend_title='Models',
        width=800,
        height=600
    )
    
    return fig

def plot_confusion_matrices(model_results):
    """
    Create confusion matrices for all models.
    
    Args:
        model_results (dict): Dictionary of model evaluation results
        
    Returns:
        matplotlib.figure.Figure: Confusion matrices figure
    """
    # Get number of models
    n_models = len(model_results)
    
    # Create figure
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    
    # If only one model, convert axes to list for consistent indexing
    if n_models == 1:
        axes = [axes]
    
    # Plot confusion matrix for each model
    for i, (model_name, results) in enumerate(model_results.items()):
        cm = results["confusion_matrix"]
        
        # Plot confusion matrix
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues",
            ax=axes[i]
        )
        
        # Set labels
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
        axes[i].set_title(f'Confusion Matrix - {model_name}')
    
    plt.tight_layout()
    
    return fig
