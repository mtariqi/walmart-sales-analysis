"""
Visualization utilities for Walmart sales analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_sales_trends(df, date_column='Date', sales_column='Weekly_Sales', store_column='Store'):
    """
    Plot sales trends over time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Sales data
    date_column : str
        Name of date column
    sales_column : str
        Name of sales column
    store_column : str
        Name of store column
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Overall sales trend
    if date_column in df.columns and sales_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        monthly_sales = df.groupby(df[date_column].dt.to_period('M'))[sales_column].sum()
        axes[0,0].plot(monthly_sales.index.astype(str), monthly_sales.values)
        axes[0,0].set_title('Monthly Sales Trend')
        axes[0,0].tick_params(axis='x', rotation=45)
    
    # Sales by store
    if store_column in df.columns and sales_column in df.columns:
        store_sales = df.groupby(store_column)[sales_column].sum().sort_values(ascending=False)
        axes[0,1].bar(store_sales.index.astype(str), store_sales.values)
        axes[0,1].set_title('Total Sales by Store')
        axes[0,1].tick_params(axis='x', rotation=45)
    
    # Seasonal patterns
    if date_column in df.columns and sales_column in df.columns:
        df['month'] = pd.to_datetime(df[date_column]).dt.month
        monthly_avg = df.groupby('month')[sales_column].mean()
        axes[1,0].plot(monthly_avg.index, monthly_avg.values, marker='o')
        axes[1,0].set_title('Average Sales by Month')
        axes[1,0].set_xlabel('Month')
    
    # Sales distribution
    if sales_column in df.columns:
        axes[1,1].hist(df[sales_column].dropna(), bins=50, alpha=0.7)
        axes[1,1].set_title('Sales Distribution')
        axes[1,1].set_xlabel('Sales')
    
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df, numerical_columns=None):
    """
    Plot correlation heatmap for numerical variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data for correlation analysis
    numerical_columns : list
        List of numerical columns to include
    """
    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_columns) > 1:
        corr_matrix = df[numerical_columns].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax)
        ax.set_title('Correlation Heatmap')
        
        return fig
    else:
        logger.warning("Not enough numerical columns for correlation heatmap")
        return None

def plot_feature_importance(importance_df, top_n=15):
    """
    Plot feature importance from trained models.
    
    Parameters:
    -----------
    importance_df : pandas.DataFrame
        Feature importance data
    top_n : int
        Number of top features to show
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = importance_df.head(top_n)
    sns.barplot(data=top_features, x='importance', y='feature', ax=ax)
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    
    return fig

def create_interactive_sales_plot(df, date_column='Date', sales_column='Weekly_Sales', 
                                 store_column='Store'):
    """
    Create interactive sales plot using Plotly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Sales data
    date_column : str
        Name of date column
    sales_column : str
        Name of sales column
    store_column : str
        Name of store column
    """
    if all(col in df.columns for col in [date_column, sales_column, store_column]):
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Aggregate data
        daily_sales = df.groupby([date_column, store_column])[sales_column].sum().reset_index()
        
        fig = px.line(daily_sales, x=date_column, y=sales_column, color=store_column,
                     title='Sales Trends by Store Over Time')
        
        return fig
    else:
        logger.warning("Required columns not found for interactive plot")
        return None

def save_plot(fig, filename, path="results/figures"):
    """
    Save plot to file.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        Plot figure to save
    filename : str
        Name of the file
    path : str
        Directory to save the plot
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    filepath = Path(path) / filename
    
    try:
        if hasattr(fig, 'savefig'):  # Matplotlib figure
            fig.savefig(filepath, bbox_inches='tight', dpi=300)
        elif hasattr(fig, 'write_image'):  # Plotly figure
            fig.write_image(str(filepath))
        else:
            logger.error("Unsupported figure type")
            return False
        
        logger.info(f"Plot saved: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving plot {filename}: {e}")
        return False

