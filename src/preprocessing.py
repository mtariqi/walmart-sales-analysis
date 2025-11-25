"""
Data preprocessing and feature engineering for Walmart sales data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

def clean_data(df):
    """
    Clean and preprocess the sales data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw sales data
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned data
    """
    df_clean = df.copy()
    
    # Handle missing values
    if 'MarkDown' in df_clean.columns:
        markdown_cols = [col for col in df_clean.columns if 'MarkDown' in col]
        for col in markdown_cols:
            df_clean[col].fillna(0, inplace=True)
    
    # Convert date column if exists
    date_columns = ['Date', 'Week', 'WeekDate']
    for date_col in date_columns:
        if date_col in df_clean.columns:
            df_clean[date_col] = pd.to_datetime(df_clean[date_col])
            logger.info(f"Converted {date_col} to datetime")
    
    return df_clean

def feature_engineering(df):
    """
    Create new features from existing data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned sales data
        
    Returns:
    --------
    pandas.DataFrame
        Data with new features
    """
    df_featured = df.copy()
    
    # Date-based features
    date_columns = ['Date', 'Week', 'WeekDate']
    for date_col in date_columns:
        if date_col in df_featured.columns:
            df_featured[f'{date_col}_year'] = pd.to_datetime(df_featured[date_col]).dt.year
            df_featured[f'{date_col}_month'] = pd.to_datetime(df_featured[date_col]).dt.month
            df_featured[f'{date_col}_week'] = pd.to_datetime(df_featured[date_col]).dt.isocalendar().week
            df_featured[f'{date_col}_dayofweek'] = pd.to_datetime(df_featured[date_col]).dt.dayofweek
    
    # Holiday features
    if 'IsHoliday' in df_featured.columns:
        df_featured['IsHoliday'] = df_featured['IsHoliday'].astype(int)
    
    # Lag features for time series
    if 'Weekly_Sales' in df_featured.columns:
        for lag in [1, 2, 3, 4]:
            df_featured[f'Sales_lag_{lag}'] = df_featured.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)
    
    # Rolling statistics
    if 'Weekly_Sales' in df_featured.columns:
        df_featured['Sales_rolling_mean_4'] = df_featured.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean()
        )
    
    return df_featured

def prepare_modeling_data(df, target_column='Weekly_Sales'):
    """
    Prepare data for modeling by handling categorical variables and scaling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Feature-engineered data
    target_column : str
        Name of the target variable
        
    Returns:
    --------
    tuple
        (X, y) for modeling
    """
    df_model = df.copy()
    
    # Separate target variable
    if target_column in df_model.columns:
        y = df_model[target_column]
        X = df_model.drop(columns=[target_column])
    else:
        y = None
        X = df_model
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle numerical missing values
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='median')
    X[numerical_cols] = imputer.fit_transform(X[numerical_cols])
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y

