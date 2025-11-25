"""
Machine learning models for Walmart sales forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
import logging
import joblib

logger = logging.getLogger(__name__)

class SalesForecaster:
    """
    Sales forecasting model for Walmart data.
    """
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.scalers = {}
        
    def train_models(self, X, y, models=None):
        """
        Train multiple forecasting models.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series
            Target variable
        models : dict
            Dictionary of models to train
        """
        if models is None:
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression(),
                'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42)
            }
        
        for name, model in models.items():
            try:
                logger.info(f"Training {name}...")
                model.fit(X, y)
                self.models[name] = model
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                
                logger.info(f"Successfully trained {name}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models on test data.
        
        Parameters:
        -----------
        X_test : pandas.DataFrame
            Test feature matrix
        y_test : pandas.Series
            Test target variable
            
        Returns:
        --------
        pandas.DataFrame
            Model performance metrics
        """
        results = []
        
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
                
                metrics = {
                    'model': name,
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                }
                results.append(metrics)
                logger.info(f"{name} - RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.3f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
        
        return pd.DataFrame(results)
    
    def predict(self, X, model_name=None):
        """
        Make predictions using trained models.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix for prediction
        model_name : str
            Specific model to use (if None, uses all models)
            
        Returns:
        --------
        dict
            Predictions from each model
        """
        predictions = {}
        
        if model_name:
            models_to_use = {model_name: self.models[model_name]}
        else:
            models_to_use = self.models
        
        for name, model in models_to_use.items():
            try:
                predictions[name] = model.predict(X)
            except Exception as e:
                logger.error(f"Error predicting with {name}: {e}")
        
        return predictions
    
    def save_models(self, path="results/models"):
        """
        Save trained models to disk.
        
        Parameters:
        -----------
        path : str
            Path to save models
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            try:
                joblib.dump(model, f"{path}/{name}.joblib")
                logger.info(f"Saved model: {name}")
            except Exception as e:
                logger.error(f"Error saving {name}: {e}")
    
    def load_models(self, path="results/models"):
        """
        Load trained models from disk.
        
        Parameters:
        -----------
        path : str
            Path to load models from
        """
        import glob
        
        model_files = glob.glob(f"{path}/*.joblib")
        for file_path in model_files:
            try:
                model_name = file_path.split('/')[-1].replace('.joblib', '')
                self.models[model_name] = joblib.load(file_path)
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

def create_time_series_features(df, date_column='Date'):
    """
    Create time series specific features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    date_column : str
        Name of date column
        
    Returns:
    --------
    pandas.DataFrame
        Data with time series features
    """
    df_ts = df.copy()
    
    if date_column in df_ts.columns:
        df_ts[date_column] = pd.to_datetime(df_ts[date_column])
        
        # Time-based features
        df_ts['year'] = df_ts[date_column].dt.year
        df_ts['month'] = df_ts[date_column].dt.month
        df_ts['week'] = df_ts[date_column].dt.isocalendar().week
        df_ts['day_of_week'] = df_ts[date_column].dt.dayofweek
        df_ts['day_of_year'] = df_ts[date_column].dt.dayofyear
        df_ts['quarter'] = df_ts[date_column].dt.quarter
        df_ts['is_weekend'] = df_ts['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical features
        df_ts['month_sin'] = np.sin(2 * np.pi * df_ts['month'] / 12)
        df_ts['month_cos'] = np.cos(2 * np.pi * df_ts['month'] / 12)
        df_ts['day_sin'] = np.sin(2 * np.pi * df_ts['day_of_week'] / 7)
        df_ts['day_cos'] = np.cos(2 * np.pi * df_ts['day_of_week'] / 7)
    
    return df_ts

