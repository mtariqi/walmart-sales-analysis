"""
Data loading utilities for Walmart sales data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_walmart_data(data_path="data/raw"):
    """
    Load Walmart sales data from CSV files.
    
    Parameters:
    -----------
    data_path : str
        Path to directory containing raw data files
        
    Returns:
    --------
    dict
        Dictionary containing loaded DataFrames
    """
    data_path = Path(data_path)
    data_files = {
        'train': 'train.csv',
        'test': 'test.csv', 
        'stores': 'stores.csv',
        'features': 'features.csv'
    }
    
    loaded_data = {}
    
    for key, filename in data_files.items():
        file_path = data_path / filename
        try:
            if file_path.exists():
                loaded_data[key] = pd.read_csv(file_path)
                logger.info(f"Successfully loaded {filename}")
            else:
                logger.warning(f"File not found: {file_path}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
    
    return loaded_data

def validate_data(data_dict):
    """
    Basic data validation for loaded datasets.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing DataFrames
        
    Returns:
    --------
    dict
        Validation results
    """
    validation_results = {}
    
    for key, df in data_dict.items():
        results = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum()
        }
        validation_results[key] = results
    
    return validation_results
