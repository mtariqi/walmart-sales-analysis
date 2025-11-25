# Create a script that demonstrates generating models and visualizations
"""
Script to generate sample ML models and visualizations for demonstration.
This creates example files in the results/ directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json

def create_sample_data():
    """Create sample Walmart-like sales data for demonstration."""
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    data = {
        'Store': np.random.randint(1, 46, n_samples),
        'Dept': np.random.randint(1, 100, n_samples),
        'Temperature': np.random.normal(70, 15, n_samples),
        'Fuel_Price': np.random.uniform(2.5, 4.5, n_samples),
        'CPI': np.random.uniform(150, 250, n_samples),
        'Unemployment': np.random.uniform(3, 12, n_samples),
        'IsHoliday': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'MarkDown1': np.random.exponential(1000, n_samples),
        'MarkDown2': np.random.exponential(500, n_samples),
    }
    
    # Create target variable (Weekly_Sales) with some relationships
    data['Weekly_Sales'] = (
        10000 
        + data['Store'] * 50 
        + data['Dept'] * 30 
        + data['IsHoliday'] * 5000
        + data['MarkDown1'] * 0.5
        + data['MarkDown2'] * 0.3
        + np.random.normal(0, 2000, n_samples)
    )
    
    return pd.DataFrame(data)

def generate_visualizations(df, output_path="results/figures"):
    """Generate sample visualizations."""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    print("Generating visualizations...")
    
    # 1. Sales distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['Weekly_Sales'], bins=50, alpha=0.7, color='skyblue')
    plt.title('Distribution of Weekly Sales')
    plt.xlabel('Weekly Sales')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_path}/sales_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Sales by store
    plt.figure(figsize=(12, 6))
    store_sales = df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)
    store_sales.head(15).plot(kind='bar', color='lightgreen')
    plt.title('Average Weekly Sales by Store (Top 15)')
    plt.xlabel('Store')
    plt.ylabel('Average Weekly Sales')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_path}/sales_by_store.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'{output_path}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Holiday impact
    plt.figure(figsize=(8, 6))
    holiday_sales = df.groupby('IsHoliday')['Weekly_Sales'].mean()
    holiday_sales.plot(kind='bar', color=['lightcoral', 'lightblue'])
    plt.title('Average Sales: Holiday vs Non-Holiday')
    plt.xlabel('Is Holiday (0=No, 1=Yes)')
    plt.ylabel('Average Weekly Sales')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_path}/holiday_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_path}")

def train_sample_models(df, output_path="results/models"):
    """Train sample ML models and save them."""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    print("Training sample models...")
    
    # Prepare features and target
    feature_columns = ['Store', 'Dept', 'Temperature', 'Fuel_Price', 'CPI', 
                      'Unemployment', 'IsHoliday', 'MarkDown1', 'MarkDown2']
    X = df[feature_columns]
    y = df['Weekly_Sales']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2_score': r2_score(y_test, y_pred),
        'feature_importance': dict(zip(feature_columns, rf_model.feature_importances_))
    }
    
    # Save model
    joblib.dump(rf_model, f'{output_path}/random_forest_model.joblib')
    
    # Save metrics
    with open(f'{output_path}/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save feature importance
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance_df.to_csv(f'{output_path}/feature_importance.csv', index=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x='importance', y='feature')
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(f'{output_path}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Models and metrics saved to {output_path}")
    print(f"Model Performance - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2_score']:.3f}")

def main():
    """Main function to generate all sample outputs."""
    print("Walmart Sales Analysis - Sample Output Generator")
    print("=" * 50)
    
    # Create sample data
    print("1. Creating sample data...")
    df = create_sample_data()
    
    # Generate visualizations
    generate_visualizations(df)
    
    # Train and save models
    train_sample_models(df)
    
    print("\n" + "=" * 50)
    print("Sample outputs generated successfully!")
    print("Check the following directories:")
    print("  - results/figures/ : Contains visualizations")
    print("  - results/models/  : Contains trained models and metrics")
    
    # Print file listing
    print("\nGenerated files:")
    figures_path = Path("results/figures")
    models_path = Path("results/models")
    
    if figures_path.exists():
        print(f"\nFigures ({len(list(figures_path.glob('*.png')))} files):")
        for file in figures_path.glob("*.png"):
            print(f"  - {file.name}")
    
    if models_path.exists():
        print(f"\nModels and metrics ({len(list(models_path.glob('*')))} files):")
        for file in models_path.glob("*"):
            print(f"  - {file.name}")

if __name__ == "__main__":
    main()

