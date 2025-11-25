# Model Card: Walmart Sales Forecasting

## Model Details
- **Model Type**: Random Forest Regressor
- **Input Features**: Store, Department, Temperature, Fuel Price, CPI, Unemployment, Holiday Flag, Markdowns
- **Target Variable**: Weekly_Sales
- **Training Date**: $(date)

## Performance Metrics
- **RMSE**: [Value from model_metrics.json]
- **R² Score**: [Value from model_metrics.json]

## Feature Importance
Top 5 most important features:
1. [Top feature from feature_importance.csv]
2. [Second feature from feature_importance.csv]
3. [Third feature from feature_importance.csv]
4. [Fourth feature from feature_importance.csv]
5. [Fifth feature from feature_importance.csv]

## Usage
```python
import joblib
model = joblib.load('random_forest_model.joblib')
predictions = model.predict(new_data)
