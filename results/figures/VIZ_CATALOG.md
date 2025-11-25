```bash
# Visualization Catalog

## Generated Plots

### 1. Sales Distribution (`sales_distribution.png`)
- Shows the distribution of weekly sales across all stores
- Helps identify typical sales ranges and outliers

### 2. Sales by Store (`sales_by_store.png`)
- Displays average sales for top 15 stores
- Useful for store performance comparison

### 3. Correlation Heatmap (`correlation_heatmap.png`)
- Visualizes relationships between all numeric features
- Helps identify multicollinearity

### 4. Holiday Impact (`holiday_impact.png`)
- Compares average sales on holidays vs non-holidays
- Shows the business impact of holiday periods

### 5. Feature Importance (`feature_importance.png`)
- Shows which features most influence sales predictions
- Useful for feature selection and business insights

## Regeneration
To regenerate these visualizations with real data:
```python
from src.visualization import plot_sales_trends, plot_correlation_heatmap
# Use your actual Walmart data
fig = plot_sales_trends(real_data)
