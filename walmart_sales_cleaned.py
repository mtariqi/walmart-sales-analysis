"""
Cleaned and improved version of the `walmart_sales.ipynb` notebook
- Paths expect CSVs in /mnt/data (features.csv, stores.csv, sales_data.csv)
- If sales_data.csv is missing the script will explain and continue with available merges where possible
- Includes:
    * tidy imports and settings
    * robust loading with checks
    * preprocessing and feature engineering
    * EDA helper functions
    * example modeling pipeline (baseline model)
    * saving cleaned datasets and figures

Run as a Jupyter cell or as a script (it prints helpful status messages).
"""

# Standard imports
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ML imports (scikit-learn)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Settings
DATA_DIR = Path("/mnt/data")
FEATURES_FP = DATA_DIR / "features.csv"
STORES_FP = DATA_DIR / "stores.csv"
SALES_FP = DATA_DIR / "sales_data.csv"  # expected filename used in original notebook
OUTPUT_DIR = DATA_DIR / "walmart_output"
OUTPUT_DIR.mkdir(exist_ok=True)

pd.options.display.max_columns = 200
sns.set_style("whitegrid")

# Utility functions

def load_csv_checked(fp: Path):
    """Load CSV if exists, otherwise return None and print helpful message."""
    if fp.exists():
        print(f"Loading: {fp}")
        return pd.read_csv(fp)
    else:
        print(f"WARNING: {fp.name} not found in {fp.parent}. Please upload it and re-run.\n")
        return None


def parse_dates_in_df(df: pd.DataFrame, date_col: str = "Date"):
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


# 1) Load data
features = load_csv_checked(FEATURES_FP)
stores = load_csv_checked(STORES_FP)
sales = load_csv_checked(SALES_FP)

# Quick info
print('\nFiles status:')
for name, df in [('features', features), ('stores', stores), ('sales', sales)]:
    if df is None:
        print(f" - {name}: MISSING")
    else:
        print(f" - {name}: loaded, shape={df.shape}")

# 2) Basic cleaning and sanity checks
if features is not None:
    features = parse_dates_in_df(features, 'Date')
    # Trim whitespace from column names
    features.columns = features.columns.str.strip()

if stores is not None:
    stores.columns = stores.columns.str.strip()

if sales is not None:
    # Some datasets use 'Weekly_Sales' or 'WeeklySales' etc. Normalise
    sales.columns = sales.columns.str.strip()
    # Attempt to parse Date
    sales = parse_dates_in_df(sales, 'Date')

# 3) Merge datasets (if possible)
merged = None
if sales is not None:
    if features is not None:
        merged = sales.merge(features, on=['Store', 'Date'], how='left', validate='m:1')
    else:
        merged = sales.copy()
    if stores is not None:
        merged = merged.merge(stores, on='Store', how='left', validate='m:1')
    print(f"Merged shape: {merged.shape}")
else:
    # sales missing — but we can at least inspect features & stores merge
    if features is not None and stores is not None:
        merged = features.merge(stores, on='Store', how='left', validate='m:1')
        print(f"No sales file — features x stores merged shape: {merged.shape}")

# 4) Exploratory Data Analysis helpers

def summarize_missing(df: pd.DataFrame):
    miss = df.isnull().sum().sort_values(ascending=False)
    pct = (miss / len(df)).sort_values(ascending=False)
    res = pd.concat([miss, pct], axis=1)
    res.columns = ['missing_count', 'missing_fraction']
    return res[res['missing_count'] > 0]


def plot_time_series_sample(df: pd.DataFrame, store:int=1, n_series=3):
    """Plot sample weekly sales for specified store (if available)"""
    if 'Weekly_Sales' not in df.columns:
        print('No Weekly_Sales column to plot')
        return
    tmp = df[df['Store'] == store].sort_values('Date')
    plt.figure(figsize=(12,4))
    plt.plot(tmp['Date'], tmp['Weekly_Sales'], marker='.', linewidth=1)
    plt.title(f'Store {store} Weekly Sales')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.tight_layout()
    plt.show()

# 5) If merged exists provide summary & missingness
if merged is not None:
    print('\nMerged head:')
    display(merged.head())
    print('\nMissing value summary:')
    display(summarize_missing(merged).head(20))

# 6) Feature engineering — create useful time features and flag for holidays
if merged is not None:
    df = merged.copy()
    # Ensure Date is datetime
    if 'Date' in df.columns:
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Day'] = df['Date'].dt.day
    # If 'IsHoliday' exists but not boolean ensure bool
    if 'IsHoliday' in df.columns:
        # sometimes holidays stored as string 'True' or 0/1
        df['IsHoliday'] = df['IsHoliday'].astype(bool)

    # Example: fill MarkDown columns with 0 (assuming NaN means no markdown)
    markdown_cols = [c for c in df.columns if c.startswith('MarkDown')]
    for c in markdown_cols:
        df[c] = df[c].fillna(0)

    # CPI and Unemployment may have occasional NaNs -> forward fill by store/time
    for c in ['CPI', 'Unemployment']:
        if c in df.columns:
            df[c] = df[c].fillna(method='ffill').fillna(method='bfill')

    # Convert categorical columns if present
    if 'Type' in df.columns:
        df['Type'] = df['Type'].astype('category')

    # Save cleaned merged
    cleaned_fp = OUTPUT_DIR / 'merged_cleaned.csv'
    df.to_csv(cleaned_fp, index=False)
    print(f"Cleaned merged dataset saved to: {cleaned_fp}")

# 7) Example EDA plots (only if sales exists)
if merged is not None and 'Weekly_Sales' in merged.columns:
    # Weekly sales distribution
    plt.figure(figsize=(8,4))
    sns.histplot(merged['Weekly_Sales'].clip(upper=merged['Weekly_Sales'].quantile(0.99)), bins=50)
    plt.title('Weekly Sales (99th percentile clipped)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'weekly_sales_dist.png')
    plt.show()

    # Sales over time (aggregated)
    ts = merged.groupby('Date')['Weekly_Sales'].sum().reset_index()
    plt.figure(figsize=(12,4))
    plt.plot(ts['Date'], ts['Weekly_Sales'])
    plt.title('Total Weekly Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Weekly Sales')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'total_weekly_sales_over_time.png')
    plt.show()

# 8) Simple baseline model (time-aware split) if sales available
if merged is not None and 'Weekly_Sales' in merged.columns:
    # Select feature columns
    target = 'Weekly_Sales'
    # Use a modest set of predictors
    predictors = []
    if 'Type' in merged.columns:
        predictors.append('Type')
    for c in ['Temperature','Fuel_Price','CPI','Unemployment']:
        if c in merged.columns:
            predictors.append(c)
    predictors += ['IsHoliday','Year','Month','DayOfWeek']
    # keep only those that exist
    predictors = [p for p in predictors if p in merged.columns]

    print(f"Using predictors: {predictors}")

    modeling_df = merged.dropna(subset=[target]).copy()
    # drop rows with missing predictors (simple approach)
    # instead create a preprocessing pipeline below to handle missingness

    # Create preprocessing
    numeric_features = [c for c in predictors if modeling_df[c].dtype.kind in 'bfi']
    categorical_features = [c for c in predictors if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Time series split
    tss = TimeSeriesSplit(n_splits=5)
    X = modeling_df[predictors]
    y = modeling_df[target]
    print('Running time-series cross validation (MAE)...')
    scores = -1 * cross_val_score(pipe, X, y, cv=tss, scoring='neg_mean_absolute_error', n_jobs=-1)
    print('MAE per split:', np.round(scores,2))
    print('MAE mean: {:.2f}, std: {:.2f}'.format(scores.mean(), scores.std()))

    # Fit on full data and show feature importances (approx via model)
    pipe.fit(X, y)
    # Extract feature names after preprocessing
    ohe_cols = []
    if categorical_features:
        ohe = pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        ohe_cols = list(ohe.get_feature_names_out(categorical_features))
    feature_names = numeric_features + ohe_cols
    importances = pipe.named_steps['model'].feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print('\nTop feature importances:')
    display(fi.head(15))
    # Save
    fi.head(50).to_csv(OUTPUT_DIR / 'feature_importances.csv')

else:
    print('\nSkipping modeling — Weekly_Sales column not available in merged data.')

# Final notes and outputs
print('\nFinished. Outputs (if produced) are in:', OUTPUT_DIR)
print('If sales_data.csv is missing, upload it to /mnt/data named exactly `sales_data.csv` and re-run to enable full analysis.')
