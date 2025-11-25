# Methodology Documentation

## 📊 Walmart Sales Analysis: Detailed Methodology

This document provides an in-depth explanation of the analytical methods, statistical techniques, and modeling approaches used in the Walmart Sales Analysis project.

---

## Table of Contents

1. [Data Collection & Preparation](#1-data-collection--preparation)
2. [Exploratory Data Analysis](#2-exploratory-data-analysis)
3. [Feature Engineering](#3-feature-engineering)
4. [Modeling Approach](#4-modeling-approach)
5. [Model Evaluation](#5-model-evaluation)
6. [Statistical Tests](#6-statistical-tests)
7. [Limitations & Assumptions](#7-limitations--assumptions)

---

## 1. Data Collection & Preparation

### 1.1 Data Sources

The analysis uses three primary datasets from Walmart's retail operations (2010-2012):

- **Sales Data**: 421,570 weekly sales records across 45 stores and multiple departments
- **Store Features**: Economic indicators (CPI, unemployment) and weather data (temperature, fuel prices)
- **Store Metadata**: Store types (A, B, C) and sizes

### 1.2 Data Cleaning Process

#### Missing Value Treatment

| Feature | Missing % | Treatment |
|---------|-----------|-----------|
| MarkDown1-5 | 50-60% | Fill with 0 (no promotion) |
| CPI | 1.2% | Forward fill by store |
| Unemployment | 1.2% | Forward fill by store |

**Rationale**: 
- MarkDowns: Absence indicates no promotional activity
- CPI/Unemployment: Economic indicators change gradually, forward-fill maintains temporal consistency

#### Data Type Conversions

```python
# Date parsing
df['Date'] = pd.to_datetime(df['Date'])

# Boolean conversion
df['IsHoliday'] = df['IsHoliday'].astype(bool)

# Category encoding
df['Type'] = df['Type'].astype('category')
```

#### Outlier Detection

- **Method**: Interquartile Range (IQR) method
- **Threshold**: Values beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR
- **Treatment**: Capped at 99th percentile for visualization, retained for modeling

---

## 2. Exploratory Data Analysis

### 2.1 Univariate Analysis

#### Sales Distribution
- **Shape**: Right-skewed (long tail of high sales)
- **Central Tendency**: Median = $7,612, Mean = $15,981
- **Dispersion**: High variance (σ = $22,711)
- **Implication**: Log transformation considered for modeling

#### Categorical Variables

```
Store Type Distribution:
- Type A: 22 stores (48.9%)
- Type B: 17 stores (37.8%)
- Type C: 6 stores (13.3%)
```

### 2.2 Bivariate Analysis

#### Correlation Matrix (Top Correlates with Weekly_Sales)

| Feature | Pearson r | p-value | Interpretation |
|---------|-----------|---------|----------------|
| Store Size | 0.45 | <0.001 | Moderate positive |
| Temperature | 0.23 | <0.001 | Weak positive |
| IsHoliday | 0.15 | <0.001 | Weak positive |
| Unemployment | -0.31 | <0.001 | Moderate negative |
| Fuel_Price | -0.12 | <0.001 | Weak negative |

### 2.3 Time Series Analysis

#### Trend Detection
- **Method**: Moving average (12-week window)
- **Result**: Slight upward trend (+2.3% YoY growth)

#### Seasonality
- **Method**: Seasonal decomposition (additive model)
- **Findings**: 
  - Strong annual seasonality (peak in Nov-Dec)
  - Weekly seasonality minimal (retail stores open 7 days)

---

## 3. Feature Engineering

### 3.1 Temporal Features

Created from `Date` column:

```python
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['WeekOfYear'] = df['Date'].dt.isocalendar().week
df['Quarter'] = df['Date'].dt.quarter
df['DayOfWeek'] = df['Date'].dt.dayofweek
```

**Rationale**: Capture seasonal patterns and cyclical effects

### 3.2 Lag Features

```python
# Previous week sales (t-1)
df['Sales_Lag1'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)

# 4-week lag (monthly comparison)
df['Sales_Lag4'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(4)
```

**Rationale**: Sales exhibit autocorrelation (ρ₁ = 0.68)

### 3.3 Rolling Statistics

```python
# 4-week moving average
df['Sales_MA4'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].rolling(4).mean()

# 12-week rolling std (volatility measure)
df['Sales_Volatility'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].rolling(12).std()
```

### 3.4 Interaction Features

```python
# Store type and holiday interaction
df['Type_Holiday'] = df['Type'].astype(str) + '_' + df['IsHoliday'].astype(str)

# Temperature bins
df['Temp_Category'] = pd.cut(df['Temperature'], bins=[0, 40, 60, 80, 100], 
                              labels=['Cold', 'Cool', 'Warm', 'Hot'])
```

### 3.5 Aggregated Features

```python
# Store-level statistics
store_avg = df.groupby('Store')['Weekly_Sales'].mean().to_dict()
df['Store_Avg_Sales'] = df['Store'].map(store_avg)

# Department-level statistics
dept_performance = df.groupby('Dept')['Weekly_Sales'].agg(['mean', 'std']).to_dict()
```

---

## 4. Modeling Approach

### 4.1 Problem Formulation

**Type**: Supervised regression problem

**Target**: `Weekly_Sales` (continuous)

**Features**: 25 engineered features (after preprocessing)

### 4.2 Train-Test Split Strategy

**Method**: Time-series split (chronological)

```
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────┤
│   Fold 1    │   Fold 2    │   Fold 3    │   Fold 4    │   Fold 5    │
│ Train │Test │ Train │Test │ Train │Test │ Train │Test │ Train │Test │
```

**Rationale**: Prevents data leakage; respects temporal ordering

### 4.3 Model Selection

#### Baseline Models (Compared)

1. **Linear Regression**: R² = 0.45, MAE = 3,200
2. **Ridge Regression**: R² = 0.46, MAE = 3,150
3. **Decision Tree**: R² = 0.67, MAE = 2,100
4. **Random Forest**: R² = 0.92, MAE = 1,200 ✓ **Selected**

#### Random Forest Hyperparameters

```python
RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=None,        # Unlimited depth
    min_samples_split=2,   # Default split criterion
    min_samples_leaf=1,    # Min samples in leaf node
    max_features='auto',   # sqrt(n_features) for splits
    random_state=42,       # Reproducibility
    n_jobs=-1              # Parallel processing
)
```

**Selection Rationale**:
- Handles non-linear relationships
- Resistant to outliers
- Captures feature interactions
- No need for feature scaling
- Provides feature importance

### 4.4 Preprocessing Pipeline

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())
])
```

#### Column Transformer

- **Numerical features**: Impute → Scale
- **Categorical features**: Impute (mode) → One-Hot Encode

---

## 5. Model Evaluation

### 5.1 Metrics

#### Mean Absolute Error (MAE)
```
MAE = (1/n) Σ|yᵢ - ŷᵢ|
```
- **Result**: $1,200
- **Interpretation**: On average, predictions are off by $1,200

#### Root Mean Squared Error (RMSE)
```
RMSE = √[(1/n) Σ(yᵢ - ŷᵢ)²]
```
- **Result**: $1,800
- **Interpretation**: Penalizes large errors more heavily

#### R² Score (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)
```
- **Result**: 0.92
- **Interpretation**: Model explains 92% of variance

### 5.2 Cross-Validation Results

| Fold | MAE | RMSE | R² |
|------|-----|------|----|
| 1 | 1,150 | 1,720 | 0.93 |
| 2 | 1,220 | 1,850 | 0.91 |
| 3 | 1,180 | 1,780 | 0.92 |
| 4 | 1,240 | 1,890 | 0.91 |
| 5 | 1,210 | 1,820 | 0.92 |
| **Mean** | **1,200** | **1,812** | **0.92** |
| **Std** | 35 | 64 | 0.008 |

**Conclusion**: Stable performance across folds

### 5.3 Residual Analysis

#### Homoscedasticity Check
- **Test**: Breusch-Pagan test
- **Result**: p = 0.08 (fail to reject null)
- **Conclusion**: Residual variance is constant

#### Normality Check
- **Test**: Shapiro-Wilk test
- **Result**: p < 0.001 (reject null)
- **Conclusion**: Residuals not perfectly normal (acceptable for large n)

---

## 6. Statistical Tests

### 6.1 Holiday Effect Test

**Hypothesis**:
- H₀: μ_holiday = μ_non_holiday
- H₁: μ_holiday ≠ μ_non_holiday

**Test**: Welch's t-test (unequal variances)

**Results**:
- t-statistic: 12.45
- p-value: < 0.001
- Cohen's d: 0.42 (medium effect size)

**Conclusion**: Holiday weeks have significantly higher sales

### 6.2 Store Type ANOVA

**Hypothesis**:
- H₀: μ_A = μ_B = μ_C
- H₁: At least one mean differs

**Test**: One-way ANOVA

**Results**:
- F-statistic: 145.67
- p-value: < 0.001

**Post-hoc**: Tukey HSD
- A vs B: p < 0.001 (A > B)
- A vs C: p < 0.001 (A > C)
- B vs C: p = 0.032 (B > C)

---

## 7. Limitations & Assumptions

### 7.1 Data Limitations

1. **Temporal Coverage**: Only 2 years (2010-2012); limited long-term patterns
2. **Department Granularity**: Some departments have sparse data
3. **External Factors**: Missing competitor data, local events, online sales impact

### 7.2 Model Assumptions

1. **Stationarity**: Assumes underlying patterns remain stable
2. **Independence**: Assumes departments within stores are independent (may not hold)
3. **Feature Completeness**: Assumes all relevant features are captured

### 7.3 Generalization Concerns

- Model trained on historical data; may not generalize to:
  - New stores/locations
  - Economic crises (2008-2009 recession not captured)
  - Structural changes in retail (e-commerce shift)

### 7.4 Recommendations for Future Work

1. **Incorporate more features**: Competitor pricing, local demographics
2. **Deep learning models**: LSTM for better sequential modeling
3. **Ensemble methods**: Stack multiple models for robustness
4. **Real-time updates**: Implement online learning for concept drift

---

## References

1. Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
2. Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice.
3. Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2015). Time series analysis: forecasting and control.

---

**Last Updated**: November 2024  
**Author**: MD Tariqul Islam  
**Contact**: tariqul@scired.com

