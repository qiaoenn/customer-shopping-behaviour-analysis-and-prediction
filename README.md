# customer-shopping-behaviour-analysis-and-prediction

This project explores a synthetic consumer shopping dataset from Kaggle using Python and Jupyter Notebook.  
The goal is to perform end-to-end Exploratory Data Analysis (EDA) and build regression models to predict **Purchase Amount (USD)**, while critically assessing the **limitations of the dataset** and the **failure of machine learning models when there is little or no signal in the data**.


## 1. Project Overview

- **Objective:**  
Investigate whether customer purchase amount can be accurately predicted from demographic, behavioural and transactional features (age, gender, location, season, item purchased, frequency, subscription status, etc.).

- **Key Tasks:**
  1. Data loading, cleaning and encoding (binary, ordinal and one-hot encoding)
  2. Exploratory Data Analysis (EDA) and visualisation
  3. Modelling:
     - Linear Regression with Lasso feature selection
     - Random Forest Regressor (`feature_importances_`)
     - XGBoost Regressor (gain-based feature importance)
  4. Model evaluation using **RMSE** and **R²**
  5. Critical interpretation of why models perform poorly on this dataset

---

## 2. Dataset

- **Source:** Kaggle – consumer shopping behaviour dataset  
- **Target variable:** `Purchase Amount (USD)`  
- **Example features:**
  - `Age`, `Gender`, `Location`
  - `Category`, `Item Purchased`, `Color`, `Size`, `Season`
  - `Payment Method`, `Shipping Type`
  - `Subscription Status`, `Discount Applied`, `Promo Code Used`
  - `Previous Purchases`, `Frequency of Purchases`, `Review Rating`

The dataset appears to be **synthetically generated**. EDA shows that many variables are uniformly distributed and weakly related (or unrelated) to the target.

---

## 3. Methods

### 3.1 Preprocessing

- Loaded data with `pandas`
- Dropped non-informative identifiers (e.g. `Customer ID`)
- **Binary encoding:**  
  `Yes/No` → `1/0` for variables like `Subscription Status`, `Discount Applied`, `Promo Code Used`
- **Ordinal encoding:**  
  Encoded `Frequency of Purchases` into numerical values (e.g. Annually, Quarterly, Monthly, Weekly, Bi-Weekly)
- **One-hot encoding:**  
  Applied to categorical features such as `Gender`, `Category`, `Size`, `Color`, `Season`, `Shipping Type`, `Payment Method`, `Item Purchased`, `Location`
- Split data into **train** and **test** sets using `train_test_split`

### 3.2 Exploratory Data Analysis (EDA)

- Descriptive statistics (`df.info()`, `df.describe()`)
- Distribution plots (histograms, bar chart, pie chart)

**Key EDA Finding:**  
The target (`Purchase Amount (USD)`) is approximately **uniformly distributed** between \$20 and \$100, with **no strong linear or nonlinear relationships** observed with any feature. Many features are also uniformly or evenly distributed (e.g. locations, colours, seasons).

---

## 4. Models

### 4.1 Linear Regression with Lasso Feature Selection

- Used a `Pipeline(StandardScaler + LassoCV)` to:
  - Standardise features
  - Perform L1-regularised regression with cross-validated alpha
- Extracted Lasso-selected features and refit a simpler Linear Regression model.
- Evaluated using **RMSE** and **R²** on the test set.

### 4.2 Random Forest Regressor

- Trained `RandomForestRegressor` with 300 trees.
- Computed:
  - Test set **RMSE** and **R²**
  - `feature_importances_` to assess which variables the forest relied on.
- Feature importance values were all relatively small, with no clearly dominant predictor.

### 4.3 XGBoost Regressor

- Trained `XGBRegressor` with:
  - `n_estimators=300`, `learning_rate=0.05`, `max_depth=6`,
    `subsample=0.8`, `colsample_bytree=0.8`
- Evaluated using **RMSE** and **R²**.
- Extracted gain-based feature importance using:
  - `xgb.get_booster().get_score(importance_type='gain')`
- Top features were mostly location and item dummy variables, but gain values were close together, suggesting overfitting to noise.

---

## 5. Results

Across all models:

- **R² scores were near zero or negative**, indicating performance **worse than a baseline** that simply predicts the mean purchase amount.
- **RMSE** remained high relative to the target range (\$20–\$100).
- EDA and feature importance analyses both show:
  - No strong signal in the data
  - No feature with a clear, stable relationship to purchase amount
  - Synthetic, uniformly distributed variables that resemble random noise

---

## 6. Key Insights

1. **Data Quality Matters More Than Model Complexity**  
   Even advanced models like Random Forest and XGBoost cannot recover meaningful patterns when the dataset contains little or no predictive signal.

2. **EDA is Essential**  
   Visual and statistical exploration revealed flat distributions and near-zero correlations, warning that modelling would likely perform poorly.

3. **Feature Importance Can Be Misleading on Noisy Data**  
   Random Forest and XGBoost produced ranked feature importances, but due to noise, these rankings do not reflect true causal relationships.
