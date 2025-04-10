#  Travel Insurance Prediction

##  Overview
This project builds a robust machine learning model to predict whether a customer is likely to purchase a travel insurance package, based on their demographic and behavioral attributes. The dataset reflects offerings from a tour and travel company in India and includes coverage against COVID.

##  Objective
To classify potential customers who are more likely to buy travel insurance. The goal is to assist the company in optimizing its marketing strategy by targeting the right audience.

##  Dataset
The dataset includes ~2,000 previous customers with the following features:

- `Age`: Age of the customer
- `Employment Type`: Government or Private/Self-employed
- `GraduateOrNot`: Whether the customer is a college graduate
- `AnnualIncome`: Annual income in INR (rounded to nearest ₹50,000)
- `FamilyMembers`: Total family members
- `ChronicDiseases`: Presence of any major chronic diseases
- `FrequentFlyer`: Customer has flown ≥4 times in last 2 years
- `EverTravelledAbroad`: Whether customer has travelled internationally
- `TravelInsurance`: Target (1 = purchased insurance, 0 = did not)

##  Methodology

### 1. 🧹 Data Cleaning & Preprocessing
- Removed **738 duplicate rows**
- Binary and nominal encoding of categorical features
- Combined multiple levels of `ChronicDiseases` into a binary indicator
- Feature scaling for models like Logistic Regression
- **No SMOTE used**; `scale_pos_weight` in XGBoost is used to handle imbalance

### 2.  Exploratory Data Analysis (EDA)
- Univariate and bivariate distribution plots
- Annotated bar plots for class comparisons
- Correlation heatmap and VIF for multicollinearity checks

### 3.  Modeling
Built and evaluated multiple models:
- Logistic Regression
- Random Forest
- XGBoost (with `scale_pos_weight`)
- Gradient Boosting
- **Stacked Ensemble**:
  - Base models: Logistic Regression, RF, GB
  - Meta-learner: XGBoost

### 4.  Hyperparameter Tuning
- Bayesian Optimization using `BayesSearchCV`
- Tuned meta-learner with best-found parameters

### 5.  Evaluation Metrics
- Confusion Matrix
- Precision, Recall, F1-Score (focus on **weighted average** due to class imbalance)
- ROC AUC Score
- 5-Fold Stratified Cross-Validation

### 6.  Threshold Tuning
- Optimized decision threshold to improve F1-score
- Evaluated model performance across thresholds 0.1–0.9

### 7.  Feature Importance
- Gain-based feature importance from tuned XGBoost
- **Top predictors**:
  - `AnnualIncome`
  - `EverTravelledAbroad`
  - `FamilyMembers`
  - `Age`
  - `FrequentFlyer`

##  Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 73.6% |
| **Weighted Precision** | 0.742 |
| **Weighted Recall** | 0.736 |
| **Weighted F1-score** | 0.738 |
| **ROC AUC (Test)** | 0.789 |
| **Cross-Validated ROC AUC** | 0.763 ± 0.033 |

##  Business Insight
Customers more likely to purchase insurance:
- Earn higher incomes
- Have travelled abroad
- Belong to smaller families
- Are frequent flyers

**Strategic targeting** of these segments can improve marketing ROI and conversion rates.

##  How to Run

1. Clone the repository  
2. Create and activate a virtual environment  
3. Run `pip install -r requirements.txt`  
4. Launch the notebook: `TravelInsurancePrediction.ipynb`

## 🧩 Dependencies
See `requirements.txt` for complete package list and versions.

##  License
This project is for academic and educational purposes only.

##  Author
Created by **[Michael Bond](https://github.com/bondpapi)**  
📅 April 2025
"""