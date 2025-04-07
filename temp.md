# Travel Insurance Prediction

## Overview
This project aims to build an intelligent machine learning model that predicts whether a customer will be interested in purchasing a travel insurance package. The data is based on historical offerings by a tour and travel company, including a COVID coverage extension.

## Objective
To classify customers who are likely to purchase travel insurance using demographic and behavioral features. This helps the company target potential customers more effectively.

## Dataset
The dataset consists of ~2000 customers and includes the following features:

- `Age`: Age of the customer
- `Employment Type`: Sector of employment
- `GraduateOrNot`: Graduation status
- `AnnualIncome`: Yearly income in INR
- `FamilyMembers`: Number of members in the family
- `ChronicDiseases`: Presence of major chronic diseases
- `FrequentFlyer`: Whether the customer frequently flies
- `EverTravelledAbroad`: Has the customer ever travelled abroad
- `TravelInsurance`: Target variable (1 = bought insurance, 0 = did not buy)

## Methodology

### 1. Data Cleaning & Preprocessing
- Categorical encoding using `LabelEncoder`
- Handling class imbalance using **SMOTE**
- Standard scaling of features

### 2. Exploratory Data Analysis (EDA)
- Univariate & bivariate analysis
- Class distribution & imbalance check
- Correlation heatmap
- Multicollinearity analysis using **VIF and Tolerance**

### 3. Modeling
We implemented and evaluated several classification models:

- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting
- Stacked Ensemble (base: RF, GB; meta: XGBoost)

### 4. Hyperparameter Tuning
- Used **Bayesian Optimization** for optimal XGBoost parameters
- Final meta-learner trained with tuned parameters in the stacking classifier

### 5. Evaluation Metrics
- Confusion Matrix
- Precision, Recall, F1-Score (macro, weighted)
- ROC AUC Score
- 5-Fold Stratified Cross-Validation

### 6. Threshold Optimization
- Adjusted classification threshold from default (0.5) to best-performing based on ROC curve

### 7. Feature Importance
- Gain-based importance from tuned XGBoost meta-learner revealed:
  - Top predictors: **AnnualIncome**, **EverTravelledAbroad**, **FamilyMembers**, **Age**, **FrequentFlyer**

## Results
- Final stacked model (tuned + threshold optimized):
  - **Accuracy**: 71.2%
  - **AUC**: ~0.716
  - **Weighted F1 Score**: ~0.702

## Business Insight
Customers with higher income, prior international travel, and chronic conditions are more likely to buy travel insurance. Targeting such customers can improve marketing efficiency and conversion.

## Project Structure
```
travel_insurance_prediction/
├── TravelInsurancePrediction.csv
├── TravelInsurancePrediction.ipynb
├── requirements.txt
└── README.md
```

## How to Run
1. Clone this repo
2. Create and activate virtual environment
3. Run: `pip install -r requirements.txt`
4. Open the notebook: `TravelInsurancePrediction.ipynb`

## Dependencies
See `requirements.txt` for all used packages and versions.

## License
This project is for academic and educational purposes.

---

## Author
Created by [Michael Bond / https://github.com/bondpapi]

**Date:** April 2025