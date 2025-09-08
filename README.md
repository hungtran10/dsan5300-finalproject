# Predicting Food Inspection Outcomes in Chicago

This project aims to support public health decision-making by predicting the outcomes of food safety inspections in Chicago. Using open data from the City of Chicago, we built a machine learning pipeline to forecast whether food establishments are likely to pass or fail inspections, both **before** and **after** an inspection occurs.

---

## Project Overview

Chicago’s Department of Public Health conducts thousands of food inspections annually. These inspections determine whether establishments comply with health regulations. However, the city must allocate limited resources across a large number of facilities.

Our goal was to use historical inspection and licensing data to:

* **Before inspection**: Predict if an establishment will fail **prior to the visit**, based on past behavior and business characteristics.
* **After inspection begins**: Predict if an establishment will fail **during the visit**, using partial information such as the number of violations recorded so far.

By making accurate predictions in both scenarios, the city can more efficiently schedule inspections, prioritize high-risk establishments, and intervene earlier to prevent food safety hazards.

---

## Data Collection

We used four primary datasets from the [City of Chicago Data Portal](https://data.cityofchicago.org/):

* Food Inspections: City of Chicago
* Census & ACS data: Community area demographics
* GIS Boundaries: Community area mapping
* Crime Data: Aggregated by community area

---

## Data Cleaning

Key data cleaning steps included:

* **Standardization**: Merged and cleaned categorical columns of type string such as facility_type and city
* **Merging**: Inspections data was enriched by joining inspection records with neighborhood-level crime and demographic data. 
* **Imputation**: Filled missing values using logical defaults or simple statistical methods (e.g., imputed missing zip codes with mode).
* **Feature engineering**: 

  * Temporal features: inspection year, month, season, weekday/weekend, and day of the week were extracted
  * Geographic and demographic characteristics
  * High cardinality features were grouped
  * Categorical variables were one-hot encoded

---

## Model Selection

We framed two separate binary classification tasks:

### 1. **Pre-Inspection Prediction**

Predict the outcome **before the inspection occurs**, using only information known in advance.

* **Features**: Facility metadata, violation codes, critical-item flags, historical inspection records.
* **Models Evaluated**: Logistic Regression, Support Vector Machine (SVM), XGBoost.
* **Best Model**: XGBoost, due to its strong performance with categorical variables and class imbalance.

### 2. **Post-Inspection Prediction**

Predict the outcome **once an inspection begins**, based on partial inspection data.

* **Additional Features**: Number and type of violations observed so far, inspection type, inspector ID.
* **Models Evaluated**: Same as above, with emphasis on handling high-cardinality features (e.g., inspector).
* **Best Model**: Again, XGBoost performed best, leveraging the richer post-visit feature set.

All models were trained with stratified 10-fold cross-validation for a 60000-row subset of the dataset and optimized using grid search.

---

## Model Evaluation

### **Pre-Inspection Task:** XGBoost

* **AUC-ROC**: \~0.647
* **Accuracy**: \~69%
* **Top Predictors**: Facility_Type_Nan, count_deceptive_practice_per_100k, pct_black_or_african_american


### **Post-Inspection Task**

* **AUC-ROC**: \~0.914
* **Accuracy**: \~83%
* **Top Predictors**: first_violation_code_18, violation_level_Low, violation_level_None

---

## Conclusion

This project demonstrates the feasibility of using publicly available municipal data to predict food inspection outcomes in Chicago. Both **pre-** and **post-inspection** models achieved moderate to moderately high performance and could inform real-world inspection scheduling:

* **Pre-inspection model**: Can flag high-risk establishments before visits, supporting proactive resource allocation.
* **Post-inspection model**: Can assist inspectors or automated systems in determining likely outcomes in real-time.

Future extensions could incorporate natural language processing of violation descriptions, time-series forecasting of inspection demand, or integration into a decision-support dashboard for city officials.

