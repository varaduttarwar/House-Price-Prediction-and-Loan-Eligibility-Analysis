# 🏠📊 House Price Prediction & Loan Eligibility Analysis

This project aims to develop machine learning models for **predicting house prices** based on real estate features and **assessing loan eligibility** based on applicant financial data. It leverages regression models for house pricing and classification models for loan eligibility — ensuring accurate, explainable, and reliable predictions.

---

## 📑 Table of Contents
- [📖 Abstract](#-abstract)
- [📌 Introduction](#-introduction)
- [🗂️ Dataset Description](#-dataset-description)
- [🛠️ Preprocessing Techniques](#-preprocessing-techniques)
- [⚙️ Methodology](#-methodology)
- [📊 Final Results](#-final-results)
- [✅ Conclusion](#-conclusion)
- [📚 References](#-references)

---

## 📖 Abstract

This project focuses on applying machine learning models for:
- Predicting house prices based on property and location features.
- Assessing loan eligibility based on applicant financial details.

Models were evaluated for accuracy and explainability, with deployment using **Flask/Streamlit** for interactive use.

---

## 📌 Introduction

Real estate pricing and loan approvals are pivotal in the financial sector. This project proposes a **data-driven approach** using ML algorithms for:
- Better **price prediction** to assist buyers.
- More accurate **loan eligibility approvals** for lenders.

---

## 🗂️ Dataset Description

### 📈 Loan Eligibility Dataset
- **Total Entries**: 614
- **Key Features**: Loan ID, Gender, Married, Education, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan Amount Term, Credit History, Property Area
- **Target**: Loan Status (Y/N)
- **Missing Values**: LoanAmount (22), Credit History (50), and other categorical fields.

### 🏠 House Price Prediction Dataset
- **Total Entries**: 21,614
- **Key Features**: Bedrooms, Bathrooms, Flat Area, Lot Area, Floors, Waterfront View, Condition, Overall Grade, Latitude, Longitude
- **Target**: Sale Price
- **Missing Values**: Sale Price (4), some structural and locational features.

---

## 🛠️ Preprocessing Techniques

### 🏠 House Price Prediction
- **Handled Missing Values**: Median imputation.
- **Categorical Encoding**: One-hot and label encoding.
- **Feature Scaling**: `StandardScaler`.
- **Feature Selection**: Removed IDs and redundant columns.

### 📈 Loan Eligibility Prediction
- **Missing Value Handling**: Median imputation.
- **Label Encoding**: Categorical fields.
- **Custom Mapping**: `Education` (Graduate=1, Not Graduate=0).
- **Correlation Analysis**: Heatmap for feature relationship understanding.
- **Feature Engineering (Optional)**: Debt-to-income ratio, etc.

---

## ⚙️ Methodology

### 🏠 House Price Prediction
**Models Used:**
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost (Gradient Boosting)

**Evaluation Metrics:**
- MAE
- RMSE
- R² Score

---

### 📈 Loan Eligibility Prediction
**Models Used:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Voting Classifier (Ensemble)

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score

---

## 📊 Final Results

### 🏠 House Price Prediction Model Comparison

| Model               | MAE     | RMSE     | R² Score |
|:--------------------|:----------|:----------|:------------|
| Linear Regression    | 131,898   | 213,225   | 0.688      |
| Decision Tree        | 104,597   | 206,571   | 0.707      |
| Gradient Boosting    | 80,913    | 142,283   | 0.861      |
| **Random Forest**     | **71,774**    | **134,138**   | **0.877**      |

---

### 📈 Loan Eligibility Prediction Model Comparison

| Model                     | Accuracy | Precision | Recall | F1-Score |
|:---------------------------|:----------|:------------|:---------|:------------|
| Logistic Regression        | 78%       | 0.75       | 0.82    | 0.78       |
| Decision Tree              | 76%       | 0.74       | 0.79    | 0.76       |
| Random Forest              | 81%       | 0.80       | 0.85    | 0.82       |
| **Voting Classifier** (Ensemble) | **80%**       | **0.79**       | **0.96**    | **0.87**       |

---

## ✅ Conclusion

This project successfully developed ML models for:
- **House Price Prediction:** Random Forest Regressor achieved the best R² score.
- **Loan Eligibility Prediction:** Voting Classifier achieved the highest recall and F1-Score.

**Future enhancements**:
- Hyperparameter tuning
- Integration of geospatial/temporal features
- Web application deployment via **Flask** or **Streamlit**

---

## 📚 References
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Kaggle Datasets](https://www.kaggle.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---
## 👨‍💻 Author
**Varad Uttarwar**  
[GitHub](https://github.com/varaduttarwar)

---

