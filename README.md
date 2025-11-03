# Credit Default Risk Prediction

This project predicts the probability of credit card default using TabNet and several baseline machine learning models.

Dataset: Default of Credit Card Clients (UCI Machine Learning Repository)
Each record contains demographic and financial information about a borrower.
The target variable is `default_payment_next_month` (1 = default, 0 = non-default).

## 1. Project structure:

tabnet_application/
  default-of-credit-card-clients-predictive-models.ipynb   - Jupyter notebook for training and evaluation
  default_of_credit_card_clients.csv                       - Dataset file
  tabnet_credit_default.zip                                - Trained model file
  catboost_info/                                           - CatBoost logs and auxiliary data
  README.md                                                - This documentation

Models used:
  - Logistic Regression
  - Random Forest
  - XGBoost / CatBoost
  - TabNet (main model)

TabNet combines tree-like feature selection with neural network learning, designed for tabular structured data.

## 2. Main workflow:
1. Data preprocessing
   - Handle missing values
   - Normalize numeric features
   - Encode categorical variables

2. Model training
   - Split dataset into training and test sets
   - Train models, tune hyperparameters
   - Evaluate using standard classification metrics

3. Model evaluation
   - Compare baseline models with TabNet
   - Analyze feature importance and interpretability

## 3. Requirements:
  pip install pandas numpy torch pytorch-tabnet scikit-learn matplotlib

## 4. Usage:
1. Open and run the Jupyter notebook file:
   jupyter notebook default-of-credit-card-clients-predictive-models.ipynb
2. Follow the steps inside the notebook to train and evaluate models.

## 5. Author:
Yankai Zhao

