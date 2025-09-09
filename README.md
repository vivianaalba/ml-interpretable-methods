# ml-interpretable-methods
Exploration of interpretable ML models: decision trees, LASSO, boosting, and ensembles.

## Overview
This repository contains implementations and experiments with interpretable and ensemble machine learning methods, focusing on both regression and classification tasks. <br>

Methods include: <br>
- Decision Trees (with pruning + IF-THEN rule extraction) <br>
- LASSO Regression and Boosting <br>
- Random Forests with class imbalance handling <br>
- XGBoost and model trees <br>

The repo is organized around applied case studies using real-world datasets from the UCI Machine Learning Repository as uses the following Python libraries: pandas, numpy, seaborn, scipy, matplotlib, scikit-learn, xgboost, imblearn, scikit-multilearn <br>

## Methods & Experiments
1. Decision Trees as Interpretable Models <br>
- Trained decision trees on Acute Inflammations dataset. <br>
- Converted trees into IF-THEN rules. <br>
- Applied cost-complexity pruning for interpretable models. <br>

2. Regression with Regularization & Boosting <br>
- Applied Ridge, LASSO, PCR, and Boosted Trees on Communities & Crime dataset. <br>
- Compared feature selection (via LASSO) vs full models. <br>

3. Tree-Based Methods with Imbalance <br>
- Random Forests on APS Failure dataset. <br>
- Addressed missing values and class imbalance. <br>
- Compared baseline Random Forest vs balanced RF vs XGBoost with SMOTE. <br>

## Results
**1. Regression Models (Communities & Crime Dataset)**<br>

All Features<br>
- Linear Regression, Ridge, and LASSO all achieved similar performance with MSE ≈ 0.0181 and R² ≈ 0.62. <br>
- Principal Component Regression (28 PCs) yielded identical results to Linear Regression. <br>
- XGBoost outperformed linear models, achieving the lowest MSE (0.0168) and highest R² (0.647). <br>

Top Features (Selected via Coefficient of Variation) <br>
- Restricting to top CV features degraded performance for linear models (MSE ≈ 0.0305, R² ≈ 0.36). <br>
- XGBoost with feature selection achieved a notable improvement (MSE = 0.0237, R² = 0.50), but still underperformed compared to using all features. <br>

**Key Takeaway:** <br>
- Linear models struggled with feature selection, showing loss of explanatory power. <br>
- XGBoost consistently achieved the best test error and R², demonstrating its strength in handling complex feature interactions.<br>

**2. Ensemble Methods for Classification (APS Failure Dataset)**<br>

XGBoost <br>
- Without class balancing: Train Accuracy = 98.9%, Test Accuracy = 98.8%, CV error ≈ 1.3%. <br>
- With SMOTE oversampling: Slightly lower performance (Train = 96.6%, Test = 97.8%, CV error ≈ 3.4%) but improved handling of minority classes. <br>

Random Forests<br>
- Unbalanced RF achieved Test Accuracy = 99.2% with OOB error ≈ 0.62%. <br>
- Balanced Subsample RF performed similarly (Test Accuracy = 98.9%). <br>
- Balanced Random Forest (Imbalanced-Learn) reduced overfitting but lowered accuracy (Train = 95.3%, Test = 95.5%). <br>

**Key Takeaway:** <br>
- Random Forests (unbalanced) provided the highest raw accuracy, but balancing methods (SMOTE, BRF) helped improve fairness across classes. <br>
- XGBoost delivered consistently high performance, with strong generalization across train/test and cross-validation splits. <br>

## How To Run
1. Clone this repo: git clone https://github.com/vivianaalba/ml-interpretable-methods.git cd time-series-classification <br>
2. Install dependencies: pip install -r requirements.txt <br>
3. Run LR, PCA notebook: jupyter notebook lr_pcr_trees.ipynb <br>
4. Run feature trees, SMOTE, XGBOOST models notebook: jupyter notebook trees_xgboost.ipynb <br>