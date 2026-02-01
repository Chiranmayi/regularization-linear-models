# Regularization in Linear Models

## Overview
This project demonstrates how regularization techniques help control overfitting in linear regression models using a real-world dataset. The study compares Multiple Linear Regression, Ridge Regression, Lasso Regression, and Elastic Net Regression by analyzing training and testing errors and visualizing coefficient shrinkage behavior.

The goal is to understand how different regularization penalties affect model complexity, generalization, and feature importance.

---

##  Objective
- To study overfitting in linear regression
- To apply regularization techniques (Ridge, Lasso, Elastic Net)
- To tune regularization strength using multiple values of alpha
- To compare model performance on training and testing data
- To visualize coefficient shrinkage paths

---

##  Dataset
- Dataset Used: California Housing Dataset (real-world)
- Source: `sklearn.datasets.fetch_california_housing`
- Target Variable:Median house value
- Features: Multiple numerical features related to housing statistics

This dataset:
- Contains multiple input features
- Requires feature scaling
- Is suitable for regression analysis

---

##  Models Implemented
1. Multiple Linear Regression
2. Ridge Regression (L2 Regularization)
3. Lasso Regression (L1 Regularization)
4. Elastic Net Regression (L1 + L2 Regularization)

---

##  Methodology
1. Load real-world housing dataset
2. Split data into training and testing sets (80/20)
3. Apply feature scaling using `StandardScaler`
4. Train baseline Linear Regression model
5. Train Ridge, Lasso, and Elastic Net models for different alpha values
6. Evaluate performance using Mean Squared Error (MSE)
7. Plot:
   - Training vs Testing error vs regularization strength
   - Coefficient shrinkage paths for each regularized model

---

##  Visualizations
The following plots are generated:
-  Training vs Testing Error Plot
  - Demonstrates how regularization reduces overfitting
-  Ridge Coefficient Shrinkage Plot
  - Shows smooth reduction of all coefficients
-  Lasso Coefficient Shrinkage Plot
  - Shows feature elimination (coefficients become zero)
-  Elastic Net Coefficient Shrinkage Plot
  - Shows balanced behavior between Ridge and Lasso

---

## Results & Observations
- Linear Regression shows signs of overfitting
- Ridge Regression reduces variance by shrinking coefficients
- Lasso Regression performs feature selection by eliminating weak features
- Elastic Net provides the best balance between bias and variance
- Regularization improves test performance and generalization

---

##  Conclusion
Regularization is essential when working with real-world datasets. Ridge, Lasso, and Elastic Net modify the loss function to control model complexity. Among the models tested, **Elastic Net performed best**, as it combines the strengths of both Ridge and Lasso while handling correlated features effectively.

---

## Technologies Used
- Python
- NumPy
- Matplotlib
- Scikit-learn

---

##  How to Run
```bash
python assignment2_regularization.py
