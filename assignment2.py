import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

# -----------------------------
# Load REAL-WORLD dataset
# -----------------------------
data = fetch_california_housing()
X = data.data
y = data.target

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Feature Scaling (MANDATORY)
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Baseline: Linear Regression
# -----------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)

print("Linear Regression Test MSE:",
      mean_squared_error(y_test, lr.predict(X_test)))

# -----------------------------
# Regularization Models
# -----------------------------
alphas = np.logspace(-3, 2, 10)

ridge_train, ridge_test = [], []
lasso_train, lasso_test = [], []
enet_train, enet_test = [], []

ridge_coefs, lasso_coefs, enet_coefs = [], [], []

for a in alphas:
    # Ridge
    ridge = Ridge(alpha=a)
    ridge.fit(X_train, y_train)
    ridge_train.append(mean_squared_error(y_train, ridge.predict(X_train)))
    ridge_test.append(mean_squared_error(y_test, ridge.predict(X_test)))
    ridge_coefs.append(ridge.coef_)

    # Lasso
    lasso = Lasso(alpha=a, max_iter=5000)
    lasso.fit(X_train, y_train)
    lasso_train.append(mean_squared_error(y_train, lasso.predict(X_train)))
    lasso_test.append(mean_squared_error(y_test, lasso.predict(X_test)))
    lasso_coefs.append(lasso.coef_)

    # Elastic Net
    enet = ElasticNet(alpha=a, l1_ratio=0.5, max_iter=5000)
    enet.fit(X_train, y_train)
    enet_train.append(mean_squared_error(y_train, enet.predict(X_train)))
    enet_test.append(mean_squared_error(y_test, enet.predict(X_test)))
    enet_coefs.append(enet.coef_)

# -----------------------------
# Plot 1: Training vs Testing Error
# -----------------------------
plt.figure()
plt.plot(alphas, ridge_train, label="Ridge Train")
plt.plot(alphas, ridge_test, label="Ridge Test")
plt.plot(alphas, lasso_train, label="Lasso Train")
plt.plot(alphas, lasso_test, label="Lasso Test")
plt.plot(alphas, enet_train, label="ElasticNet Train")
plt.plot(alphas, enet_test, label="ElasticNet Test")

plt.xscale("log")
plt.xlabel("Alpha (Regularization Strength)")
plt.ylabel("MSE")
plt.legend()
plt.title("Training vs Testing Error")
plt.show()

# -----------------------------
# Plot 2: Ridge Coefficient Shrinkage
# -----------------------------
plt.figure()
for coef in np.array(ridge_coefs).T:
    plt.plot(alphas, coef)
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Coefficient Value")
plt.title("Ridge Coefficient Shrinkage")
plt.show()

# -----------------------------
# Plot 3: Lasso Coefficient Shrinkage
# -----------------------------
plt.figure()
for coef in np.array(lasso_coefs).T:
    plt.plot(alphas, coef)
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Coefficient Value")
plt.title("Lasso Coefficient Shrinkage")
plt.show()

# -----------------------------
# Plot 4: Elastic Net Coefficient Shrinkage
# -----------------------------
plt.figure()
for coef in np.array(enet_coefs).T:
    plt.plot(alphas, coef)
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Coefficient Value")
plt.title("Elastic Net Coefficient Shrinkage")
plt.show()


