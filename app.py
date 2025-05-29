import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset (update the filename if needed)
df = pd.read_csv('Housing.csv')

# Drop rows with missing values
df = df.dropna()

# Simple Linear Regression: 'area' -> 'price'
X_simple = df[['area']]
y = df['price']

# Multiple Linear Regression: 'area', 'bedrooms', 'bathrooms' -> 'price'
X_multi = df[['area', 'bedrooms', 'bathrooms']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y, test_size=0.2, random_state=42)

# Fit models
lr_simple = LinearRegression()
lr_simple.fit(X_train, y_train)
lr_multi = LinearRegression()
lr_multi.fit(X_train_multi, y_train_multi)

# Evaluate models
# Simple
y_pred = lr_simple.predict(X_test)
print('Simple Linear Regression:')
print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('R2:', r2_score(y_test, y_pred))

# Multiple
y_pred_multi = lr_multi.predict(X_test_multi)
print('\nMultiple Linear Regression:')
print('MAE:', mean_absolute_error(y_test_multi, y_pred_multi))
print('MSE:', mean_squared_error(y_test_multi, y_pred_multi))
print('R2:', r2_score(y_test_multi, y_pred_multi))

# Plot regression line (simple regression)
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('SquareFeet')
plt.ylabel('Price')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

# Interpret coefficients
print('Simple Regression Coefficient:', lr_simple.coef_[0])
print('Simple Regression Intercept:', lr_simple.intercept_)
print('Multiple Regression Coefficients:', lr_multi.coef_)
print('Multiple Regression Intercept:', lr_multi.intercept_)
