# ================================
# CAR PRICE PREDICTION PROJECT
# ================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# ================================
# LOAD DATASET
# ================================

df = pd.read_csv("car data.csv")

print(df.head())
print(df.info())

# ================================
# DATA PREPROCESSING
# ================================

# Check missing values
print(df.isnull().sum())

# Encode categorical columns
le = LabelEncoder()

categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ================================
# FEATURE SELECTION
# ================================

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# ================================
# TRAIN TEST SPLIT
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ================================
# MODEL TRAINING
# ================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ================================
# PREDICTION
# ================================

y_pred = model.predict(X_test)

# ================================
# MODEL EVALUATION
# ================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("---------------------")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# ================================
# VISUALIZATION
# ================================

plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()

# ================================
# FEATURE IMPORTANCE
# ================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importance)
plt.title("Feature Importance")
plt.show()

# ================================
# SAVE MODEL
# ================================

import joblib

joblib.dump(model, "car_price_model.pkl")

print("\nModel Saved Successfully!")