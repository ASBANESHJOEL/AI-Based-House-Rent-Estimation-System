
# =========================
# Import Libraries
# =========================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =========================
# Load Dataset
# =========================

df = pd.read_excel("RentalDS.xlsx")

# Remove unwanted spaces from column names
df.columns = df.columns.str.strip()

# =========================
# Exploratory Data Analysis
# =========================

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# =========================
# Check Missing Values
# =========================

print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# Define Features and Target
# =========================

X = df[['LOCALITY', 'BHK', 'SQ. FT', 'CONDITION']]
y = df['RENT']

# =========================
# Split Dataset
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =========================
# Data Preprocessing
# =========================

categorical_features = ['LOCALITY']

preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_features
        )
    ],
    remainder='passthrough'
)

# =========================
# Build Random Forest Model
# =========================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# =========================
# Create ML Pipeline
# =========================

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# =========================
# Train Model
# =========================

pipeline.fit(X_train, y_train)

# =========================
# Make Predictions
# =========================

y_pred = pipeline.predict(X_test)

# =========================
# Evaluate Model
# =========================

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance")
print("=" * 30)
print(f"R² Score : {r2:.4f}")
print(f"MAE      : ₹{mae:,.2f}")
print(f"RMSE     : ₹{rmse:,.2f}")
