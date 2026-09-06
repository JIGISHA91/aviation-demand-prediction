import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# =====================================================================
# 1. LOAD REAL DATA
# =====================================================================
df = pd.read_csv("data/carrier.csv")

df = df[~df["Airline"].str.contains("Total", case=False, na=False)]

df = df.dropna(subset=[
    "Passenger Number", "Aircraft Kilometres",
    "Seat Kilometers", "Passenger Load Factor", "Airline", "Month", "Year"
])

# =====================================================================
# 2. ENCODE CATEGORICAL VARIABLES (Airline + Month)
# =====================================================================
df_encoded = pd.get_dummies(df, columns=["Airline", "Month"], drop_first=True)

feature_columns = [c for c in df_encoded.columns if c.startswith("Airline_") or c.startswith("Month_")] + [
    "Aircraft Kilometres", "Seat Kilometers", "Passenger Load Factor", "Year"
]

X = df_encoded[feature_columns]
y = df_encoded["Passenger Number"]

# =====================================================================
# 3. TRAIN-TEST SPLIT
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# =====================================================================
# 4. MODEL 1: LINEAR REGRESSION (Regression Analysis)
# =====================================================================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

lr_train_pred = lr_model.predict(X_train)
lr_train_r2 = r2_score(y_train, lr_train_pred)

print("--- Linear Regression ---")
print(f"Training R²: {lr_train_r2:.4f}")
print(f"Test R²: {lr_r2:.4f}")
print(f"MAE: {lr_mae:.2f}")
print(f"RMSE: {lr_rmse:.2f}")

joblib.dump(lr_model, "models/aviation_model.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")

# =====================================================================
# 5. MODEL 2: NEURAL NETWORK / MLP (Deep Learning)
# =====================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    max_iter=1000,
    random_state=42
)
nn_model.fit(X_train_scaled, y_train)
nn_pred = nn_model.predict(X_test_scaled)

nn_mae = mean_absolute_error(y_test, nn_pred)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))
nn_r2 = r2_score(y_test, nn_pred)

nn_train_pred = nn_model.predict(X_train_scaled)
nn_train_r2 = r2_score(y_train, nn_train_pred)

print("\n--- Neural Network (MLP) ---")
print(f"Training R²: {nn_train_r2:.4f}")
print(f"Test R²: {nn_r2:.4f}")
print(f"MAE: {nn_mae:.2f}")
print(f"RMSE: {nn_rmse:.2f}")

joblib.dump(nn_model, "models/aviation_nn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print(f"\nFinal dataset size: {len(df)} rows")
print("Both models and supporting files saved successfully.")