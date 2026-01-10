import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------
# Load Dataset
# ---------------------------
data = pd.read_csv("dataset/vehicle_data.csv")

# Standardize column names
data.columns = data.columns.str.lower().str.strip()

# ---------------------------
# Drop text columns if present
# ---------------------------
for col in ["name", "car_name", "model"]:
    if col in data.columns:
        data.drop(col, axis=1, inplace=True)

# ---------------------------
# Handle missing values
# ---------------------------
data.dropna(inplace=True)

# ---------------------------
# Feature Engineering
# ---------------------------
if "year" in data.columns:
    data["vehicle_age"] = 2025 - data["year"]
    data.drop("year", axis=1, inplace=True)

# ---------------------------
# Encode categorical columns
# ---------------------------
cat_cols = data.select_dtypes(include="object").columns.tolist()

le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

# ---------------------------
# Detect target column
# ---------------------------
possible_targets = ["selling_price", "price", "sellingprice"]

target = None
for col in possible_targets:
    if col in data.columns:
        target = col
        break

if target is None:
    raise ValueError("❌ Target column not found!")

# ---------------------------
# Split features & target
# ---------------------------
X = data.drop(target, axis=1)
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Train model
# ---------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------
# Evaluation
# ---------------------------
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# ---------------------------
# Save Model & Feature Names
# ---------------------------
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

print("✅ model.pkl and features.pkl saved successfully")

