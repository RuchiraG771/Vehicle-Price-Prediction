import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
data = pd.read_csv("dataset/vehicle_data.csv")

# Clean columns
data.columns = data.columns.str.lower().str.strip()
data.dropna(inplace=True)

# Check target column
assert "price" in data.columns, "❌ price not found!"

# Feature engineering
data["vehicle_age"] = 2025 - data["year"]
data.drop(["year", "name"], axis=1, errors="ignore", inplace=True)

# Encode categoricals
encoders = {}
for col in data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Split
X = data.drop("price", axis=1)
y = data["price"]

print("Target price range:")
print(y.describe())   # 🔴 IMPORTANT DEBUG

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Feature importance check
importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importance:")
print(importance.sort_values(ascending=False))

# Save
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")
joblib.dump(encoders, "encoders.pkl")

print("\n✅ New model trained & saved successfully") 
import os
print("\nFiles in directory:", os.listdir())
 
