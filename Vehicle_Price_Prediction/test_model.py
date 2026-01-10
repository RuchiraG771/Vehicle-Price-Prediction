import joblib
import pandas as pd
import os

# --------------------------------
# Load model & feature list
# --------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))

# --------------------------------
# Sample input as DICTIONARY
# (keys must match training column names)
# --------------------------------
input_dict = {
    "km_driven": 45000,
    "mileage": 18.5,
    "engine": 1197,
    "max_power": 82,
    "seats": 5,
    "fuel": 2,          # Petrol
    "seller_type": 1,   # Individual
    "transmission": 1,  # Manual
    "owner": 0,         # First owner
    "vehicle_age": 6
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# --------------------------------
# Align input with training features
# --------------------------------
input_df = input_df.reindex(columns=features, fill_value=0)

# --------------------------------
# Predict
# --------------------------------
predicted_price = model.predict(input_df)[0]

print("Predicted Vehicle Price:", int(predicted_price))
