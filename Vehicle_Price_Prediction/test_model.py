import joblib
import pandas as pd

model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

sample_input = {
    "km_driven": 45000,
    "mileage": 18.5,
    "engine": 1197,
    "max_power": 82,
    "seats": 5,
    "fuel": 2,
    "seller_type": 1,
    "transmission": 1,
    "owner": 0,
    "vehicle_age": 6
}

df = pd.DataFrame([sample_input])
df = df.reindex(columns=features, fill_value=0)

prediction = model.predict(df)[0]
print("Predicted Vehicle Price:", int(prediction))
