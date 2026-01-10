# ================================
# Vehicle Price Prediction App
# ================================

import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------
# Load model & features
# --------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(
    page_title="Vehicle Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# --------------------------------
# Header
# --------------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🚗 Vehicle Price Prediction System</h1>
    <p style='text-align: center; font-size:18px;'>
    Machine Learning based resale value estimation
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# --------------------------------
# Sidebar Inputs
# --------------------------------
st.sidebar.header("🔧 Vehicle Details")

km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, step=1000)
mileage = st.sidebar.number_input("Mileage (km/l)", min_value=0.0, step=0.1)
engine = st.sidebar.number_input("Engine Capacity (CC)", min_value=0)
max_power = st.sidebar.number_input("Max Power (bhp)", min_value=0.0)
seats = st.sidebar.selectbox("Seats", [2, 4, 5, 6, 7, 8])

fuel = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.sidebar.selectbox("Seller Type", ["Individual", "Dealer"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.sidebar.selectbox("Owner Type", ["First", "Second", "Third+"])

vehicle_age = st.sidebar.number_input("Vehicle Age (Years)", min_value=0, step=1)

# --------------------------------
# Encode categorical values
# --------------------------------
fuel_map = {"Petrol": 2, "Diesel": 1, "CNG": 0}
seller_map = {"Individual": 1, "Dealer": 0}
transmission_map = {"Manual": 1, "Automatic": 0}
owner_map = {"First": 0, "Second": 1, "Third+": 2}

fuel = fuel_map[fuel]
seller_type = seller_map[seller_type]
transmission = transmission_map[transmission]
owner = owner_map[owner]

# --------------------------------
# Main Layout
# --------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Prediction Result")

    if st.button("🔮 Predict Vehicle Price"):

        input_dict = {
            "km_driven": km_driven,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "vehicle_age": vehicle_age
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=features, fill_value=0)

        prediction = model.predict(input_df)[0]

        st.success(f"💰 Estimated Vehicle Price: ₹ {int(prediction):,}")

with col2:
    st.subheader("ℹ️ Project Info")
    st.info(
        """
        **Algorithm:** Random Forest Regressor  
        **Features Used:** Vehicle usage, engine specs, ownership & fuel type  
        **ML Type:** Supervised Regression  
        **Deployment:** Streamlit  
        """
    )

# --------------------------------
# Footer
# --------------------------------
st.divider()
st.markdown(
    "<p style='text-align:center;'>Built with  using Python, Scikit-learn & Streamlit</p> <p style='text-align:center;'>© 2026 Vehicle Price Prediction</p>",
    unsafe_allow_html=True
)
