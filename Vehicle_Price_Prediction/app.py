import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Vehicle Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# -------------------------------
# Custom CSS (Inspired UI)
# -------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #2b2b2b;
    color: white;
}

/* Hero section */
.hero {
    background-image: linear-gradient(
        rgba(0,0,0,0.6),
        rgba(0,0,0,0.6)
    ),
    url("https://images.unsplash.com/photo-1503376780353-7e6692767b70");
    background-size: cover;
    background-position: center;
    padding: 80px 20px;
    border-radius: 12px;
    margin-bottom: 30px;
}

/* Titles */
.hero h1 {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: white;
}

.hero p {
    text-align: center;
    font-size: 20px;
    color: #f5c542;
}

/* Section headings */
.section-title {
    text-align: center;
    color: #f5c542;
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 20px;
}

/* Card style */
.card p {
    font-size: 16px;
    line-height: 1.8;
    margin-bottom: 8px;
}

.card b {
    color: #f5c542;
}


/* Button */
.stButton > button {
    background-color: #f5c542;
    color: black;
    font-size: 18px;
    font-weight: 700;
    border-radius: 25px;
    padding: 10px 30px;
    border: none;
    width: 100%;
}

.stButton > button:hover {
    background-color: #ffd966;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1c1c1c;
}

/* Footer */
.footer {
    text-align: center;
    color: #bbbbbb;
    margin-top: 40px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load model & encoders
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "encoders.pkl"))

# -------------------------------
# Hero Section
# -------------------------------
st.markdown("""
<div class="hero">
    <h1>Vehicle Price Prediction</h1>
    <p>Predict your old Vehicle price according to your Vehicle features</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("🚗 Vehicle Details")

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

# -------------------------------
# Build input dict
# -------------------------------
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

# Encode categoricals
for col, encoder in encoders.items():
    if col in input_dict:
        val = input_dict[col]
        if val in encoder.classes_:
            input_dict[col] = encoder.transform([val])[0]
        else:
            input_dict[col] = encoder.transform([encoder.classes_[0]])[0]

df = pd.DataFrame([input_dict])
df = df.reindex(columns=features, fill_value=0)

# -------------------------------
# Main Layout
# -------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    if st.button("🔮 Predict Vehicle Price"):
        price = model.predict(df)[0]
        st.success(f"💰 Estimated Price: ₹ {int(price):,}")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown(
        """
        <h2 style="color:#f5c542; margin-bottom:20px;">📌 Project Information</h2>

        <p><b>Project Name:</b> Vehicle Price Prediction</p>
        <p><b>Algorithm:</b> Random Forest Regressor</p>
        <p><b>ML Type:</b> Supervised Learning (Regression)</p>
        <p><b>Deployment:</b> Streamlit Web Application</p>
        <p><b>Use Case:</b> Used vehicle resale value estimation</p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<div class="footer">
© 2026 Vehicle Price Prediction | Built with Python & Machine Learning
</div>
""", unsafe_allow_html=True)
