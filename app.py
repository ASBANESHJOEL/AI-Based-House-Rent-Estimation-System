import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("house_rent_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

# Load dataset (for locality dropdown)
df = pd.read_csv("Rental_Dataset.csv")

# Remove unwanted spaces
df.columns = df.columns.str.strip()

# Get unique localities
localities = sorted(df["LOCALITY"].astype(str).str.strip().unique())

# Page Settings
st.set_page_config(
    page_title="House Rent Prediction",
    page_icon="🏠",
    layout="centered"
)

# Title
st.title("🏠 AI-Based House Rent Estimation System")
st.write("Enter property details to estimate monthly rent.")

# Inputs
locality = st.selectbox(
    "📍 Select Locality",
    localities
)

bhk_options = {
    "1 RK": 0,
    "1 BHK": 1,
    "2 BHK": 2,
    "3 BHK": 3
}

bhk_display = st.selectbox(
    "🛏️ BHK",
    list(bhk_options.keys())
)

bhk = bhk_options[bhk_display]


sqft = st.number_input(
    "📐 Square Feet",
    min_value=100,
    max_value=10000,
    value=1000,
    step=50
)

condition_options = {
    "Unfurnished": 0,
    "Furnished": 1
}

condition_display = st.selectbox(
    "🏡 Furnishing",
    list(condition_options.keys())
)

condition = condition_options[condition_display]

# Prediction
if st.button("Predict Rent"):

    input_data = pd.DataFrame({
        "LOCALITY": [locality],
        "BHK": [bhk],
        "SQ. FT": [sqft],
        "CONDITION": [condition]
    })

    prediction = model.predict(input_data)

    st.success(
        f"💰 Estimated Monthly Rent: ₹{prediction[0]:,.0f}"
    )
