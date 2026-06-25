import streamlit as st
import numpy as np
import pandas as pd
import pickle

# --------------------------------
# LOAD MODEL + SCALER + COLUMNS
# --------------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))   # list of 7 features

st.title("💰 Insurance Charges Prediction App")
st.write("Enter the details below to estimate your insurance cost.")

# --------------------------------
# USER INPUTS
# --------------------------------

age = st.number_input("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
children = st.number_input("Number of Children", 0, 10, 0)
smoker = st.selectbox("Smoker?", ["no", "yes"])
region = st.selectbox("Region", ["southeast", "not southeast"])

# --------------------------------
# FEATURE ENGINEERING
# --------------------------------

isfemale = 1 if sex == "female" else 0
is_smoker = 1 if smoker == "yes" else 0
region_southeast = 1 if region == "southeast" else 0
bmi_category_obese = 1 if bmi >= 30 else 0

# Create a DataFrame with the EXACT 7 columns expected by your scaler
input_df = pd.DataFrame([{
    'age': age,
    'isfemale': isfemale,
    'bmi': bmi,
    'children': children,
    'is_smoker': is_smoker,
    'region_southeast': region_southeast,
    'bmi_category_obese': bmi_category_obese
}])

# --------------------------------
# SCALE THE FULL 7-FEATURE INPUT
# --------------------------------
scaled_input = scaler.transform(input_df)

# --------------------------------
# PREDICT
# --------------------------------
if st.button("Predict Insurance Charges"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"Estimated Insurance Charge: 💵 ${prediction:,.2f}")
