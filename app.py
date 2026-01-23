import streamlit as st
import pandas as pd
import joblib

# Loading Model
model = joblib.load("clv_random_forest.pkl")

st.set_page_config(page_title ="CLV Predictor", layout = "centered")

st.title("Customer Lifetime Value Predictor")
st.write("Enter customer details to predict CLV")

# Input Fields
recency = st.number_input("Recency (Days since last purchase), min_value = 0")
frequency = st.number_input("Total Orders", min_value = 1)
monetary = st.number_input("Total Monetary Value", min_value=0.0)
avg_order_value = st.number_input("Average Order Value", min_value = 0.0)
tenure = st.number_input("Tenure", min_value= 1)

# Input DataFrame
input_df = pd.DataFrame([{
    'frequency': frequency, 
    'monetary': monetary, 
    'recency': recency, 
    'tenure': tenure, 
    'avg_order_value': avg_order_value
}])


# Prediction 
if st.button("Predict CLV"):
    prediction = model.predict(input_df)[0]

    if prediction < 500:
        segment = "Low CLV"
    
    elif prediction < 1500:
        segment = "Medium CLV"
    
    else:
        segment = "High CLV"

    st.success(f"Predicted CLV: {prediction: ,.2f}")
    st.info(f"Customer Segment: {segment}")

    














