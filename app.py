import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sklearn

# Define the name of your Streamlit app
st.title("My 1st Deployment App")

# Load model
loaded_model = joblib.load('8Features-2ML-NaiveBayes.pkl')

# Define function to make predictions and display results
def predict_late_delivery_risk(days_for_shipment_scheduled, category_id, customer_zipcode, longitude, product_card_id, product_category_id, shipping_mode, order_hour):
    input_data = pd.DataFrame({'days_for_shipment_scheduled': [days_for_shipment_scheduled],
                               'category_id': [category_id],
                               'customer_zipcode': [customer_zipcode],
                               'longitude': [longitude],
                               'product_card_id': [product_card_id],
                               'product_category_id': [product_category_id],
                               'shipping_mode': [shipping_mode],
                               'order_hour': [order_hour]})
    prediction = loaded_model.predict(input_data)[0]
    probability = loaded_model.predict_proba(input_data)[:, 1][0]
    st.write(f"Prediction: {prediction}")
    st.write(f"Probability: {probability}")

# Create Streamlit app
st.title("Late Delivery Risk Prediction")
days_for_shipment_scheduled = st.slider("Days for Shipment (Scheduled)", min_value=1, max_value=6, value=1)
category_id = st.slider("Category ID", min_value=1, max_value=80, value=1)
customer_zipcode = st.slider("Customer Zipcode", min_value=603, max_value=99205, value=1)
longitude = st.slider("Longitude", min_value=-158.0259857, max_value=115.2630768, value=0.0000001 , format="%.7f")
product_card_id = st.slider("Product Card ID", min_value=19, max_value=1363, value=1)
product_category_id = st.slider("Product Category ID", min_value=1, max_value=76, value=1)
shipping_mode_options = ['Standard Class', 'Second Class', 'First Class', 'Same Day']
shipping_mode = st.selectbox("Shipping Mode", shipping_mode_options, index=0)
order_hour = st.slider("Order Hour", min_value=1, max_value=11, value=1)

# Map the selected shipping mode to a numerical value
shipping_mode_mapping = {'Standard Class': 1, 'Second Class': 2, 'First Class': 3, 'Same Day': 4}
shipping_mode = shipping_mode_mapping[shipping_mode]

if st.button("Predict"):
    predict_late_delivery_risk(days_for_shipment_scheduled, category_id, customer_zipcode, longitude, product_card_id, product_category_id, shipping_mode, order_hour)