import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.header("Hyderabad Home Forecast")
st.caption('Estimate the price of a home based on Locality and Sqft (2021 data)')
st.image('bg.png', use_container_width=True)

model = pickle.load(open('forest_model.pkl', 'rb'))

areas = [
    "Alkapur Township, Manikonda, Outer Ring Road",
    "Aminpur",
    "Attapur",
    "Bachupally",
    "Bandlaguda Jagir",
    "Beeramguda, Ramachandra Puram, NH 9",
    "Boduppal, NH 22",
    "Chandanagar, NH 9",
    "Dammaiguda",
    "Gachibowli",
    "Gajularamaram",
    "Hafeezpet, NH 9",
    "Kokapet, Outer Ring Road",
    "Kollur, Outer Ring Road",
    "Kompally",
    "Kondapur",
    "Kukatpally, NH 9",
    "LB Nagar, NH 9",
    "Madhapur",
    "Madinaguda, Hafeezpet, NH 9",
    "Mallampet, Outer Ring Road",
    "Manikonda, Outer Ring Road",
    "Miyapur, NH 9",
    "Nagaram",
    "Nagole",
    "Nallagandla, Serilingampally",
    "Narsingi, Outer Ring Road",
    "Nizampet",
    "Patancheru",
    "Pragathi Nagar, Kukatpally",
    "Puppalaguda",
    "Rajendra Nagar, Outer Ring Road",
    "Shamirpet",
    "Tellapur, Outer Ring Road",
    "Toli Chowki",
    "Uppal, NH 22",
]

st.caption('This is a Random Forest based ML model having r-squared score of ~0.70.')

area = st.selectbox("Select Locality:", areas)
area_index = areas.index(area)

sqft = st.slider('Select SqFt Area:', min_value=200, max_value=3500, step=50)

if st.button("Submit", type='primary'):
    if sqft > 0:
        feature_names = ['Locality','Sqft']
        input_arr = pd.DataFrame([[ area_index, sqft]], columns=feature_names)

        prediction = model.predict(input_arr)

        output = round(prediction[0], 2)
        
        st.success(f'The predicted price of the house is {output} Lakhs')
        st.balloons()
        sqft=0
        
    else:
        st.error("Please select a valid square footage.")


