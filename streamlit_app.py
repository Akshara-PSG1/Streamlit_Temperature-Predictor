import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import subprocess
import sys
import tensorflow as tf
try:
    import tensorflow as tf
    st.success("TensorFlow is installed and imported successfully.")
except ModuleNotFoundError:
    st.warning("TensorFlow is not installed. Installing TensorFlow...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    import tensorflow as tf
    st.success("TensorFlow was successfully installed.")

from tensorflow.keras.models import load_model

model = tf.keras.models.load_model('Temp_predict.h5')


def get_user_input():
    features = {}
    st.write('Temperature readings from Thermal Image in Degrees Celsius')
    features['T_RC1'] = st.number_input("Enter T_RC1:", value=0.0, help = 'Average temperature of the highest four pixels in a square of 24x24 pixels around the right canthus')
    features['T_RC_Dry1'] = st.number_input("Enter T_RC_Dry1:", value=0.0, help = "Average temperature of the highest four pixels in the right canthus dry area, a rectangle of 16x24 pixels.")
    features['T_RC_Wet1'] = st.number_input("Enter T_RC_Wet1:", value=0.0, help = "Average temperature of the highest four pixels in the right canthus wet area, a rectangle of 8x24 pixels.")
    features['T_RC_Max1'] = st.number_input("Enter T_RC_Max1:", value=0.0, help = "Max value of a square of 24x24 pixels around the right canthus")
    features['T_LC1'] = st.number_input("Enter T_LC1:", value=0.0, help = "Average temperature of the highest four pixels in a square of 24x24 pixels around the left canthus")
    features['T_LC_Dry1'] = st.number_input("Enter T_LC_Dry1:", value=0.0, help = "Average temperature of the highest four pixels in the left canthus dry area, a rectangle of 16x24 pixels.")
    
    gender = st.selectbox("Select Gender", ("Female", "Male"))
    features['Gender'] = 0 if gender == "Female" else 1

    features['Age'] = st.number_input("Enter Age:", value=0.0)

    ethnicity = st.selectbox("Select Ethnicity", ["White", "Black or African-American", "Asian", "Multiracial", "Hispanic/Latino", "American Indian or Alaskan Native"])
    e_map = {'White': 0, 'Black or African-American': 1, 'Asian': 2, 'Multiracial': 3, 'Hispanic/Latino': 4, 'American Indian or Alaskan Native': 5}
    features['Ethnicity'] = e_map[ethnicity]

    features['T_atm'] = st.number_input("Enter T_atm:", value=0.0)
    features['Humidity'] = st.number_input("Enter Humidity:", value=0.0)
    features['Distance'] = st.number_input("Enter Distance:", value=0.0, help = "Distance between the subjects and the Infrared Thermometers")
    
    return pd.DataFrame([features])

import joblib
scaler = joblib.load('scaler.pkl')

st.title("Infrared Thermography\nOral Temperature Prediction")
st.image('/content/infrared.png')
user_input = get_user_input()

if st.button("Predict Oral Temperature"):
    user_input_scaled = scaler.transform(user_input)
    pred = model.predict(user_input_scaled)
    st.write(f"Predicted Oral Temperature in fast mode is: {pred[0][0]} Degree Celsius")
