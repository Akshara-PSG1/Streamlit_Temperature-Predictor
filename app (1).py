import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import streamlit as st
import joblib
from tensorflow import keras
from keras import regularizers
from keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

data = fetch_ucirepo(id=925)
X = data.data.features
Y = data.data.targets

print(X)

df = pd.DataFrame(X, columns = ['T_RC1', 'T_RC_Dry1', 'T_RC_Wet1', 'T_RC_Max1', 'T_LC1', 'T_LC_Dry1','Gender', 'Age', 'Ethnicity', 'T_atm', 'Humidity', 'Distance'])

df.head()

df.columns

df_t = pd.DataFrame(Y).drop(columns = 'aveOralM')
df_t.head()

df.describe()

df.info()

df.isnull().sum().sum()

df = df.fillna(0)

df.isnull().sum().sum()

df.nunique()

df['Age'].unique()

df['Age'].unique()
df['Ethnicity'].unique()

def to_float(age):
  if '-'in age:
    l, u = map(int, age.split('-'))
    return (l+u)/2
  elif age.startswith('>'):
    lower = int(age[1:])
    return lower + 8
  else:
    return None
df['Age'] = df['Age'].apply(to_float)

g_map = {'Female': 0, 'Male': 1}
df['Gender'] = df['Gender'].map(g_map)

e_map = {'White': 0, 'Black or African-American': 1, 'Asian': 2, 'Multiracial': 3, 'Hispanic/Latino': 4, 'American Indian or Alaskan Native': 5}
df['Ethnicity'] = df['Ethnicity'].map(e_map)


plt.figure(figsize =(10,5))
corr_matrix = df.corr(method = 'pearson')
sns.heatmap(corr_matrix, annot = True, cmap = 'Blues')

x1 = df['T_RC1']
x2 = df['T_RC_Dry1']
Y = df_t['aveOralF']

plt.figure(figsize = (3,2))
plt.scatter(x1, Y, alpha = 0.5)
plt.scatter(x2, Y, alpha = 0.5)


x3 = df['T_RC_Max1']
plt.figure(figsize = (3,2))
plt.scatter(x1, Y, alpha = 0.5)
plt.scatter(x3, Y, alpha = 0.5)


x_train, x_test, y_train, y_test = train_test_split(df, df_t, test_size = 0.2)
x_train.shape, x_test.shape , y_train.shape, y_test.shape

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = Sequential()
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(x_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(x_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.3, callbacks=[early_stopping])

loss, mae = model.evaluate(x_test_scaled, y_test)
print(f"Loss: {loss}\nMean absolute error: {mae}")

predictions = model.predict(x_test_scaled)

print(len(predictions))

print(len(y_test))

plt.figure(figsize=(4,3))
predictions = predictions.flatten()
y_test = y_test.values.flatten()
plt.scatter(y_test, predictions, alpha=0.5, label='Predictions', color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)


model.save('Temp_predict.h5')


joblib.dump(scaler, 'scaler.pkl')

import joblib
scaler = joblib.load('scaler.pkl')


import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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
