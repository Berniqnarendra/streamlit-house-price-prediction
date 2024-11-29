import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load California Housing dataset
data = fetch_california_housing(as_frame=True)
X = data['data']
y = data['target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit app
st.title("Prediksi Harga Rumah di California")
st.write("Masukkan detail rumah untuk mendapatkan prediksi harga.")

# Input fitur rumah
MedInc = st.slider("Pendapatan Median (MedInc)", float(X.MedInc.min()), float(X.MedInc.max()), float(X.MedInc.mean()))
HouseAge = st.slider("Umur Rumah (HouseAge)", float(X.HouseAge.min()), float(X.HouseAge.max()), float(X.HouseAge.mean()))
AveRooms = st.slider("Jumlah Rata-rata Kamar (AveRooms)", float(X.AveRooms.min()), float(X.AveRooms.max()), float(X.AveRooms.mean()))
AveBedrms = st.slider("Jumlah Rata-rata Kamar Tidur (AveBedrms)", float(X.AveBedrms.min()), float(X.AveBedrms.max()), float(X.AveBedrms.mean()))
Population = st.slider("Populasi (Population)", float(X.Population.min()), float(X.Population.max()), float(X.Population.mean()))
AveOccup = st.slider("Rata-rata Penghuni per Rumah (AveOccup)", float(X.AveOccup.min()), float(X.AveOccup.max()), float(X.AveOccup.mean()))
Latitude = st.slider("Latitude", float(X.Latitude.min()), float(X.Latitude.max()), float(X.Latitude.mean()))
Longitude = st.slider("Longitude", float(X.Longitude.min()), float(X.Longitude.max()), float(X.Longitude.mean()))

# Prediksi harga
input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
predicted_price = model.predict(input_data)[0]

st.subheader("Hasil Prediksi")
st.write(f"Harga rumah diperkirakan: ${predicted_price * 100000:.2f}")
