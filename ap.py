import streamlit as st
import requests
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('crop_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🌾 Smart Crop Predictor with Live Weather 🌦️")

# Weather API setup
API_KEY = "5b61339634a7bf8a81f224fcb0473820"  # <-- Replace with your actual key
city = st.text_input("Umuahia:", value="Umuahia")

if st.button("Use Live Weather"):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url).json()
        
        temperature = response['main']['temp']
        humidity = response['main']['humidity']
        rainfall = response.get('rain', {}).get('1h', 0) or 0

        st.success(f"🌤️ Weather in {city}: {temperature}°C, {humidity}%, {rainfall}mm")
    except Exception as e:
        st.error("❌ Could not fetch weather. Please check your city or API key.")
        temperature, humidity, rainfall = 25, 60, 100  # fallback

else:
    # Manual input
    temperature = st.slider("Temperature (°C)", 0, 50, 25)
    humidity = st.slider("Humidity (%)", 0, 100, 60)
    rainfall = st.slider("Rainfall (mm)", 0, 300, 100)

# Always require NPK + pH
N = st.slider("Nitrogen (N)", 0, 140, 80)
P = st.slider("Phosphorus (P)", 0, 140, 40)
K = st.slider("Potassium (K)", 0, 200, 40)
ph = st.slider("Soil pH", 0.0, 14.0, 6.5)

if st.button("🌱 Predict Best Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    st.success(f"✅ Recommended Crop: **{prediction.upper()}**")

