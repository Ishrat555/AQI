import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ---------- Page config ----------
st.set_page_config(
    page_title="Pakistan AQI Forecaster",
    page_icon="🇵🇰",
    layout="centered"
)

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load('rf_model.pkl')
    le = joblib.load('label_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    feat_cols = joblib.load('feature_cols.pkl')
    return model, le, scaler, feat_cols

model, le, scaler, feat_cols = load_artifacts()

# ---------- Header ----------
st.title("🇵🇰 Pakistan AQI Forecaster")
st.caption("Predict AQI category 6–24 hours ahead from pollutant & weather history")

# ---------- City list ----------
CITIES = ["Lahore", "Karachi", "Islamabad", "Faisalabad",
          "Peshawar", "Multan", "Quetta", "Rawalpindi",
          "Sialkot", "Rahim Yar Khan"]

# ---------- Sidebar: current conditions ----------
st.sidebar.header("📍 Current Conditions")

city = st.sidebar.selectbox("City", CITIES)

st.sidebar.subheader("Pollutants")
pm2_5 = st.sidebar.number_input("PM2.5 (µg/m³)", 0.0, 500.0, 60.0, 1.0)
pm10  = st.sidebar.number_input("PM10 (µg/m³)", 0.0, 800.0, 120.0, 1.0)
co    = st.sidebar.number_input("CO (µg/m³)", 0.0, 5000.0, 800.0, 10.0)
no2   = st.sidebar.number_input("NO₂ (µg/m³)", 0.0, 300.0, 30.0, 1.0)
so2   = st.sidebar.number_input("SO₂ (µg/m³)", 0.0, 200.0, 8.0, 1.0)
o3    = st.sidebar.number_input("Ozone (µg/m³)", 0.0, 300.0, 50.0, 1.0)
dust  = st.sidebar.number_input("Dust", 0.0, 500.0, 20.0, 1.0)

st.sidebar.subheader("Weather")
temp  = st.sidebar.number_input("Temperature (°C)", -10.0, 50.0, 18.0, 0.5)
hum   = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 60.0, 1.0)
wind  = st.sidebar.number_input("Wind speed (m/s)", 0.0, 50.0, 2.0, 0.5)
press = st.sidebar.number_input("Pressure (hPa)", 950.0, 1050.0, 1013.0, 0.5)

# ---------- Build feature row ----------
def build_feature_row():
    row = {c: 0 for c in feat_cols}
    current = {
        'pm2_5': pm2_5, 'pm10': pm10, 'carbon_monoxide': co,
        'nitrogen_dioxide': no2, 'sulphur_dioxide': so2, 'ozone': o3,
        'dust': dust, 'temperature': temp, 'humidity': hum,
        'wind_speed': wind, 'pressure': press
    }
    # Lags ko current values se approximate karo (demo ke liye)
    for base, val in current.items():
        for lag in ['lag1', 'lag3', 'lag24']:
            k = f'{base}_{lag}'
            if k in row:
                row[k] = val
    # Rolling means
    if 'pm2_5_roll6' in row: row['pm2_5_roll6'] = pm2_5
    if 'pm10_roll6'  in row: row['pm10_roll6']  = pm10
    # Time
    now = datetime.now()
    if 'hour' in row:      row['hour'] = now.hour
    if 'dayofweek' in row: row['dayofweek'] = now.weekday()
    if 'month' in row:     row['month'] = now.month
    # City one-hot
    city_key = f'city_{city}'
    if city_key in row: row[city_key] = 1
    # Season one-hot (current month se)
    season_map = {12:'Winter',1:'Winter',2:'Winter',
                  3:'Spring',4:'Spring',5:'Spring',
                  6:'Summer',7:'Summer',8:'Summer',
                  9:'Autumn',10:'Autumn',11:'Autumn'}
    season = season_map[now.month]
    season_key = f'season_{season}'
    if season_key in row: row[season_key] = 1
    return row

# ---------- Main panel ----------
st.subheader(f"🔮 Forecast for {city}")

if st.button("Predict AQI 24h ahead", type="primary"):
    with st.spinner("Running forecast..."):
        row = build_feature_row()
        X = pd.DataFrame([row])[feat_cols]
        Xs = scaler.transform(X)
        proba = model.predict_proba(Xs)[0]
        classes = le.classes_
        pred_idx = proba.argmax()
        pred_label = classes[pred_idx]

        # Result card
        color_map = {
            'Good': '#00e400',
            'Moderate': '#ffff00',
            'Unhealthy for Sensitive Groups': '#ff7e00',
            'Unhealthy': '#ff0000',
            'Very Unhealthy': '#8f3f97',
            'Hazardous': '#7e0023'
        }
        color = color_map.get(pred_label, '#cccccc')
        st.markdown(
            f"<div style='background:{color}; padding:20px; border-radius:10px; "
            f"text-align:center; color:white; font-size:24px; font-weight:bold;'>"
            f"{pred_label}</div>",
            unsafe_allow_html=True
        )
        st.metric("Confidence", f"{proba[pred_idx]*100:.1f}%")

        # Probability bar chart
        proba_df = pd.DataFrame({
            'Category': classes,
            'Probability': proba
        }).sort_values('Probability', ascending=True)
        st.bar_chart(proba_df.set_index('Category'))

        st.caption(
            "⚠️ Demo uses current pollutant values as lag proxies. "
            "Real deployment would use live sensor feeds."
        )

# ---------- Footer ----------
st.divider()
st.caption(
    "Built with Streamlit · Model: Random Forest · "
    "Trained on 21,840 hourly records from 10 Pakistani cities"
)