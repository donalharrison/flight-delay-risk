from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.flight_delay_risk.app.inference import load_artifacts, predict_one


st.set_page_config(page_title="Flight Delay Risk (BTS-only v1)", layout="centered")

st.title("✈️ Flight Delay Risk")
st.caption("BTS-only v1 • Predicts probability of arrival delay ≥ 15 minutes")

ARTIFACTS_DIR = Path("models/artifacts")

@st.cache_resource
def get_artifacts():
    return load_artifacts(ARTIFACTS_DIR)

artifacts = get_artifacts()

# --- Inputs ---
st.subheader("Flight details")

col1, col2 = st.columns(2)

with col1:
    flight_dt = st.date_input("Flight date", value=date(2024, 6, 15))
    carrier = st.text_input("Carrier (OP_CARRIER)", value="DL", help="Example: DL, AA, UA, WN")
    origin = st.text_input("Origin airport", value="ATL")
    dest = st.text_input("Destination airport", value="JFK")

with col2:
    dep_time = st.time_input("Scheduled departure time (local)", value=pd.to_datetime("18:30").time())
    distance = st.number_input("Distance (miles)", min_value=50, max_value=6000, value=760)

# Derived fields (must match training schema)
flight_date = pd.to_datetime(flight_dt)
day_of_week = int(flight_date.dayofweek) + 1  # pandas: Mon=0 -> BTS: Mon=1
is_weekend = 1 if day_of_week in (6, 7) else 0
week_of_year = int(flight_date.isocalendar().week)
day_of_month = int(flight_date.day)

crs_dep_hour = int(dep_time.hour)
crs_dep_minutes = int(dep_time.hour) * 60 + int(dep_time.minute)

# For now, CRS_ARR fields aren’t user inputs; set NA if they were in training
# (Your current base features include them, so we’ll approximate with NA)
crs_arr_hour = pd.NA
crs_arr_minutes = pd.NA

route = f"{origin.strip().upper()}-{dest.strip().upper()}"

dep_daypart = (
    "night" if crs_dep_hour <= 5 else
    "morning" if crs_dep_hour <= 11 else
    "afternoon" if crs_dep_hour <= 17 else
    "evening" if crs_dep_hour <= 21 else
    "late"
)

# Build feature dict; include superset, inference will select exact feature_cols
feature_payload = {
    "YEAR": int(flight_date.year),
    "MONTH": int(flight_date.month),
    "DAY_OF_WEEK": day_of_week,
    "is_weekend": is_weekend,
    "week_of_year": week_of_year,
    "day_of_month": day_of_month,
    "OP_CARRIER": carrier.strip().upper(),
    "ORIGIN": origin.strip().upper(),
    "DEST": dest.strip().upper(),
    "route": route,
    "dep_daypart": dep_daypart,
    "DISTANCE": float(distance),
    "CRS_DEP_HOUR": crs_dep_hour,
    "CRS_DEP_MINUTES": crs_dep_minutes,
    "CRS_ARR_HOUR": crs_arr_hour,
    "CRS_ARR_MINUTES": crs_arr_minutes,
}

st.divider()

if st.button("Predict delay risk", type="primary"):
    with st.spinner("Scoring…"):
        result = predict_one(artifacts, feature_payload)

    p = result["probability"]
    bucket = result["risk_bucket"]

    st.subheader("Result")
    st.metric("Delay risk (P[arrival delay ≥ 15m])", f"{p:.1%}", bucket)

    st.write("**Why this might be happening (v1 explanations):**")
    for r in result["reasons"]:
        st.write(f"- {r}")

    with st.expander("Debug: features used"):
        st.json(result["features_used"])