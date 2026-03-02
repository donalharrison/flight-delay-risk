from __future__ import annotations

from datetime import date
from pathlib import Path
import re
from typing import Tuple, Any

import pandas as pd
import streamlit as st
import joblib

from src.flight_delay_risk.app.inference import (
    load_artifacts,
    predict_one,
    shap_explain_one,
)

from src.flight_delay_risk.app.explain import (
    build_evidence,
    render_deterministic_bullets,
    llm_explain,
    LLMConfig,
)

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Flight Delay Risk (BTS-only v1)", layout="centered")
st.title("✈️ Flight Delay Risk")
st.caption("BTS-only v1 • Predicts probability of arrival delay ≥ 15 minutes")

st.sidebar.header("Explanation settings")
use_llm = st.sidebar.toggle("Use LLM explanation (OpenAI API)", value=True)
llm_model = st.sidebar.text_input("LLM model", value="gpt-5.2")

# ----------------------------
# Paths
# ----------------------------
ARTIFACTS_DIR = Path("models/artifacts")
FEATURE_STORE_INDEX_PATH = Path("data/feature_store/with_aggs_index.joblib")

# ----------------------------
# Caching: artifacts
# ----------------------------
@st.cache_resource
def get_artifacts():
    return load_artifacts(ARTIFACTS_DIR)

artifacts = get_artifacts()

# ----------------------------
# Labels (base + auto aggs)
# ----------------------------
BASE_FEATURE_LABELS = {
    "YEAR": "Year",
    "MONTH": "Month",
    "DAY_OF_WEEK": "Day of week (1=Mon…7=Sun)",
    "is_weekend": "Weekend flag",
    "week_of_year": "Week of year",
    "day_of_month": "Day of month",
    "OP_CARRIER": "Carrier",
    "ORIGIN": "Origin airport",
    "DEST": "Destination airport",
    "route": "Route",
    "dep_daypart": "Time of day (bucket)",
    "DISTANCE": "Distance (miles)",
    "CRS_DEP_HOUR": "Scheduled departure hour",
    "CRS_DEP_MINUTES": "Scheduled departure minutes (from midnight)",
    "CRS_ARR_HOUR": "Scheduled arrival hour",
    "CRS_ARR_MINUTES": "Scheduled arrival minutes (from midnight)",
}

SPEC_LABELS = {
    "origin": "Origin airport",
    "dest": "Destination airport",
    "carrier": "Carrier",
    "route": "Route",
    "carrier_origin": "Carrier × Origin",
    "origin_dep_hour": "Origin × Departure hour",
}

def build_feature_labels(feature_cols: list[str]) -> dict[str, str]:
    labels = dict(BASE_FEATURE_LABELS)
    pat = re.compile(r"^(?P<spec>[a-z_]+)_(?P<kind>delay_rate|freq)_(?P<w>\d+)d$")

    for c in feature_cols:
        if c in labels:
            continue
        m = pat.match(c)
        if not m:
            continue

        spec = m.group("spec")
        kind = m.group("kind")
        w = m.group("w")

        spec_label = SPEC_LABELS.get(spec, spec.replace("_", " ").title())
        if kind == "delay_rate":
            labels[c] = f"{spec_label} delay rate (last {w} days)"
        else:
            labels[c] = f"{spec_label} flights observed (last {w} days)"

    return labels

FEATURE_LABELS = build_feature_labels(artifacts.feature_cols)

# ----------------------------
# Feature store index: joblib load
# ----------------------------
@st.cache_resource
def get_feature_store_index() -> dict[str, Any]:
    if not FEATURE_STORE_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature store index at {FEATURE_STORE_INDEX_PATH}. "
            "Run: python -m src.flight_delay_risk.dataset.build_feature_store_index"
        )
    return joblib.load(FEATURE_STORE_INDEX_PATH)

fs_index = get_feature_store_index()

def apply_aggregate_features_indexed(feature_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Fill aggregate features using prebuilt hash maps loaded from joblib.
    Uses lookup_date = min(requested, max available).
    """
    features = dict(feature_payload)

    req_date = pd.to_datetime(features["flight_date"]).normalize()
    lookup_date = req_date if req_date <= fs_index["max_date"] else fs_index["max_date"]

    ORIGIN = str(features.get("ORIGIN", "__MISSING__")).strip()
    DEST = str(features.get("DEST", "__MISSING__")).strip()
    OP_CARRIER = str(features.get("OP_CARRIER", "__MISSING__")).strip()
    route = str(features.get("route", "__MISSING__")).strip()
    dep_hour = int(features.get("CRS_DEP_HOUR", -1))

    # Defaults
    global_rate = float(fs_index["global_rate"])
    for c in fs_index["agg_cols"]:
        if "_freq_" in c:
            features.setdefault(c, 0.0)
        else:
            features.setdefault(c, global_rate)

    def overlay(map_name: str, key: tuple):
        row = fs_index[map_name].get(key)
        if not row:
            return
        for c, v in row.items():
            features[c] = v

    overlay("by_origin", (lookup_date, ORIGIN))
    overlay("by_dest", (lookup_date, DEST))
    overlay("by_carrier", (lookup_date, OP_CARRIER))
    overlay("by_route", (lookup_date, route))
    overlay("by_carrier_origin", (lookup_date, OP_CARRIER, ORIGIN))
    overlay("by_origin_dep_hour", (lookup_date, ORIGIN, dep_hour))

    return features

# ----------------------------
# Streamlit UI (FORM)
# ----------------------------
with st.form("flight_form"):
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

    submitted = st.form_submit_button("Predict delay risk")

def compute_payload(
    flight_dt_: date,
    carrier_: str,
    origin_: str,
    dest_: str,
    dep_time_,
    distance_: float,
) -> Tuple[pd.Timestamp, dict[str, Any]]:
    flight_date = pd.to_datetime(flight_dt_)
    day_of_week = int(flight_date.dayofweek) + 1  # pandas: Mon=0 -> BTS: Mon=1
    is_weekend = 1 if day_of_week in (6, 7) else 0
    week_of_year = int(flight_date.isocalendar().week)
    day_of_month = int(flight_date.day)

    crs_dep_hour = int(dep_time_.hour)
    crs_dep_minutes = int(dep_time_.hour) * 60 + int(dep_time_.minute)

    # Keep CRS_ARR_* as NA for now unless you add arrival time input
    crs_arr_hour = pd.NA
    crs_arr_minutes = pd.NA

    ORIGIN = origin_.strip().upper()
    DEST = dest_.strip().upper()
    OP_CARRIER = carrier_.strip().upper()
    route = f"{ORIGIN}-{DEST}"

    dep_daypart = (
        "night" if crs_dep_hour <= 5 else
        "morning" if crs_dep_hour <= 11 else
        "afternoon" if crs_dep_hour <= 17 else
        "evening" if crs_dep_hour <= 21 else
        "late"
    )

    feature_payload = {
        "flight_date": flight_date,  # required for agg lookup
        "YEAR": int(flight_date.year),
        "MONTH": int(flight_date.month),
        "DAY_OF_WEEK": day_of_week,
        "is_weekend": is_weekend,
        "week_of_year": week_of_year,
        "day_of_month": day_of_month,
        "OP_CARRIER": OP_CARRIER,
        "ORIGIN": ORIGIN,
        "DEST": DEST,
        "route": route,
        "dep_daypart": dep_daypart,
        "DISTANCE": float(distance_),
        "CRS_DEP_HOUR": crs_dep_hour,
        "CRS_DEP_MINUTES": crs_dep_minutes,
        "CRS_ARR_HOUR": crs_arr_hour,
        "CRS_ARR_MINUTES": crs_arr_minutes,
    }

    return flight_date, feature_payload

if submitted:
    _, base_payload = compute_payload(flight_dt, carrier, origin, dest, dep_time, distance)

    with st.spinner("Looking up historical aggregates…"):
        scoring_payload = apply_aggregate_features_indexed(base_payload)

    with st.spinner("Scoring & explaining…"):
        result = predict_one(artifacts, scoring_payload)

        X_row = pd.DataFrame([{c: scoring_payload.get(c, pd.NA) for c in artifacts.feature_cols}])
        shap_out = shap_explain_one(artifacts, X_row, top_k=6)

        fs_meta = {"lookup_date": str(fs_index["max_date"])}

        # --- LLM explanation layer ---
        evidence = build_evidence(
            flight_inputs=scoring_payload,
            prediction=result,
            shap_out=shap_out,
            feature_labels=FEATURE_LABELS,
            fs_meta=fs_meta,
        )
        bullets = render_deterministic_bullets(evidence)

        if use_llm:
            with st.spinner("Generating explanation…"):
                explanation = llm_explain(
                    evidence,
                    bullets,
                    config=LLMConfig(model=llm_model),
                )
        else:
            explanation = "LLM explanation disabled."

    p = result["probability"]
    bucket = result["risk_bucket"]

    st.subheader("Result")
    st.metric("Delay risk (P[arrival delay ≥ 15m])", f"{p:.1%}", bucket)

    st.subheader("Top drivers (SHAP)")
    st.caption("Positive pushes risk up; negative pushes risk down. Ranked by contribution magnitude.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Risk ↑**")
        for feat, val, s in shap_out["top_positive"]:
            label = FEATURE_LABELS.get(feat, feat)
            st.write(f"- **{label}** (`{feat}`) = **{val}** (contribution: {s:+.3f})")

    with colB:
        st.markdown("**Risk ↓**")
        for feat, val, s in shap_out["top_negative"]:
            label = FEATURE_LABELS.get(feat, feat)
            st.write(f"- **{label}** (`{feat}`) = **{val}** (contribution: {s:+.3f})")

    st.subheader("Explanation")

    st.markdown("**Grounded facts (deterministic):**")
    for b in bullets:
        st.write(f"- {b}")

    st.markdown("**Narrative (LLM):**")
    st.write(explanation)

    with st.expander("Debug: evidence JSON sent to LLM"):
        st.json(evidence)
        
    with st.expander("All top SHAP contributions"):
        df_shap = pd.DataFrame(shap_out["all"], columns=["feature", "value", "shap"])
        df_shap["label"] = df_shap["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
        df_shap = df_shap[["label", "feature", "value", "shap"]]
        df_shap["abs_shap"] = df_shap["shap"].abs()
        st.dataframe(df_shap, use_container_width=True)

    st.subheader("Why this might be happening (v1 narrative)")
    for feat, _, _ in shap_out["top_positive"][:3]:
        st.write(f"- **{FEATURE_LABELS.get(feat, feat)}** is pushing risk upward.")

    with st.expander("Debug: features used"):
        st.json({c: scoring_payload.get(c, None) for c in artifacts.feature_cols})