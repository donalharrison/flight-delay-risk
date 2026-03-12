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
st.set_page_config(page_title="Flight Delay Risk (Reduced Scope)", layout="centered")
st.title("✈️ Flight Delay Risk")
st.caption("Reduced-scope BTS model • Predicts probability of arrival delay ≥ 15 minutes")
st.info(
    "Current demo scope is optimized for major U.S. airports "
    "(ATL, ORD, DFW, DEN, LAX, JFK, CLT, LAS, PHX, MCO) "
    "and major carriers (DL, AA, UA, WN)."
)

# ----------------------------
# Paths
# ----------------------------
ARTIFACTS_DIR = Path("models/artifacts_reduced")
FEATURE_STORE_INDEX_PATH = Path("data/feature_store/with_aggs_index_reduced.joblib")

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

SUPPORTED_AIRPORTS = ["ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "CLT", "LAS", "PHX", "MCO"]
SUPPORTED_CARRIERS = ["DL", "AA", "UA", "WN"]


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
            "Run: python -m src.flight_delay_risk.dataset.build_feature_store_index "
            "--store data/feature_store/with_aggs_store_reduced.parquet "
            "--out data/feature_store/with_aggs_index_reduced.joblib"
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

    origin = str(features.get("ORIGIN", "__MISSING__")).strip()
    dest = str(features.get("DEST", "__MISSING__")).strip()
    carrier = str(features.get("OP_CARRIER", "__MISSING__")).strip()
    route = str(features.get("route", "__MISSING__")).strip()
    dep_hour = int(features.get("CRS_DEP_HOUR", -1))

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

    overlay("by_origin", (lookup_date, origin))
    overlay("by_dest", (lookup_date, dest))
    overlay("by_carrier", (lookup_date, carrier))
    overlay("by_route", (lookup_date, route))
    overlay("by_carrier_origin", (lookup_date, carrier, origin))
    overlay("by_origin_dep_hour", (lookup_date, origin, dep_hour))

    return features


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Explanation settings")
use_llm = st.sidebar.toggle("Use LLM explanation (OpenAI API)", value=True)
llm_model = st.sidebar.text_input("LLM model", value="gpt-5.2")

st.sidebar.markdown("### Model snapshot")
st.sidebar.write(f"Feature store snapshot date: {pd.Timestamp(fs_index['max_date']).date()}")
st.sidebar.write(f"Aggregate features: {len(fs_index['agg_cols'])}")

# ----------------------------
# Streamlit UI (FORM)
# ----------------------------
with st.form("flight_form"):
    st.subheader("Flight details")

    col1, col2 = st.columns(2)

    with col1:
        flight_dt = st.date_input("Flight date", value=date(2025, 6, 15))
        carrier = st.selectbox("Carrier (OP_CARRIER)", options=SUPPORTED_CARRIERS, index=0)
        origin = st.selectbox("Origin airport", options=SUPPORTED_AIRPORTS, index=0)
        dest = st.selectbox("Destination airport", options=SUPPORTED_AIRPORTS, index=5)

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
    day_of_week = int(flight_date.dayofweek) + 1  # pandas Mon=0 -> BTS Mon=1
    is_weekend = 1 if day_of_week in (6, 7) else 0
    week_of_year = int(flight_date.isocalendar().week)
    day_of_month = int(flight_date.day)

    crs_dep_hour = int(dep_time_.hour)
    crs_dep_minutes = int(dep_time_.hour) * 60 + int(dep_time_.minute)

    crs_arr_hour = pd.NA
    crs_arr_minutes = pd.NA

    origin = origin_.strip().upper()
    dest = dest_.strip().upper()
    carrier = carrier_.strip().upper()
    route = f"{origin}-{dest}"

    dep_daypart = (
        "night" if crs_dep_hour <= 5 else
        "morning" if crs_dep_hour <= 11 else
        "afternoon" if crs_dep_hour <= 17 else
        "evening" if crs_dep_hour <= 21 else
        "late"
    )

    feature_payload = {
        "flight_date": flight_date,
        "YEAR": int(flight_date.year),
        "MONTH": int(flight_date.month),
        "DAY_OF_WEEK": day_of_week,
        "is_weekend": is_weekend,
        "week_of_year": week_of_year,
        "day_of_month": day_of_month,
        "OP_CARRIER": carrier,
        "ORIGIN": origin,
        "DEST": dest,
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
    if origin == dest:
        st.error("Origin and destination cannot be the same.")
    else:
        _, base_payload = compute_payload(flight_dt, carrier, origin, dest, dep_time, distance)

        with st.spinner("Looking up historical aggregates…"):
            scoring_payload = apply_aggregate_features_indexed(base_payload)

        with st.spinner("Scoring & explaining…"):
            result = predict_one(artifacts, scoring_payload)

            x_row = pd.DataFrame([{c: scoring_payload.get(c, pd.NA) for c in artifacts.feature_cols}])
            shap_out = shap_explain_one(artifacts, x_row, top_k=6)

        p = result["probability"]
        bucket = result["risk_bucket"]

        fs_meta = {"lookup_date": str(fs_index["max_date"])}

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

        st.subheader("Result")
        st.metric("Delay risk (P[arrival delay ≥ 15m])", f"{p:.1%}", bucket)

        st.subheader("Top drivers (SHAP)")
        st.caption("Positive pushes risk up; negative pushes risk down. Ranked by contribution magnitude.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Risk ↑**")
            for feat, val, s in shap_out["top_positive"]:
                label = FEATURE_LABELS.get(feat, feat)
                st.write(f"- **{label}** (`{feat}`) = **{val}** (contribution: {s:+.3f})")

        with col_b:
            st.markdown("**Risk ↓**")
            for feat, val, s in shap_out["top_negative"]:
                label = FEATURE_LABELS.get(feat, feat)
                st.write(f"- **{label}** (`{feat}`) = **{val}** (contribution: {s:+.3f})")

        with st.expander("All top SHAP contributions"):
            df_shap = pd.DataFrame(shap_out["all"], columns=["feature", "value", "shap"])
            df_shap["label"] = df_shap["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
            df_shap = df_shap[["label", "feature", "value", "shap"]]
            df_shap["abs_shap"] = df_shap["shap"].abs()
            st.dataframe(df_shap, use_container_width=True)

        st.subheader("Explanation")

        st.markdown("**Grounded facts (deterministic):**")
        for b in bullets:
            st.write(f"- {b}")

        st.markdown("**Narrative (LLM):**")
        if isinstance(explanation, str) and explanation.startswith("LLM explanation is unavailable"):
            st.warning(explanation)
        else:
            st.write(explanation)

        with st.expander("Debug: evidence JSON sent to LLM"):
            st.json(evidence)

        with st.expander("Debug: features used"):
            st.json({c: scoring_payload.get(c, None) for c in artifacts.feature_cols})