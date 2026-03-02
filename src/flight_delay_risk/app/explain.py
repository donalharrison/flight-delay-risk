from __future__ import annotations

import json
import os
import httpx
import certifi

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from openai import OpenAI


@dataclass(frozen=True)
class LLMConfig:
    model: str = "gpt-5.2"
    max_output_tokens: int = 350
    temperature: float = 0.2


def build_evidence(
    *,
    flight_inputs: Dict[str, Any],
    prediction: Dict[str, Any],
    shap_out: Dict[str, Any],
    feature_labels: Dict[str, str],
    fs_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a strictly grounded evidence object for the LLM.

    flight_inputs: user-entered + derived flight fields (carrier/origin/dest/date/time, etc.)
    prediction: output from predict_one() (probability, risk_bucket, ...)
    shap_out: output from shap_explain_one() (top_positive, top_negative, base_value, ...)
    fs_meta: optional info about feature store (e.g., max_date used for lookup)
    """
    def _driver_rows(rows: List[Tuple[str, Any, float]], direction: str) -> List[Dict[str, Any]]:
        out = []
        for feat, val, shap_v in rows:
            out.append(
                {
                    "feature": feat,
                    "label": feature_labels.get(feat, feat),
                    "value": _jsonable(val),
                    "direction": direction,  # "up" or "down"
                    "shap": float(shap_v),
                    "abs_shap": float(abs(shap_v)),
                }
            )
        # already sorted in your shap_out; keep stable
        return out

    evidence = {
        "version": "v1",
        "task": "Explain flight arrival-delay risk prediction (>=15 minutes).",
        "prediction": {
            "probability": float(prediction.get("probability", 0.0)),
            "risk_bucket": str(prediction.get("risk_bucket", "unknown")),
        },
        "flight": {
            # keep it small and user-facing
            "flight_date": _jsonable(flight_inputs.get("flight_date")),
            "carrier": flight_inputs.get("OP_CARRIER"),
            "origin": flight_inputs.get("ORIGIN"),
            "dest": flight_inputs.get("DEST"),
            "route": flight_inputs.get("route"),
            "scheduled_departure_hour": flight_inputs.get("CRS_DEP_HOUR"),
            "day_of_week": flight_inputs.get("DAY_OF_WEEK"),
        },
        "lookup": fs_meta or {},
        "drivers": {
            "risk_up": _driver_rows(shap_out.get("top_positive", []), "up"),
            "risk_down": _driver_rows(shap_out.get("top_negative", []), "down"),
        },
        # Helpful for future debugging / eval, not shown to user necessarily
        "shap": {
            "base_value": float(shap_out.get("base_value", 0.0)),
        },
    }
    return evidence


def render_deterministic_bullets(evidence: Dict[str, Any], *, max_bullets: int = 6) -> List[str]:
    """
    Deterministic bullets the LLM should paraphrase.
    These should be strictly derived from evidence (no extra facts).
    """
    p = evidence["prediction"]["probability"]
    bucket = evidence["prediction"]["risk_bucket"]
    flight = evidence["flight"]
    lookup = evidence.get("lookup", {}) or {}

    bullets: List[str] = []
    bullets.append(f"Predicted delay risk is {p:.1%} (bucket: {bucket}).")
    bullets.append(
        f"Flight inputs: {flight.get('carrier')} {flight.get('origin')}→{flight.get('dest')} on {flight.get('flight_date')} "
        f"departing around hour {flight.get('scheduled_departure_hour')}."
    )

    if lookup:
        # Example: show which date we used for aggregates
        if "lookup_date" in lookup and lookup["lookup_date"] is not None:
            bullets.append(f"Historical aggregates were looked up using date: {lookup['lookup_date']}.")

    # Include top SHAP drivers with feature labels + values
    up = evidence["drivers"]["risk_up"][:3]
    down = evidence["drivers"]["risk_down"][:2]

    for d in up:
        bullets.append(f"Risk ↑: {d['label']} = {d['value']} (feature: {d['feature']}).")
    for d in down:
        bullets.append(f"Risk ↓: {d['label']} = {d['value']} (feature: {d['feature']}).")

    return bullets[:max_bullets]


def llm_explain(
    evidence: Dict[str, Any],
    bullets: List[str],
    *,
    config: Optional[LLMConfig] = None,
) -> str:
    """
    Calls OpenAI Responses API to generate a grounded explanation.
    If OPENAI_API_KEY is not set, returns a safe fallback explanation.
    """
    if config is None:
        config = LLMConfig()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Safe offline fallback: no hallucinations, purely derived from bullets
        return _fallback_explanation(bullets)

    # Force TLS verification using certifi's CA bundle
    http_client = httpx.Client(verify=certifi.where(), timeout=30.0)

    client = OpenAI(api_key=api_key, http_client=http_client)

    # System instruction: constrain to evidence + bullets only
    system = (
        "You are a product explanation assistant for a flight delay risk app.\n"
        "You MUST only use the provided evidence JSON and deterministic bullets.\n"
        "Do NOT invent facts (weather, ATC events, airport construction, holidays, etc.).\n"
        "If evidence is insufficient, say so.\n"
        "Write a clear, helpful explanation for a traveler.\n"
        "Structure:\n"
        "1) One-sentence summary of risk.\n"
        "2) 3–5 bullets of key drivers (plain language).\n"
        "3) 2–3 practical suggestions (generic, non-factual) to manage the risk.\n"
        "Keep it concise.\n"
    )

    user_payload = {
        "deterministic_bullets": bullets,
        "evidence": evidence,
    }

    # Use Responses API (recommended for new projects)
    resp = client.responses.create(
        model=config.model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        max_output_tokens=config.max_output_tokens,
        temperature=config.temperature,
    )

    # output_text is provided by the SDK
    return resp.output_text.strip()


# -----------------------
# Helpers
# -----------------------
def _jsonable(x: Any) -> Any:
    # pandas timestamps / numpy scalars -> python scalars / strings
    try:
        import pandas as pd  # local import to avoid hard dependency issues
        if isinstance(x, pd.Timestamp):
            return x.isoformat()
    except Exception:
        pass

    try:
        import numpy as np
        if isinstance(x, (np.generic,)):
            return x.item()
    except Exception:
        pass

    # pd.NA -> None
    if str(x) == "<NA>":
        return None

    return x


def _fallback_explanation(bullets: List[str]) -> str:
    # Deterministic, safe fallback
    lines = []
    lines.append("LLM explanation is unavailable (OPENAI_API_KEY not set).")
    lines.append("")
    lines.append("Here is the grounded evidence we have:")
    for b in bullets:
        lines.append(f"- {b}")
    return "\n".join(lines)