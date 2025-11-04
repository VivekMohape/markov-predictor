# utils/llm_utils.py
import streamlit as st
import requests
import json
import re
import traceback
from utils import markov_model

# -------------------------------------------------------------
# 🔑  Configuration
# -------------------------------------------------------------
GROQ_URL = "https://api.groq.com/openai/v1/responses"
MODEL = "openai/gpt-oss-20b"

# -------------------------------------------------------------
# ⚙️  Helper Functions
# -------------------------------------------------------------
def _safe_call(prompt, timeout=60):
    """Send a request to the Groq API and handle HTTP errors gracefully."""
    key = st.secrets.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("⚠️ Missing GROQ_API_KEY in Streamlit secrets.")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": MODEL, "input": prompt}
    r = requests.post(GROQ_URL, headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _extract_text(resp_json):
    """Extract main text content from Groq response object."""
    if not resp_json:
        return ""
    if "output_text" in resp_json and resp_json["output_text"]:
        return resp_json["output_text"].strip()
    if "output" in resp_json and resp_json["output"]:
        pieces = []
        for item in resp_json["output"]:
            for c in item.get("content", []):
                if "text" in c:
                    pieces.append(c["text"])
        return "\n".join(pieces).strip()
    return str(resp_json)

def _try_parse_json(raw_text):
    """
    Robust JSON parser that extracts the first valid JSON block from an LLM response.
    Works even if the response has text before/after the JSON.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None

    # Extract {...} block
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Extract [...] block
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None

# -------------------------------------------------------------
# 🧠  Core Predictor
# -------------------------------------------------------------
def universal_predictor(user_text):
    """
    Universal LLM + Markov hybrid predictor.
    Handles arbitrary text domains (career, finance, health, learning, etc.).
    """
    result = {"input": user_text}
    try:
        # --- 1️⃣ Ask LLM to identify domain, states, and current state
        prompt = (
            "Analyze this input and respond strictly in JSON format with keys:\n"
            "'domain': short string (e.g., 'career', 'finance', 'health', 'learning', 'other'),\n"
            "'states': ordered list (4–8 progressive states relevant to the context),\n"
            "'current_state': one of the states that best matches current situation.\n\n"
            "Return ONLY JSON. The first character of your response must be '{'.\n\n"
            f"Input:\n{user_text}"
        )
        j = _safe_call(prompt)
        raw = _extract_text(j)
        parsed = _try_parse_json(raw)

        # If parsing failed, attempt a repair prompt
        if not parsed:
            repair_prompt = (
                f"The following output is not valid JSON:\n{raw}\n\n"
                "Reformat it strictly as valid JSON. Do not add any explanation."
            )
            j2 = _safe_call(repair_prompt)
            raw2 = _extract_text(j2)
            parsed = _try_parse_json(raw2)

        if not parsed:
            raise ValueError(f"Groq returned non-JSON or empty output:\n{raw}")

        states = parsed.get("states", [])
        current = parsed.get("current_state")
        if not states or not current:
            raise ValueError(f"LLM JSON missing 'states' or 'current_state':\n{parsed}")

        # --- 2️⃣ Markov step: predict next probable state
        matrix = markov_model.build_uniform_matrix(states)
        pred = markov_model.predict_next_state(states, matrix, current, deterministic=True)
        if "error" in pred:
            raise RuntimeError(pred["error"])
        next_state = pred["next_state"]

        # --- 3️⃣ Ask LLM for contextual explanation
        explain_prompt = (
            f"Domain: {parsed.get('domain')}\n"
            f"Current state: {current}\nNext state: {next_state}\n\n"
            f"User text:\n{user_text}\n\n"
            "Write one clear paragraph (80–120 words) explaining naturally "
            "why moving from the current state to the next state makes sense. "
            "Avoid mentioning tasks or instructions; sound human and insightful."
        )
        j3 = _safe_call(explain_prompt)
        explanation = _extract_text(j3)

        # --- 4️⃣ Return structured result
        result.update({
            "domain": parsed.get("domain"),
            "states": states,
            "current_state": current,
            "predicted_next_state": next_state,
            "explanation": explanation,
        })

    except Exception as e:
        result["error"] = f"Universal predictor failed: {e}\n{traceback.format_exc()}"

    return result
