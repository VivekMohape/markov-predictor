# utils/llm_utils.py
import streamlit as st, requests, json, traceback
from utils import markov_model

GROQ_URL = "https://api.groq.com/openai/v1/responses"
MODEL = "openai/gpt-oss-20b"

def _safe_call(prompt, timeout=60):
    key = st.secrets.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY in Streamlit secrets.")
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": MODEL, "input": prompt}
    r = requests.post(GROQ_URL, headers=h, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _extract_text(j):
    if "output_text" in j and j["output_text"]:
        return j["output_text"].strip()
    if "output" in j and j["output"]:
        txt = []
        for o in j["output"]:
            for c in o.get("content", []):
                if "text" in c:
                    txt.append(c["text"])
        return "\n".join(txt).strip()
    return str(j)

def universal_predictor(user_text):
    """
    Dynamic pipeline:
    1. LLM infers domain + possible states + current state.
    2. Markov predicts next state.
    3. LLM summarizes result.
    """
    result = {"input": user_text}
    try:
        # --- 1. Ask LLM for domain and states
        prompt = (
            "Analyze this text and output JSON with keys:\n"
            "'domain' (string, short category like finance, career, health, learning, etc.),\n"
            "'states' (ordered list of 4–8 progressive states representing this context),\n"
            "'current_state' (string, one item from states that best fits current situation).\n"
            "Return only JSON.\n\n"
            f"Input:\n{user_text}"
        )
        j = _safe_call(prompt)
        raw = _extract_text(j)
        parsed = json.loads(raw)
        states = parsed.get("states", [])
        current = parsed.get("current_state")
        if not states or not current:
            raise ValueError("LLM did not return states/current_state")

        # --- 2. Build uniform Markov if none given
        matrix = markov_model.build_uniform_matrix(states)
        pred = markov_model.predict_next_state(states, matrix, current, deterministic=True)
        if "error" in pred:
            raise RuntimeError(pred["error"])
        next_state = pred["next_state"]

        # --- 3. Ask LLM to interpret prediction
        prompt2 = (
            f"Domain: {parsed.get('domain')}\n"
            f"Current state: {current}\nNext state: {next_state}\n\n"
            f"User text:\n{user_text}\n\n"
            "Write a short paragraph (80–120 words) explaining in plain language "
            "why moving from the current state to the next state is a reasonable progression. "
            "Avoid meta-commentary or instructions."
        )
        j2 = _safe_call(prompt2)
        explanation = _extract_text(j2)

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
