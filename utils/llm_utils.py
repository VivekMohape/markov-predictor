import streamlit as st
import requests
import json
import re
import traceback
from utils import markov_model

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
GROQ_URL = "https://api.groq.com/openai/v1/responses"
MODEL = "openai/gpt-oss-20b"

# -------------------------------------------------------------
# Helper: call Groq safely
# -------------------------------------------------------------
def _safe_call(prompt, timeout=60):
    key = st.secrets.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("⚠️ Missing GROQ_API_KEY in Streamlit secrets.")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": MODEL, "input": prompt}
    r = requests.post(GROQ_URL, headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()

# -------------------------------------------------------------
# Extract clean text from Groq response
# -------------------------------------------------------------
def _extract_text(resp_json):
    """Extracts meaningful paragraph text from various Groq response formats."""
    if not resp_json:
        return ""
    text = ""

    # Handle Groq's multiple formats
    if isinstance(resp_json, dict):
        if "output_text" in resp_json and resp_json["output_text"]:
            text = resp_json["output_text"].strip()
        elif "output" in resp_json:
            segments = []
            for block in resp_json["output"]:
                for c in block.get("content", []):
                    if "text" in c:
                        segments.append(c["text"])
            text = "\n".join(segments).strip()
    elif isinstance(resp_json, str):
        text = resp_json.strip()

    # Clean up LLM reasoning/meta lines
    lines = text.splitlines()
    filtered = []
    for line in lines:
        if re.match(r"^(we need|let'?s|so we|our task|they want|provide only|ensure|check clarity|97 words)", line.strip(), re.I):
            continue
        filtered.append(line)
    text = " ".join(filtered).strip()

    # Keep last coherent paragraph
    paragraphs = re.split(r"\n{2,}", text)
    if len(paragraphs) > 1:
        text = paragraphs[-1].strip()

    return text

# -------------------------------------------------------------
# Try parsing JSON from model output
# -------------------------------------------------------------
def _try_parse_json(raw_text):
    """Extract the last valid JSON block from a possibly messy LLM output."""
    if not raw_text or not isinstance(raw_text, str):
        return None
    candidates = re.findall(r"(\{.*?\}|\[.*?\])", raw_text, re.DOTALL)
    for block in reversed(candidates):
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict) and "states" in parsed and "current_state" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    return None

# -------------------------------------------------------------
# Polishing layer with self-healing
# -------------------------------------------------------------
def polish_output(text):
    """Polish paragraph clarity and tone with Groq, fallback-safe."""
    if not text or len(text.strip()) < 40:
        return text

    try:
        key = st.secrets.get("GROQ_API_KEY")
        if not key:
            return text

        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body = {
            "model": MODEL,
            "input": (
                "Rewrite the following paragraph naturally with smoother grammar and punctuation. "
                "Keep the same meaning. Return ONLY the improved paragraph.\n\n"
                f"{text}"
            ),
        }

        r = requests.post(GROQ_URL, headers=headers, json=body, timeout=30)
        r.raise_for_status()
        data = r.json()

        refined = ""
        if "output_text" in data and data["output_text"]:
            refined = data["output_text"].strip()
        elif "output" in data and len(data["output"]) > 0:
            refined = data["output"][0]["content"][0]["text"].strip()

        # Filter out meta echoes
        if not refined or re.match(r"^(check|ensure|return only|good|ok|fine|provide only|97 words)", refined.lower()):
            refined = text

        refined = refined.split("\n\n")[-1].strip()
        if len(refined.split()) < 40:
            refined = text  # revert if too short

        return refined

    except Exception as e:
        st.warning(f"⚠️ Polish skipped: {e}")
        return text

# -------------------------------------------------------------
# Universal Predictor
# -------------------------------------------------------------
def universal_predictor(user_text):
    """Universal LLM + Markov prediction pipeline with explanation fallback."""
    result = {"input": user_text}
    try:
        # 1️⃣ Domain + States + Current
        prompt = (
            "Analyze this input and respond strictly in JSON format with keys:\n"
            "'domain': (career, finance, learning, health, etc.),\n"
            "'states': ordered list (4–8 logical progression states),\n"
            "'current_state': one of the states that matches current situation.\n\n"
            "Return ONLY JSON (first character must be '{').\n\n"
            f"Input:\n{user_text}"
        )
        j = _safe_call(prompt)
        raw = _extract_text(j)
        parsed = _try_parse_json(raw)

        if not parsed:
            repair_prompt = f"Reformat this to valid JSON only:\n{raw}"
            j2 = _safe_call(repair_prompt)
            raw2 = _extract_text(j2)
            parsed = _try_parse_json(raw2)

        if not parsed:
            raise ValueError(f"Groq returned non-JSON output:\n{raw}")

        states = parsed.get("states", [])
        current = parsed.get("current_state")
        if not states or not current:
            raise ValueError("Missing states or current_state in JSON.")

        # 2️⃣ Predict next state using Markov logic
        matrix = markov_model.build_uniform_matrix(states)
        pred = markov_model.predict_next_state(states, matrix, current, text=user_text, deterministic=True)
        next_state = pred.get("next_state", states[-1])

        # 3️⃣ Generate explanation
        explain_prompt = (
            f"Domain: {parsed.get('domain')}\n"
            f"Current state: {current}\nNext state: {next_state}\n\n"
            f"User text:\n{user_text}\n\n"
            "Write one paragraph (80–120 words) explaining naturally why moving "
            "from the current state to the next state makes sense. Avoid instructions or meta talk."
        )
        j3 = _safe_call(explain_prompt)
        explanation = _extract_text(j3)

        # Retry fallback if empty or meta
        if not explanation or len(explanation.split()) < 40:
            retry_prompt = (
                f"Summarize why transitioning from '{current}' to '{next_state}' is logical, "
                f"based on this user text:\n{user_text}\n\nReturn one clear paragraph (80–120 words)."
            )
            j4 = _safe_call(retry_prompt)
            explanation = _extract_text(j4)

        explanation = polish_output(explanation)

        # ✅ Final result
        result.update({
            "domain": parsed.get("domain"),
            "states": states,
            "current_state": current,
            "predicted_next_state": next_state,
            "explanation": explanation or "No explanation generated."
        })

    except Exception as e:
        result["error"] = f"Universal predictor failed: {e}\n{traceback.format_exc()}"

    return result
