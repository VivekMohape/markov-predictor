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
# Helper Functions
# -------------------------------------------------------------
def _safe_call(prompt, timeout=60):
    """Send a request to Groq API safely."""
    key = st.secrets.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("⚠️ Missing GROQ_API_KEY in Streamlit secrets.")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": MODEL, "input": prompt}
    r = requests.post(GROQ_URL, headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _extract_text(resp_json):
    """Extracts clean main text and trims meta reasoning."""
    if not resp_json:
        return ""
    text = ""
    if "output_text" in resp_json and resp_json["output_text"]:
        text = resp_json["output_text"].strip()
    elif "output" in resp_json and resp_json["output"]:
        parts = []
        for item in resp_json["output"]:
            for c in item.get("content", []):
                if "text" in c:
                    parts.append(c["text"])
        text = "\n".join(parts).strip()
    else:
        text = str(resp_json)

    # Remove meta lines like "We need to...", "Let's..."
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if re.match(r"^\s*(we need|let'?s|so we|our task|we should|the goal|the user asks)", line.strip(), re.I):
            continue
        cleaned.append(line)
    text = " ".join(cleaned).strip()

    # Keep only last paragraph
    paragraphs = re.split(r"\n{2,}", text)
    if len(paragraphs) > 1:
        text = paragraphs[-1].strip()
    return text


def _try_parse_json(raw_text: str):
    """Extracts last valid JSON block from LLM output."""
    import json, re
    if not raw_text or not isinstance(raw_text, str):
        return None

    candidates = re.findall(r"(\{.*?\}|\[.*?\])", raw_text, re.DOTALL)
    if not candidates:
        return None

    for block in reversed(candidates):
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict) and "states" in parsed and "current_state" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def polish_output(text):
    """
    Final version: robust polishing with self-healing fallback.
    Prevents meta/instructional echoes like '97 words. Good.'
    and ensures at least one natural paragraph is returned.
    """
    if not text or len(text.strip()) < 40:
        return text  # skip trivial content

    try:
        key = st.secrets.get("GROQ_API_KEY")
        if not key:
            return text

        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        # ✅ Use neutral, non-instructional phrasing to avoid echoing
        body = {
            "model": MODEL,
            "input": (
                "Rewrite the following paragraph slightly to improve flow, grammar, and tone. "
                "Keep the same meaning and style. "
                "Output ONLY the improved paragraph.\n\n"
                f"{text}"
            ),
        }

        r = requests.post(GROQ_URL, headers=headers, json=body, timeout=40)
        r.raise_for_status()
        data = r.json()

        refined = ""
        if "output_text" in data and data["output_text"]:
            refined = data["output_text"].strip()
        elif "output" in data and data["output"]:
            refined = data["output"][0]["content"][0]["text"].strip()

        # 🧹 Remove meta/instruction echoes
        if not refined or re.match(
            r"^(check|ensure|return only|good|ok|fine|provide only|97 words)", 
            refined.lower()
        ):
            # fallback retry with minimal prompt
            fallback = {
                "model": MODEL,
                "input": (
                    f"Polish grammar and punctuation. Output only paragraph:\n{text}"
                )
            }
            r2 = requests.post(GROQ_URL, headers=headers, json=fallback, timeout=30)
            r2.raise_for_status()
            data2 = r2.json()
            if "output_text" in data2 and data2["output_text"]:
                refined = data2["output_text"].strip()
            elif "output" in data2 and data2["output"]:
                refined = data2["output"][0]["content"][0]["text"].strip()
            else:
                refined = text

        # If still not a paragraph, fallback entirely
        if not refined or len(refined.split()) < 40:
            refined = text

        # Keep last coherent block only
        refined = refined.strip().split("\n\n")[-1].strip()
        return refined

    except Exception as e:
        st.warning(f"⚠️ Polish skipped due to error: {e}")
        return text


# -------------------------------------------------------------
# Universal Predictor
# -------------------------------------------------------------
def universal_predictor(user_text):
    """Universal Markov + LLM hybrid predictor."""
    result = {"input": user_text}
    try:
        # Step 1: Ask Groq for domain, states, and current state
        prompt = (
            "Analyze this input and respond strictly in JSON format with keys:\n"
            "'domain': short string (career, finance, learning, health, etc.),\n"
            "'states': ordered list (4–8 progressive states relevant to the context),\n"
            "'current_state': one of the states that best matches current situation.\n\n"
            "Return ONLY JSON. The first character must be '{'.\n\n"
            f"Input:\n{user_text}"
        )
        j = _safe_call(prompt)
        raw = _extract_text(j)
        parsed = _try_parse_json(raw)

        # Retry with repair if malformed
        if not parsed:
            repair_prompt = f"The following output is invalid JSON:\n{raw}\n\nReformat strictly as JSON."
            j2 = _safe_call(repair_prompt)
            raw2 = _extract_text(j2)
            parsed = _try_parse_json(raw2)

        if not parsed:
            raise ValueError(f"Groq returned non-JSON or empty output:\n{raw}")

        states = parsed.get("states", [])
        current = parsed.get("current_state")
        if not states or not current:
            raise ValueError(f"LLM JSON missing required fields:\n{parsed}")

        # Step 2: Markov prediction
        matrix = markov_model.build_uniform_matrix(states)
        pred = markov_model.predict_next_state(states, matrix, current, text=user_text, deterministic=True)
        if "error" in pred:
            raise RuntimeError(pred["error"])
        next_state = pred["next_state"]

        # Step 3: Ask for explanation
        explain_prompt = (
            f"Domain: {parsed.get('domain')}\n"
            f"Current state: {current}\nNext state: {next_state}\n\n"
            f"User text:\n{user_text}\n\n"
            "Write one clear paragraph (80–120 words) explaining naturally why "
            "moving from the current state to the next state makes sense. "
            "Avoid instructions or self-reference."
        )
        j3 = _safe_call(explain_prompt)
        explanation = _extract_text(j3)

        # Step 4: Optional polish
        explanation = polish_output(explanation)

        # Final result
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
