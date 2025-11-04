import streamlit as st
import requests

def generate_description(current_state, next_state, keywords):
    """Generate human-readable explanation using Groq OSS model (responses.create style)."""
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
    if not GROQ_API_KEY:
        st.error("⚠️ Missing GROQ_API_KEY in Streamlit secrets.")
        return "AI summary not available."

    # Compose prompt text
    prompt = (
    f"Current state: {current_state}\n"
    f"Predicted next action: {next_state}\n"
    f"Detected keywords: {', '.join([k for k, v in keywords.items() if v]) or 'None'}\n\n"
    "Write one clear, direct paragraph (~100 words) explaining what the predicted next action means for the user. "
    "Do not describe your task or say what you will do—just give the explanation naturally and conversationally."
)


    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # ✅ Using Groq’s new responses.create API
    body = {
        "model": "openai/gpt-oss-20b",  # updated per your reference
        "input": prompt
    }

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/responses",
            headers=headers,
            json=body,
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()

        # ✅ Extract final text output from responses.create structure
        if "output" in data and len(data["output"]) > 0:
            paragraph = data["output"][0]["content"][0]["text"]
        elif "output_text" in data:
            paragraph = data["output_text"]
        else:
            paragraph = "No AI output received."

        return paragraph.strip()

    except Exception as e:
        st.error(f"❌ Groq request failed: {e}")
        return "AI summary not available."
