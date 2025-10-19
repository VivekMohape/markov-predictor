import streamlit as st
import requests

def generate_description(current_state, next_state, keywords):
    """Generate human-readable explanation using Groq OSS model."""
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
    if not GROQ_API_KEY:
        st.error("⚠️ Missing GROQ_API_KEY in Streamlit secrets.")
        return "AI summary not available."

    prompt = (
        "You are an assistant summarizing Markov-based user predictions.\n"
        f"Current state: {current_state}\n"
        f"Predicted next action: {next_state}\n"
        f"Detected keywords: {', '.join([k for k, v in keywords.items() if v]) or 'None'}\n\n"
        "Write a short, clear paragraph (around 100 words) explaining what this next action means for the user."
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Updated body for text-completion model (not chat)
    body = {
        "model": "openai/gpt-oss-120b",
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 150
    }

    try:
        resp = requests.post("https://api.groq.com/v1/completions", headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        paragraph = data["choices"][0]["text"]  # use 'text' for completion models
        return paragraph.strip()
    except Exception as e:
        st.error(f"❌ Groq request failed: {e}")
        return "AI summary not available."
