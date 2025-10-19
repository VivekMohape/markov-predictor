import streamlit as st
import requests

def generate_description(current_state, next_state, keywords):
    """Generate human-readable explanation using Groq OSS model."""
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
    if not GROQ_API_KEY:
        st.error("⚠️ Missing GROQ_API_KEY in Streamlit secrets.")
        return ""

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

    body = {
        "model": "llama3-8b-8192",  # you can switch to mixtral-8x7b-32768 if desired
        "messages": [
            {"role": "system", "content": "You are a friendly assistant writing short insights."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }

    try:
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        paragraph = data["choices"][0]["message"]["content"]
        return paragraph.strip()
    except Exception as e:
        st.error(f"❌ Groq request failed: {e}")
        return ""
