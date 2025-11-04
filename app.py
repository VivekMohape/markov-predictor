import streamlit as st
from utils.llm_utils import universal_predictor
from utils.pdf_utils import extract_text_from_pdf

st.set_page_config(page_title="Markov Predictor", layout="wide")

st.title("Markov Next-State Predictor")
st.markdown("""
This app blends **Markov reasoning** with **Groq OSS LLMs** to predict your **next likely milestone**  
from any input: a résumé, financial plan, learning goal, or health update.

---

### 🧩 How to Use
1. Upload a **PDF** (like a résumé or report) or enter text below.  
2. Click **Predict Next State**.  
3. See:
   - Domain detection  
   - Current vs. predicted next state  
   - State progression map  
   - AI-generated polished summary  
---
""")

col1, col2 = st.columns([1, 1])
uploaded_file = col1.file_uploader("📎 Upload a PDF", type=["pdf"])
user_text = col2.text_area("💬 Or type/paste text here", height=200, placeholder="E.g. I earn $550/month... or '3 years AI experience'")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    if text:
        st.success("✅ PDF processed successfully!")
        with st.expander("📄 Extracted Text"):
            st.text_area("Document Text", text[:2000], height=300)
        user_text = text

if not user_text:
    st.info("🪄 Upload a PDF or type text to begin.")
    st.stop()

if st.button("🔮 Predict Next State", type="primary", use_container_width=True):
    with st.spinner("Analyzing input and generating prediction..."):
        result = universal_predictor(user_text)

    st.markdown("---")
    if result.get("error"):
        st.error(result["error"])
    else:
        st.subheader("🧠 Prediction Summary")
        st.write(f"**🗂️ Domain:** {result.get('domain', 'Unknown')}")
        st.write(f"**🔹 Current State:** {result.get('current_state', '—')}")
        st.write(f"**🔸 Predicted Next State:** {result.get('predicted_next_state', '—')}")

        if result.get("states"):
            with st.expander("📊 State Progression Map"):
                for i, s in enumerate(result["states"], 1):
                    st.markdown(f"{i}. {s}")

        st.markdown("### 💬 AI-Generated Explanation")
        st.write(result.get("explanation", "No explanation generated."))

        st.markdown("---")
        st.caption("✨ Built by Vivek Mohape | Powered by Groq OSS Models + Markov Reasoning")

else:
    st.caption("Tip: Works for résumés, finance plans, and learning goals too!")
