import streamlit as st
from utils.llm_utils import universal_predictor
from utils.pdf_utils import extract_text_from_pdf

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Universal Next-State Predictor 🤖", page_icon="🔮", layout="wide")

st.title("🔮 Universal Next-State Predictor")
st.markdown("""
A **domain-agnostic reasoning app** powered by **Markov Chains + Groq OSS LLMs**.  
It dynamically understands your input — whether it’s a **résumé**, **financial query**, **learning plan**, or **any free-form text** —  
and predicts your **next likely state or milestone** with a natural-language explanation.

---

### 🧩 How It Works
1. You **upload a PDF** or **type/paste text**.
2. The AI:
   - Classifies your domain automatically.
   - Generates meaningful progression states.
   - Uses a **Markov model** to predict the next probable state.
   - Explains the reasoning naturally.

---
""")

# -------------------------------
# Input Section
# -------------------------------
col1, col2 = st.columns([1, 1])
uploaded_file = col1.file_uploader("📎 Upload a PDF", type=["pdf"], help="Upload a résumé or any text-based PDF.")
user_text = col2.text_area("💬 Or type/paste your text here", height=200, placeholder="E.g. I earn $550 monthly... or '3 years AI engineer experience'")

if uploaded_file:
    extracted_text = extract_text_from_pdf(uploaded_file)
    if extracted_text:
        st.success("✅ PDF processed successfully!")
        with st.expander("📄 View Extracted Text"):
            st.text_area("Extracted Text", extracted_text[:2000], height=300)
        user_text = extracted_text
    else:
        st.error("❌ Failed to extract text from PDF.")

if not user_text:
    st.info("🪄 Upload a PDF or type text above to begin.")
    st.stop()

# -------------------------------
# Run Prediction
# -------------------------------
if st.button("🔮 Predict Next State", type="primary", use_container_width=True):
    with st.spinner("Analyzing input and generating prediction..."):
        result = universal_predictor(user_text)

    st.markdown("---")
    if result.get("error"):
        st.error(f"❌ {result['error']}")
    else:
        st.subheader("🧠 Prediction Summary")
        st.write(f"**🗂️ Domain:** {result.get('domain', 'Unknown')}")
        st.write(f"**🔹 Current State:** {result.get('current_state', '—')}")
        st.write(f"**🔸 Predicted Next State:** {result.get('predicted_next_state', '—')}")

        if result.get("states"):
            with st.expander("📊 State Progression Map"):
                st.write(result["states"])

        st.markdown("### 💬 AI-Generated Explanation")
        st.write(result.get("explanation", "No summary generated."))

        st.markdown("---")
        st.caption("✨ Powered by Groq OSS Models & Markov reasoning. Designed by Vivek Mohape.")

else:
    st.markdown("---")
    st.caption("Tip: Try a résumé, financial plan, or learning goal to see domain-aware predictions.")
