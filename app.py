import streamlit as st
import pandas as pd
from utils.pdf_utils import extract_text_from_pdf
from utils.markov_model import extract_keywords, predict_next_action
from utils.llm_utils import generate_description

# Page Setup
st.set_page_config(page_title="Markov Predictor", page_icon="🤖", layout="wide")

st.title(" Markov Chain Next-Action Predictor")
st.markdown("""
Welcome to the **Markov Chain Predictor App** powered by **Groq AI** and open-source LLMs.
This app reads your uploaded PDF (like a résumé or profile), identifies context,  
and predicts your most likely *next action* using a Markov model + natural language summary.

---
### 🪄 How to Use
1. **Upload a PDF file** (e.g., résumé, project report, or document with text).  
2. Click ** Predict Next Action**.  
3. View:
   - Extracted keywords from your document  
   - Current inferred state  
   - Predicted next action  
   - Transition probability chart  
   - AI-generated explanation paragraph  
---
""")

uploaded_file = st.file_uploader("📎 Upload your PDF", type=["pdf"], help="Upload any text-based PDF file.")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    if text:
        st.success("✅ PDF processed successfully!")
        with st.expander("📄 Preview Extracted Text"):
            st.text_area("Document Text", text[:1500] + "...", height=250)

        keywords = extract_keywords(text)
        st.write("### 🔍 Detected Keywords")
        st.write({k: v for k, v in keywords.items() if v} or "No major keywords detected.")

        st.markdown("---")
        st.subheader("🎯 Prediction")
        if st.button(" Predict Next Action", use_container_width=True, type="primary"):
            with st.spinner("Running Markov chain prediction..."):
                current_state, next_state, probs, states = predict_next_action(keywords)

            st.write(f"**Current State:** {current_state}")
            st.write(f"**Predicted Next Action:** {next_state}")

            df = pd.DataFrame({"Next State": states, "Probability": probs})
            st.bar_chart(df.set_index("Next State"))

            with st.spinner("Generating AI explanation..."):
                paragraph = generate_description(current_state, next_state, keywords)

            if paragraph:
                st.markdown("### 💬 AI-Generated Summary")
                st.write(paragraph)

                with st.expander("💡 Example Output"):
                    st.markdown("""
                    > **Example:**  
                    > Current State: Ask for Technical Explanation  
                    > Predicted Next Action: Request Code Example  
                    > _"Based on your technical background and AI-related focus, you're likely to request a working code demonstration next.  
                    This step helps transform conceptual understanding into implementation."_  
                    """)

else:
    st.info("Please upload a PDF document to begin the prediction.")

st.markdown("---")
st.caption("Built with Markov Chains by Vivek Mohape.")
