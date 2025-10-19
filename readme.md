#  Markov Chain Next-Action Predictor

A modern Streamlit web app that uses a **Markov chain** to predict a user's next likely action from an uploaded PDF (like a résumé or document) — enhanced with **Groq OSS LLMs** for natural-language explanation.

---

##  How to Use

1. **Upload a PDF file** (like your résumé, report, or article).
2. Click ** Predict Next Action**.
3. The app will show:
   - Extracted keywords
   - Current inferred state
   - Predicted next likely action
   - Bar chart of transition probabilities
   - AI-generated summary paragraph from Groq OSS model

---

##  Tech Stack

-  Python 3.13+
-  Streamlit
-  PyMuPDF
-  Groq API (LLM inference using open-source models like `llama3-8b-8192`)

---

## 🔐 Setup Instructions

1. **Clone the repo:**
   ```bash
   git clone https://github.com/VivekMohape/markov-predictor.git
   cd markov-predictor
