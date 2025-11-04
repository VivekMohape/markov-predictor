
# Universal Next-State Predictor

Live Demo: [https://markov-predictor.streamlit.app/](https://markov-predictor.streamlit.app/)

---

## Overview

Universal Next-State Predictor is a Streamlit web application that uses a hybrid approach of Markov chain modeling and Groq open-source LLMs to predict the most likely next action, role, or milestone from any input text or uploaded PDF file.

The system understands the context (career, finance, learning, health, or general reasoning), identifies the user’s current state, predicts the next logical state, and generates a natural-language explanation paragraph describing why that transition makes sense.

---

## Key Features

* Domain-independent prediction for career, finance, education, or personal progress.
* Combination of deterministic Markov transitions and LLM-based semantic reasoning.
* Automatic state generation and context detection using Groq OSS LLMs.
* Clean and concise paragraph explanations refined through an automated polishing step.
* Streamlit interface for file uploads, text input, and visual state progression display.

---

## How It Works

1. The user uploads a PDF (such as a résumé) or types text directly.
2. The LLM analyzes the input and returns:

   * The relevant domain (for example, career or finance)
   * A logical list of progressive states
   * The user’s current state
3. The Markov logic predicts the most likely next state based on probabilities and intent detection.
4. The LLM then generates a short, coherent paragraph explaining why this next step makes sense.
5. The system refines and displays the explanation with clear grammar and flow.

---

## Example Outputs

**Career example**

Input:
“I am an AI engineer with 3 years of experience in NLP and automation.”

Prediction:

* Domain: career
* Current state: Intermediate AI Engineer
* Predicted next state: Senior AI Engineer

Explanation:
This professional has developed strong applied-AI skills over three years of focused NLP and automation work. The logical next step is a senior-level role that emphasizes leadership, architecture design, and strategic model optimization within production-scale systems.

---

**Finance example**

Input:
“I earned $2000 per month trading since October 2025. What will I earn in the first quarter of 2026?”

Prediction:

* Domain: finance
* Current state: Analyze historical performance
* Predicted next state: Build forecast model

Explanation:
With a solid record of recent trading income, the next logical step is to transform that data into predictive insight. By analyzing past performance and incorporating current market conditions, you can project realistic earnings for the first quarter of 2026.

---

## Tech Stack

* Python 3.13+
* Streamlit
* PyMuPDF for PDF text extraction
* Groq API (LLM inference with `openai/gpt-oss-20b`)
* Markov chain logic for state transitions

---

## Setup and Run

Clone the repository and install dependencies:

```
git clone https://github.com/yourusername/universal-predictor.git
cd universal-predictor
pip install -r requirements.txt
```

Create a `.streamlit/secrets.toml` file with your Groq API key:

```
GROQ_API_KEY = "your_groq_api_key_here"
```

Run the app:

```
streamlit run app.py
```

---

## Project Structure

```
app.py
utils/
  ├── llm_utils.py        (Hybrid predictor and explanation generator)
  ├── markov_model.py     (Adaptive Markov logic)
  ├── pdf_utils.py        (PDF text extraction)
```




