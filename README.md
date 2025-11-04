

# Universal Next-State Predictor

Live Demo: [https://markov-predictor.streamlit.app/](https://markov-predictor.streamlit.app/)

---

## Overview

Universal Next-State Predictor is a Streamlit-based web application that blends **Markov chain modeling** with **Groq open-source large language models** to predict the most likely next state, milestone, or action for any kind of input.

It can analyze résumés, personal statements, financial plans, learning goals, or any natural-language text, infer your current situation, predict what comes next, and provide a concise, human-like explanation paragraph.

---

## Key Features

* Domain-independent prediction across career, finance, education, health, or general topics
* Hybrid reasoning combining probabilistic Markov transitions and LLM-driven semantic understanding
* Automatic state generation and context detection by Groq OSS models
* Adaptive handling of both forward and backward transitions (promotion or pivot)
* Polished paragraph output written in natural, fluent language
* Streamlit interface with PDF upload, text input, and state progression visualization

---

## How It Works

1. The user uploads a PDF (for example, a résumé) or types a text prompt.
2. The LLM analyzes the content and returns a JSON object that includes:

   * The detected domain (career, finance, learning, etc.)
   * A progressive list of possible states
   * The user’s inferred current state
3. The Markov model predicts the next probable state using transition logic and intent cues.
4. The LLM then generates a one-paragraph explanation (80–120 words) describing why this transition makes sense.
5. The system refines the text to improve readability and tone before displaying it.

---

## Example Outputs

### Career Example

**Input**
"I am an AI engineer with 3 years of experience in NLP and automation."

**Prediction**

* Domain: career
* Current State: Intermediate AI Engineer
* Predicted Next State: Senior AI Engineer

**Explanation**
This professional has developed strong applied-AI skills through three years of focused NLP and automation work. The logical next step is a senior-level role that emphasizes leadership, architecture design, and strategic model optimization. This transition reflects natural career progression from technical contribution toward project ownership and decision-making responsibilities.

---

### Finance Example

**Input**
"I earned $2000 per month trading since October 2025. What will I earn in the first quarter of 2026?"

**Prediction**

* Domain: finance
* Current State: Analyze historical performance
* Predicted Next State: Build forecast model

**Explanation**
With a solid record of recent trading income, the next logical step is to build a forecast model that predicts future earnings based on your historical performance. Incorporating new market data and strategy adjustments will provide a data-backed estimate of your income for early 2026 and help refine future investment decisions.

---

### Backward or Learning-Oriented Example

**Input**
"After freelancing in AI projects, I want to refine my theoretical understanding and work with research teams."

**Prediction**

* Domain: career
* Current State: Freelance AI Engineer
* Predicted Next State: Student or Intern

**Explanation**
Although this transition appears to move backward, it represents a strategic learning pivot. After practical experience as a freelance engineer, the individual aims to strengthen academic depth and research exposure. This type of shift is common among professionals who seek structured mentorship, advanced theoretical grounding, and long-term specialization. The model recognizes this intent and interprets it as a progression in knowledge and direction, not a regression in skill or capability.

---

## Why Backward Transitions Occur

The model sometimes predicts a "backward" move (for example, from Senior Engineer to Student) when the text emphasizes **learning, research, or exploration** over advancement or leadership.
This behavior is intentional and reflects real-world scenarios such as:

* Professionals taking sabbaticals for academic growth
* Industry experts pursuing PhD or research internships
* Entrepreneurs transitioning back to structured learning environments
* Career changers acquiring foundational skills in a new field

In such cases, the Markov component interprets intent words like *pursue, learn, study, research, explore* as a temporary regression for long-term advancement.
The LLM then generates an explanation clarifying why that backward transition still represents meaningful forward progress.

---

## Tech Stack

* Python 3.13+
* Streamlit
* PyMuPDF for PDF text extraction
* Groq OSS LLMs (`openai/gpt-oss-20b`)
* Adaptive Markov chain logic for state transitions

---

## Setup and Run

Clone the repository and install dependencies:

```
git clone https://github.com/yourusername/universal-predictor.git
cd universal-predictor
pip install -r requirements.txt
```

Create a `.streamlit/secrets.toml` file and add your Groq API key:

```
GROQ_API_KEY = "your_groq_api_key_here"
```

Run the application:

```
streamlit run app.py
```

Open the app in your browser at the default address (usually [http://localhost:8501](http://localhost:8501)).

---

## Project Structure

```
app.py
utils/
  ├── llm_utils.py        (Hybrid predictor and explanation generator)
  ├── markov_model.py     (Adaptive Markov logic with intent detection)
  ├── pdf_utils.py        (PDF text extraction)
```

---

## Author

Developed by Vivek Mohape


---

Would you like me to also include a **“Customization Guide”** section at the end explaining how to tweak bias (e.g., to always predict forward for experienced professionals)? This would be useful if you plan to open-source or publish it on GitHub.
