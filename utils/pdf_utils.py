import fitz  # PyMuPDF
import streamlit as st

def extract_text_from_pdf(pdf_file):
    """Extract plain text from uploaded PDF file."""
    text = ""
    try:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
    return text.strip()
